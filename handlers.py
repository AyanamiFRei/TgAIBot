"""Хендлеры aiogram: текст, фото, стикеры, /persona, /start.

Сквозной путь:
  1. Считываем из БД последние N сообщений диалога (с ограничением по символам).
  2. Формируем messages = [system] + история + текущий user-ход.
  3. Зовём Cloudflare Workers AI, получаем текст.
  4. Дробим по <<MSG>> и шлём отдельными сообщениями с задержкой.
     Если в ответе есть стикер-маркеры — отправляем стикеры через send_sticker.
  5. Сохраняем user-ход и assistant-реплики в БД.

/persona с файлом переписки — теперь спрашивает, кого из собеседников брать
за персону, если в экспорте несколько человек (см. cmd_persona_document и
on_speaker_chosen ниже). Один спикер — без вопроса. Спикеров нет (plain txt
без авторской разметки) — откат на старое поведение «модель сама выберет».
"""
from __future__ import annotations

import asyncio
import datetime
import logging
import re
from typing import TYPE_CHECKING

from aiogram import Bot, Dispatcher, F
from aiogram.filters import Command, StateFilter
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder

import db
import media
import persona as persona_mod
import prompts
import stickers as stickers_module
from config import Config
from cloudflare_client import CloudflareClient, CloudflareError

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FSM-состояние выбора спикера для persona
# ---------------------------------------------------------------------------

class PersonaChoice(StatesGroup):
    """Состояние «ждём, пока юзер ткнёт inline-кнопку выбора спикера».

    В FSM Storage (по умолчанию MemoryStorage, см. bot.py) лежат:
      persona_raw_bytes : bytes   — сырой байты файла переписки
      persona_speakers  : list[str] — список имён собеседников в порядке
                                       первого появления (тот же порядок, что
                                       и в кнопках клавиатуры).

    MemoryStorage хватает для личного бота: на рестарте воркера данные
    теряются — юзер просто переотправит файл. На проде с Cloudflare Workers
    лучше RedisStorage, чтобы выбор пережил рестарт.
    """
    waiting_speaker = State()


# Префикс для callback_data: "persona:<index>". Имя спикера НЕ кодируем в
# callback_data — оно может быть длинным/с эмодзи/со скобками (как «Диана
# Белоусова (my love)»). Вместо этого передаём индекс, а сами имена достаём
# из FSM Storage, куда сохранили список при отправке клавиатуры.
_PERSONA_CB_PREFIX = "persona:"
# Специальный индекс для fallback-кнопки «самый активный» — старое поведение.
_PERSONA_LEGACY_IDX = -1


# ---------------------------------------------------------------------------
# Вспомогательное
# ---------------------------------------------------------------------------


async def run_db(fn, *args, **kwargs):
    """Запускаем синхронную sqlite-операцию в фоне, не блокируя event loop."""
    try:
        to_thread = asyncio.to_thread
    except AttributeError:  # Python 3.8: to_thread ещё нет
        loop = asyncio.get_event_loop()
        import functools
        return await loop.run_in_executor(
            None, functools.partial(fn, *args, **kwargs))
    return await to_thread(fn, *args, **kwargs)


# ---------------------------------------------------------------------------
# Сборка запроса и потоковая отправка ответа
# ---------------------------------------------------------------------------

# Быстрый путь «запомни: X»: явная просьба запомнить факт. Проверено на живых
# строках: «запомни: я вегетарианец», «Запомни, что я делаю игру», «запомни - ...».
_REMEMBER_RE = re.compile(r"^\s*запомни[,:-]?\s+(.+)$", re.IGNORECASE)

def _time_header(cfg: Config, user_id: int, conn: "sqlite3.Connection") -> str:
    """Короткая плашка «сейчас у собеседника HH:MM, день недели» + «сколько
    прошло с его последнего сообщения». Уходит В КОНЦЕ последнего user-хода
    (дописывается к контенту, не в system) — ради кеша промпта: в system она
    менялась бы от запроса к запросу и сбрасывала префикс на каждом пинке
    (см. prompts._proactive_block — тот же принцип).

    Без неё модель не знает реального времени (VPS в UTC, юзер может быть
    в другом поясе): путает утро с вечером («какие пары, три часа ночи?!») и
    не чувствует, что между сообщениями прошли сутки.

    Время — в зоне cfg.timezone (TIMEZONE в .env); невалидное имя — UTC.
    Возвращает плашку всегда (даже «только что» — время суток всё равно важно).
    """
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(cfg.timezone)
    except Exception:  # noqa: BLE001 — нет zoneinfo/невалидное имя
        from datetime import timezone as _tzmod
        tz = _tzmod.utc

    now = datetime.datetime.now(tz)
    dow = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"][now.weekday()]
    months = ["января", "февраля", "марта", "апреля", "мая", "июня",
              "июля", "августа", "сентября", "октября", "ноября", "декабря"]
    parts = [
        f"[системная плашка: сейчас у собеседника {now.hour:02d}:{now.minute:02d}, "
        f"{dow}, {now.day} {months[now.month - 1]} {now.year} — учитывай время "
        "суток и дату в ответе]",
    ]

    # Прошло с последнего сообщения собеседника — чтобы чувствовать паузу/сутки.
    try:
        row = conn.execute(
            "SELECT created_at FROM messages WHERE user_id = ? AND role = 'user' "
            "ORDER BY id DESC LIMIT 1",
            (user_id,),
        ).fetchone()
    except Exception:  # noqa: BLE001
        row = None
    if row is not None:
        gap_h = (now.timestamp() - row[0]) / 3600.0
        if gap_h >= 0.17:  # >10 минут — уже не «только что»
            if gap_h >= 24:
                gap_line = f"с последнего его сообщения прошло больше суток ({gap_h:.0f} ч)"
            elif gap_h >= 1:
                gap_line = f"с последнего его сообщения прошло {gap_h:.1f} ч"
            else:
                gap_line = f"с последнего его сообщения прошло {gap_h * 60:.0f} минут"
            parts.append(
                f"[системная плашка: {gap_line} — учитывай паузу, не делай вид, "
                "что разговор не прерывался]"
            )
    return " ".join(parts)


async def _build_messages(
    message: Message,
    cfg: Config,
    conn: "sqlite3.Connection",
    stickers: "stickers_module.StickerStore | None",
    current_user_content: list[dict] | str,
) -> list[dict]:
    """Собираем [system] + история + текущий user-ход. Саму модель не зовём."""
    history = await run_db(
        db.get_recent_budget, conn, message.from_user.id,
        cfg.history_chars, cfg.history_window,
    )

    messages: list[dict] = []
    persona = await run_db(db.get_persona, conn, message.from_user.id)
    sticker_vibes = stickers.top_vibes(7) if stickers and not stickers.is_empty() else None
    messages.append({
        "role": "system",
        "content": prompts.build_system(persona.summary if persona else None, sticker_vibes),
    })

    for row in history:
        role = row["role"]
        content = row["content"]
        # Память (автокомпакт старой истории) — отдельным system-блоком «Что было
        # раньше», не user/assistant-репликой: модель не должна путать её с ходами.
        if role == "memory":
            messages.append({
                "role": "system",
                "content": (
                    "Что было раньше (краткая память о старой переписке, которая уже "
                    "не входит в диалог выше):\n" + content +
                    "\nЭто реально было — опирайся на это, если собеседник ссылается "
                    "на старые темы; в видимой истории их уже нет."
                ),
            })
        # Картинки старых реплик не переотправляем — только текст/плашку.
        elif role == "user":
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "assistant", "content": content})

    messages.append({"role": "user", "content": current_user_content})
    # Плашка времени — допиской в хвост последнего user-хода (кеш-безопасно:
    # system байт-идентичен, меняется только маленький хвост, см. _time_header).
    if isinstance(current_user_content, str):
        header = _time_header(cfg, message.from_user.id, conn)
        if header:
            messages[-1]["content"] += "\n\n" + header
    return messages


# Маркеры, которые ловим в стриме. Большие — целиком; маленькие — целиком тоже.
# Стикер-маркер может прийти разрезанным между чанками, поэтому ищем только ГОТОВЫЙ
# маркер (regex с явными границами). Частичный «обрубок» в конце буфера оставляем
# до прихода следующих чанков.
#
# Модель иногда (несмотря на инструкцию в system-промпте использовать английское
# слово "sticker") пишет русское "стикер" — распознаём оба варианта, иначе маркер
# утекает в чат как обычный текст (см. баг: "[стикер: 😾]" пришёл юзеру буквально).
_STICKER_MARKER_RE = re.compile(
    r"<<\s*(?:STICKER|СТИКЕР)\s*:\s*([^>]+?)\s*>>"
    r"|\[\s*(?:sticker|стикер)\s*:\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)
# Префиксы частичных маркеров — если буфер кончается чем-то подобным, ждём ещё.
# Это нужно, чтобы не отправить текст, который на самом деле — начало стикер-маркера.
# Включает и кириллический вариант "[стикер" / "<<СТИКЕР", раз модель его тоже пишет.
_PARTIAL_PREFIXES = (
    "<<", "[s", "[S", "<<S", "<<ST", "<<STI",
    "[с", "[С", "<<С", "<<СТ", "<<СТИ",
)


def _vibe_from_match(m: "re.Match[str]") -> str:
    raw = (m.group(1) or m.group(2) or "").strip().lower()
    # Модель иногда вставляет сам эмодзи стикера вместо названия вибы (например
    # "[стикер: 😾]" вместо "[sticker: недовольство]"). Если raw — известный emoji
    # из карты стикеров, переводим его в вибу, а не ищем несуществующую "вибу 😾".
    return stickers_module.EMOJI_TO_VIBE.get(raw, raw)


def _ends_with_partial_marker(buf: str) -> bool:
    """Если конец buf — потенциальное начало стикер-маркера, ждём больше данных."""
    if not buf:
        return False
    tail_len = min(len(buf), 8)  # маркер <<STICKER…< = до ~10 символов начала
    tail = buf[-tail_len:]
    return any(tail.endswith(p) for p in _PARTIAL_PREFIXES)


async def _stream_reply(
    bot: Bot,
    chat_id: int,
    ai: CloudflareClient,
    messages: list[dict],
    cfg: Config,
    stickers: "stickers_module.StickerStore | None",
    *,
    emit_typing: bool = True,
    error_note: str | None = "сеть что-то барахлит, секунду 🌚",
) -> list[str]:
    """Стримим ответ модели и отправляем куски в чат ПО МЕРЕ прихода.

    Логика границ (режем «единый ответ» на отдельные сообщения):
      - стикер-маркер [sticker: x] / <<STICKER:x>>: при нахождении целиком —
        отправляем стикер сразу, маркер вырезаем из буфера;
      - <<MSG>> и пустая строка (\\n\\n): граница между сообщениями — всё ДО неё
        отправляем отдельным сообщением, после неё остаётся в буфере.

    Задержки между сообщениями НЕТ — шлём сразу, как только накопилось, чтобы юзер
    видел живую «печать». typing-продлялка крутится параллельно стриму.

    Опционально (keyword-only):
      emit_typing=False — НЕ светить стартовый «печатает…» до прихода первого
        чанка. Имеет смысл для proactive-пинка: юзер не писал ничего, индикатор
        «печатает» посреди тишины выглядит как глюк. Перед каждым реальным
        сообщением typing всё равно ставится в _drain_boundaries.
      error_note=None — не слать пользователю «сеть барахлит» при обрыве стрима
        ДО отправки чего-либо. Для proactive такой фит на ровном месте — спам:
        бот сам вылез с извинением, никого не спросят.

    Возвращает список отправленных ТЕКСТОВЫХ чанков (для сохранения в БД). Стикеры
    в БД не сохраняются — модель в следующих ходах видит только свой текст.
    """
    if emit_typing:
        await bot.send_chat_action(chat_id, "typing")
        typing_task = asyncio.create_task(_keep_typing(bot, chat_id))
    else:
        # proactive-пинк: не светим «печатает…» до прихода первого чанка. Перед
        # каждым реальным сообщением typing поставит _drain_boundaries — для UX
        # достаточно. Если стрим ничего не даст — ничего и не мигнёт в чат.
        typing_task = asyncio.ensure_future(_noop())

    sent: list[str] = []  # отправленные текстовые сообщения (для БД)
    # buf держим в одноэлементном списке, чтобы хелпер _drain_boundaries мог
    # его мутировать (строки — immutable, простой += наружу не отразится).
    buf_holder: list[str] = [""]

    try:
        async for chunk in ai.chat_stream(
            messages,
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            reasoning_effort=getattr(cfg, "reasoning_effort", None) or None,
        ):
            buf_holder[0] += chunk
            await _drain_boundaries(bot, chat_id, buf_holder, sent, stickers)

        # Стрим закончился — досыпаем хвост (то же самое: в нём могут быть
        # отложенные стикер-маркеры и <<MSG>>/\n\n, + чистый текст на конце).
        await _drain_boundaries(bot, chat_id, buf_holder, sent, stickers)
        # _drain_boundaries может оставить «обрубок» в buf, если он похож на
        # начало стикер-маркера (хочется дополнить следующими чанками). Но стрим
        # уже закончился — дополнения не будет, шлём остаток как обычный текст.
        tail = buf_holder[0].strip()
        if tail:
            await bot.send_chat_action(chat_id, "typing")
            safe = tail[:4000]
            await bot.send_message(chat_id, safe)
            sent.append(safe)
    except CloudflareError as exc:
        # Если уже что-то отправили — не спамим «сеть барахлит» вдогонку; просто лог.
        if sent:
            logger.warning("стрим оборвался после %d сообщений: %s", len(sent), exc)
        else:
            logger.warning("stream error: %s", exc)
            if error_note:
                try:
                    await bot.send_message(chat_id, error_note)
                except Exception:  # noqa: BLE001
                    pass
    finally:
        typing_task.cancel()
        try:
            await typing_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001, PERF203
            pass
    return sent


def _find_double_newline(s: str) -> int:
    """Ищем границу «пустая строка» — \n\n (возможно с пробелами между).
    Возвращает индекс начала первого \n. -1 если нет."""
    import re as _re
    m = _re.search(r"\n\s*\n", s)
    return m.start() if m else -1


async def _drain_boundaries(
    bot: Bot,
    chat_id: int,
    buf_holder: list[str],
    sent: list[str],
    stickers: "stickers_module.StickerStore | None",
) -> None:
    """Режем накопленный буфер по «границам», отправляя готовые части в чат.

    Границы (самая ранняя побеждает):
      - стикер-маркер [sticker: x] / <<STICKER:x>>: текст ДО = отдельное сообщение,
        дальше шлётся стикер, остаток остаётся в буфере;
      - <<MSG>>: текст до — сообщение, маркер вырезается, остаток — в буфер;
      - пустая строка (двойной перевод строки): то же, что <<MSG>>.

    Стикер-маркер может быть разрезан между чанками, поэтому если буфер кончается
    чем-то похожим на начало маркера (<<, [st, …) — не режем, ждём ещё. Мутирует
    buf_holder[0]: оставляет в нём только то, что ещё нельзя безопасно отправить
    (хвост после последней границы ИЛИ «обрубок» начинающегося маркера)."""
    while True:
        buf = buf_holder[0]
        if not buf:
            return

        # Кандидаты-границы с их позицией в buf.
        msg_pos = buf.find(prompts.MSG_SPLIT_TOKEN)
        nl_pos = _find_double_newline(buf)
        stk = _STICKER_MARKER_RE.search(buf)

        # Если хвост buf — потенциальное начало стикер-маркера, и при этом в
        # доступной части буфера НЕТ готовой границы до этого обрубка —
        # подождём ещё чанков, чтобы не отправить пользователю «<<S» как текст.
        # (если есть готовая граница раньше — её обработаем, обрубок сам дойдёт до неё)
        partial_tail = _ends_with_partial_marker(buf)
        stk_pos = stk.start() if stk else -1
        earliest = msg_pos if msg_pos != -1 else -1
        if nl_pos != -1 and (earliest == -1 or nl_pos < earliest):
            earliest = nl_pos
        if stk_pos != -1 and (earliest == -1 or stk_pos < earliest):
            earliest = stk_pos

        if earliest == -1 and partial_tail:
            # ждём — в буфере только обрубок маркера, готовых границ нет
            return

        if earliest == -1:
            # Готовых границ нет и хвост не обрубок — весь буфер это «ещё не
            # законченное сообщение». Шлём его только когда он достаточно
            # большой (кидающими листьями по слову не интересуемся), НО мы
            # хотим живую «печать»: шлём то, что накопилось, сразу. Однако для
            # целостности сообщений режем только по границам — поэтому здесь
            # ничего не шлём, ждём настоящую границу или конец стрима.
            return

        # Сам ранняя граница определена. Действуем по типу.
        # Стикер раньше всех?
        if stk_pos != -1 and (earliest == stk_pos):
            before = buf[:stk.start()]
            vibe = _vibe_from_match(stk)
            buf_holder[0] = buf[stk.end():]
            if before.strip():
                await bot.send_chat_action(chat_id, "typing")
                safe = before.strip()[:4000]
                await bot.send_message(chat_id, safe)
                sent.append(safe)
            if stickers is not None:
                file_id = stickers.get_file_id(vibe)
                if file_id:
                    await bot.send_chat_action(chat_id, "typing")
                    try:
                        await bot.send_sticker(chat_id, sticker=file_id)
                    except Exception:  # noqa: BLE001
                        logger.warning("не удалось отправить стикер %r", vibe)
                else:
                    logger.info("стикер-виба не найдена в пачке: %r — пропускаю", vibe)
            continue  # повторный разбор остатка

        # <<MSG>> раньше всех?
        if msg_pos != -1 and (earliest == msg_pos):
            piece = buf[:msg_pos]
            buf_holder[0] = buf[msg_pos + len(prompts.MSG_SPLIT_TOKEN):]
            clean = piece.strip()
            if clean:
                await bot.send_chat_action(chat_id, "typing")
                safe = clean[:4000]
                await bot.send_message(chat_id, safe)
                sent.append(safe)
            continue

        # \n\n раньше всех. nl_pos — начало первого \n блока \n\s*\n.
        piece = buf[:nl_pos]
        import re as _re
        m = _re.search(r"\n\s*\n", buf)
        buf_holder[0] = buf[m.end():] if m else buf[nl_pos + 2:]
        clean = piece.strip()
        if clean:
            await bot.send_chat_action(chat_id, "typing")
            safe = clean[:4000]
            await bot.send_message(chat_id, safe)
            sent.append(safe)


async def _noop() -> None:
    """Заглушка — корутина, которая сразу завершается. Подставляется как typing_task,
    когда emit_typing=False (proactive): finally корректно cancel/await по ней."""
    return None


async def _keep_typing(bot: Bot, chat_id: int) -> None:
    """Каждые 4с продляем «печатает…», пока жив корневой вызов модели."""
    try:
        while True:
            await asyncio.sleep(4)
            try:
                await bot.send_chat_action(chat_id, "typing")
            except Exception:  # noqa: BLE001
                pass
    except asyncio.CancelledError:
        return


# ---------------------------------------------------------------------------
# Хендлеры
# ---------------------------------------------------------------------------

async def cmd_start(message: Message, conn: "sqlite3.Connection") -> None:
    await run_db(db.ensure_user, conn, message.from_user.id)
    await message.answer("привет, я тут")


async def cmd_persona_text(
    message: Message,
    conn: "sqlite3.Connection",
) -> None:
    """  /persona reset  |  /persona <описание стиля текстом>  """
    args = (message.text or "").split(maxsplit=1)
    arg = args[1].strip() if len(args) > 1 else ""

    if not arg:
        await message.answer(
            "Можно по-разному:\n"
            "1) Напиши описание стиля текстом:\n"
            "   /persona грубый тролль, короткими фразами, без эмодзи\n"
            "2) Пришли сюда файл переписки (Telegram JSON или .txt), а в подписи к файлу — /persona\n"
            "3) /persona reset — вернуться к обычному стилю"
        )
        return

    if arg.lower() in {"reset", "сброс", "clear", "off", "выкл"}:
        await run_db(db.delete_persona, conn, message.from_user.id)
        await message.answer("ок, стиль сброшен — общаюсь как обычно")
        return

    # Имя персонажа — первые слова описания.
    name = arg.split(",")[0].strip().capitalize() or "Персонаж"
    summary = arg
    await run_db(db.set_persona, conn, message.from_user.id, name, summary, source_raw=arg)
    await message.answer(f"настроил стиль: {name}\n{summary}")


# ---------------------------------------------------------------------------
# /persona с прикреплённым файлом переписки — спрашиваем, кого брать за персону
# ---------------------------------------------------------------------------

async def cmd_persona_document(
    message: Message,
    conn: "sqlite3.Connection",
    bot: Bot,
    ai: CloudflareClient,
    state: FSMContext,
) -> None:
    """/persona с прикреплённым файлом переписки.

    Flow (новый):
      1. Скачиваем файл.
      2. extract_speakers(raw) — без AI, чисто по структуре файла.
         - 0 спикеров (plain text без авторской разметки / нераспознанный формат)
           → fallback на legacy: parse_export + summarize_style(speaker_name=None).
           Модель сама выберет «самого активного».
         - 1 спикер → сразу parse_export_for_speaker + summarize_style(speaker_name=name).
         - 2+ спикеров → сохраняем raw_bytes и speakers в FSM, шлём inline-клавиатуру.
           Дальнейший путь — on_speaker_chosen.
    """
    doc = message.document
    if doc is None:
        await cmd_persona_text(message, conn)
        return

    # Проверим, что в подписи действительно была команда /persona.
    caption = (message.caption or "").strip()
    if not caption.startswith("/persona"):
        return

    if doc.file_size is not None and doc.file_size > 5 * 1024 * 1024:
        await message.answer("файл слишком большой, дай что-то до 5 МБ")
        return

    await bot.send_chat_action(message.chat.id, "typing")
    try:
        raw = await media.download_to_bytes(bot, doc.file_id)
    except Exception:  # noqa: BLE001
        logger.exception("ошибка скачивания файла для persona")
        await message.answer("не смог скачать файл, попробуй ещё раз")
        return

    # Спикеров ищем без AI — чисто по структуре файла.
    speakers: list[str] = []
    try:
        speakers = persona_mod.extract_speakers(raw)
    except Exception:  # noqa: BLE001 — extract_speakers сам по себе не бросает,
        # но подстрахуемся: если упадёт, не роняем хендлер, идём по legacy.
        logger.exception("extract_speakers упал, откат на legacy-режим")

    if not speakers:
        # Plain text / нераспознанный формат — без авторской разметки вопрос
        # бессмысленный. Откатываемся на legacy-режим (модель сама выберет).
        logger.info(
            "user=%s: спикеров не найдено, fallback на legacy-режим",
            message.from_user.id,
        )
        await _extract_and_save_legacy(message, raw, conn, ai, bot)
        return

    if len(speakers) == 1:
        # Один спикер — спрашивать не о чем.
        name = speakers[0]
        logger.info("user=%s: один спикер %r, без вопроса", message.from_user.id, name)
        await _extract_and_save_for_speaker(message, raw, name, conn, ai, bot)
        return

    # 2+ спикеров — спросить. Кладём в FSM всё, что нужно для продолжения.
    # raw_bytes — может быть тяжёлым (до 5 МБ); MemoryStorage это переварит для
    # личного бота, но на проде с большим числом юзеров стоит кешировать во
    # временной таблице по user_id, а в FSM держать только rowid.
    await state.update_data(persona_raw_bytes=raw, persona_speakers=speakers)
    await state.set_state(PersonaChoice.waiting_speaker)

    kb = _build_speaker_keyboard(speakers)
    await message.answer(
        "В переписке несколько собеседников. На ком строим персону?",
        reply_markup=kb,
    )


def _build_speaker_keyboard(speakers: list[str]) -> InlineKeyboardMarkup:
    """Клавиатура: по одной кнопке на спикера + «самый активный» внизу
    (legacy-режим — для тех, кто не хочет выбирать)."""
    builder = InlineKeyboardBuilder()
    for idx, name in enumerate(speakers):
        # Текст кнопки = имя спикера. Если длинное — Telegram обрежет сам.
        builder.row(InlineKeyboardButton(
            text=name,
            callback_data=f"{_PERSONA_CB_PREFIX}{idx}",
        ))
    # Fallback-кнопка: старое поведение (модель сама выберет самого активного).
    builder.row(InlineKeyboardButton(
        text="Любой (самый активный)",
        callback_data=f"{_PERSONA_CB_PREFIX}{_PERSONA_LEGACY_IDX}",
    ))
    return builder.as_markup()


async def on_speaker_chosen(
    callback: CallbackQuery,
    state: FSMContext,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    bot: Bot,
) -> None:
    """Callback-обработчик на нажатие кнопки выбора спикера.

    Регистрируется в setup() с StateFilter(PersonaChoice.waiting_speaker) и
    фильтром по префиксу callback_data.
    """
    # Парсим индекс из callback_data. На любой malformed-data — отвечаем error
    # и остаёмся в ожидании (юзер может ткнуть другую кнопку).
    assert callback.data is not None  # проверено фильтром в setup()
    payload = callback.data[len(_PERSONA_CB_PREFIX):]
    try:
        idx = int(payload)
    except ValueError:
        logger.warning(
            "user=%s: невалидный callback_data=%r",
            callback.from_user.id, callback.data,
        )
        await callback.answer("Что-то пошло не так, ткни ещё раз", show_alert=True)
        return

    data = await state.get_data()
    raw: bytes | None = data.get("persona_raw_bytes")
    speakers: list[str] | None = data.get("persona_speakers")

    if not raw or not speakers:
        # FSM-данные потерялись (перезагрузка / истёк TTL / юзер ткнул «старую»
        # кнопку из прошлого сообщения). Просим переотправить файл.
        await callback.answer(
            "Сессия устарела — переотправь файл переписки", show_alert=True,
        )
        await state.clear()
        return

    user_id = callback.from_user.id

    # Выходим из состояния ДО долгого AI-вызова — иначе если AI упадёт, юзер
    # зависнет в waiting_speaker и не сможет загрузить другой файл.
    await state.clear()
    await callback.answer()  # снять «часики» с кнопки

    # callback.message — то сообщение с клавиатурой, на котором юзер ткнул кнопку.
    # Используем его для ответов (тот же chat_id, что и у исходного файла).
    reply_to: Message | None = callback.message

    if idx == _PERSONA_LEGACY_IDX:
        # Fallback: старое поведение — все реплики, модель сама выберет.
        if reply_to is not None:
            await bot.send_message(
                reply_to.chat.id,
                "Окей, беру самого разговорчивого — посмотрим, что выйдет…",
            )
        await _extract_and_save_legacy(reply_to, raw, conn, ai, bot)
        return

    if idx < 0 or idx >= len(speakers):
        logger.warning(
            "user=%s: индекс спикера %d вне диапазона %d",
            user_id, idx, len(speakers),
        )
        if reply_to is not None:
            await bot.send_message(
                reply_to.chat.id, "Что-то не сходится — переотправь файл переписки",
            )
        return

    name = speakers[idx]
    if reply_to is not None:
        await bot.send_message(
            reply_to.chat.id,
            f"Окей, строю персону на «{name}» — это займёт пару секунд…",
        )
    await _extract_and_save_for_speaker(reply_to, raw, name, conn, ai, bot)


# ---------------------------------------------------------------------------
# Внутренние хелперы — общая логика извлечения стиля и сохранения персоны.
# ---------------------------------------------------------------------------

async def _extract_and_save_for_speaker(
    message: Message | None,
    raw: bytes,
    speaker_name: str,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    bot: Bot,
) -> None:
    """Извлекает стиль конкретного спикера и сохраняет персону в БД.

    sample — только реплики выбранного спикера, без «Author:» префикса.
    speaker_name пробрасывается в summarize_style → build_persona_extraction_prompt,
    чтобы модели было жёстко сказано «опирайся ТОЛЬКО на этого собеседника».
    В БД сохраняем: name=speaker_name (то имя, что юзер выбрал на кнопке).

    bot нужен для send_chat_action — у Message в aiogram 3.x НЕТ метода
    answer_chat_action, индикатор «печатает…» ставится через бота.
    """
    if message is None:
        logger.warning("_extract_and_save_for_speaker: message is None — некуда отвечать")
        return
    chat_id = message.chat.id

    try:
        sample = persona_mod.parse_export_for_speaker(raw, speaker_name)
    except Exception:  # noqa: BLE001 — parse_export_for_speaker не бросает,
        # но подстрахуемся: на любом непонятном формате лучше вежливая ошибка,
        # чем упавший хендлер.
        logger.exception(
            "не удалось разобрать файл persona для спикера %r", speaker_name,
        )
        await message.answer(
            "не смог разобрать этот файл — формат не опознан. "
            "Подходит Telegram JSON-экспорт или обычный .txt с репликами."
        )
        return

    if not sample.strip():
        await message.answer(
            f"Не вышло вытащить реплики «{speaker_name}» — возможно, имя в "
            "файле записано иначе. Переотправь файл или выбери другого."
        )
        return

    await bot.send_chat_action(chat_id, "typing")
    try:
        summary = await persona_mod.summarize_style(ai, sample, speaker_name=speaker_name)
    except CloudflareError as exc:
        logger.warning(
            "persona extraction failed for speaker=%r user: %s",
            speaker_name, exc,
        )
        await message.answer("не получилось разобрать стиль, попробуй чуть позже 🌚")
        return

    # name = speaker_name — то имя, что юзер выбрал на кнопке. Это и есть
    # «как зовут персонажа», логично сохранять именно его.
    await run_db(
        db.set_persona, conn, message.from_user.id,
        speaker_name, summary, source_raw=sample[:2000],
    )
    await message.answer(f"настроил стиль из переписки ({speaker_name}):\n\n{summary}")


async def _extract_and_save_legacy(
    message: Message | None,
    raw: bytes,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    bot: Bot,
) -> None:
    """Legacy-путь: все реплики + модель сама выбирает самого активного.

    Используется как fallback, когда спикеров не нашлось (plain text без
    авторской разметки) или юзер ткнул «Любой (самый активный)» на клавиатуре.

    bot нужен для send_chat_action — у Message в aiogram 3.x НЕТ метода
    answer_chat_action, индикатор «печатает…» ставится через бота.
    """
    if message is None:
        logger.warning("_extract_and_save_legacy: message is None — некуда отвечать")
        return
    chat_id = message.chat.id

    try:
        sample = persona_mod.parse_export(raw)
    except Exception:  # noqa: BLE001
        logger.exception("не удалось разобрать файл persona (legacy)")
        await message.answer(
            "не смог разобрать этот файл — формат не опознан. "
            "Подходит Telegram JSON-экспорт или обычный .txt с репликами."
        )
        return
    if not sample.strip():
        await message.answer(
            "не вытащил текст из файла. Подходит Telegram JSON-экспорт или обычный .txt с репликами."
        )
        return

    await bot.send_chat_action(chat_id, "typing")
    try:
        summary = await persona_mod.summarize_style(ai, sample, speaker_name=None)
    except CloudflareError as exc:
        logger.warning("persona extraction failed (legacy): %s", exc)
        await message.answer("не получилось разобрать стиль, попробуй чуть позже 🌚")
        return

    name = "Импортированный стиль"
    await run_db(
        db.set_persona, conn, message.from_user.id,
        name, summary, source_raw=sample[:2000],
    )
    await message.answer(f"настроил стиль из переписки:\n\n{summary}")


# ---------------------------------------------------------------------------
# Медиа-ходы и текст
# ---------------------------------------------------------------------------

async def _media_turn(
    message: Message,
    bot: Bot,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    stickers: "stickers_module.StickerStore | None",
    cfg: Config,
    vision_ai: CloudflareClient,
) -> None:
    """Общий ход для фото и стикера.

    Изображение (когда получилось скачать/раскодировать) уходит в vision_ai —
    провайдер, выбранный отдельно через VISION_PROVIDER (может отличаться от
    текстового ai). Если картинки нет (не скачалась/не картинка) — используем
    обычный текстовый ai, обращаться к vision-провайдеру незачем.
    """
    await run_db(db.ensure_user, conn, message.from_user.id)

    try:
        image_data_uri, kind, extra_text = await media.prepare_media(bot, message)
    except Exception:  # noqa: BLE001
        logger.exception("ошибка обработки медиа")
        image_data_uri, kind, extra_text = None, "", None

    # Empty photo caption is still a valid value. Keep None for actual failures
    # only, so the prompt never claims that a successfully attached image could
    # not be inspected.
    if message.photo and extra_text is None and image_data_uri:
        extra_text = ""

    # Gemma получает изображение прямо в текущем сообщении. Отдельный вызов
    # caption-модели больше не нужен: один запрос одновременно видит медиа,
    # историю, persona и формирует окончательный ответ.
    description = ""

    # 3. Собираем текстовое user-сообщение.
    # Полный head (с vision-описанием) — модель видит СЕЙЧАС, чтобы среагировать на
    # содержимое. В БД (для будущей истории) пишем сжатую форму: длинное описание
    # картинки в истории только засоряет контекст и сбивает модель — новый юзер
    # переписки этому не подвергается. Реальные слова собеседника (caption фото)
    # сохраняем обязательно.
    if kind == "фото":
        head = "[пользователь прислал фото"
        if description:
            head += f": {description.strip()}"
        elif extra_text is None:
            head += " (не удалось разобрать содержимое)"
        head += "]"
    elif kind == "стикер":
        # Эмодзи стикера — только СЛАБЫЙ хинт: Telegram-тег пака часто не
        # соответствует тому, что реально нарисовано (.gg/.webm анимации, юмор
        # пака). Основной сигнал для модели — сама картинка-плашка; эмодзи
        # упоминаем с явным понижением веса, иначе модель реагирует на тег,
        # а не на изображение (баг: смайл-тег на стикере с осмысленной сценой).
        head = "[пользователь прислал стикер"
        if extra_text:
            head += (
                f" — смотри на саму картинку; эмодзи-тег в Telegram ({extra_text}) "
                "часто не соответствует изображению, ориентируйся на картинку"
            )
        if description:
            head += f"; {description.strip()}"
        head += "]"
    else:
        head = "[пользователь прислал медиа]"

    # caption к фото юзера — отдельной строкой, это его реальные слова.
    if kind == "фото" and extra_text:
        user_text = head + f"\nСообщение пользователя: {extra_text}"
    else:
        user_text = head

    # Сжатая форма для истории БД: только тип + эмодзи-тег для идентификации,
    # без длинного vision-описания (оно модели нужно один раз — в текущем ходе).
    # Эмодзи-тег в истории — просто метка «какой стикер был», вес для модели не
    # имеет значения (она видит сжатую форму как [стикер: 😾] без картинки).
    if kind == "стикер":
        emoji = None
        if extra_text and extra_text.startswith("эмодзи стикера: "):
            emoji = extra_text[len("эмодзи стикера: "):].strip()
        head_short = f"[стикер: {emoji}]" if emoji else "[стикер]"
    elif kind == "фото":
        head_short = f"[фото: {extra_text}]" if extra_text else "[фото]"
    else:
        head_short = "[медиа]"

    # Дедуп повторных отправок медиа (Telegram long-poll retry): сравниваем по сжатой
    # форме, что запишется в БД — дубль не пишем и не стримим ответ.
    if await run_db(db.is_dup_user_text, conn, message.from_user.id, head_short):
        logger.info("дубль media user=%s kind=%s пропущен", message.from_user.id, kind or "?")
        return

    user_db_content = head_short

    current_user_content: list[dict] | str = user_text
    used_ai = ai
    if image_data_uri:
        # Плашка времени для медиа-хода — внутрь text-блока (кеш: меняется только
        # маленький хвост текущего user-хода, префикс [system + история] цел).
        header = _time_header(cfg, message.from_user.id, conn)
        text_with_header = (user_text + "\n\n" + header) if header else user_text
        current_user_content = [
            {"type": "text", "text": text_with_header},
            media.image_content_block(image_data_uri, cfg.vision_provider),
        ]
        used_ai = vision_ai

    messages = await _build_messages(
        message, cfg, conn, stickers, current_user_content
    )

    await run_db(
        db.add_message, conn, message.from_user.id, "user",
        "sticker" if kind == "стикер" else "image", user_db_content,
    )
    # Стримим ответ; _stream_reply сам обрабатывает CloudflareError.
    # used_ai — vision-провайдер, если картинка реально ушла в запрос, иначе
    # обычный текстовый ai (см. выбор выше).
    sent = await _stream_reply(
        bot, message.chat.id, used_ai, messages, cfg, stickers,
    )
    for chunk in sent:
        await run_db(db.add_message, conn, message.from_user.id, "assistant", "text", chunk)


async def handle_photo(
    message: Message,
    bot: Bot,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    stickers: "stickers_module.StickerStore | None",
    cfg: Config,
    vision_ai: CloudflareClient,
) -> None:
    await _media_turn(message, bot, conn, ai, stickers, cfg, vision_ai)


async def handle_sticker(
    message: Message,
    bot: Bot,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    stickers: "stickers_module.StickerStore | None",
    cfg: Config,
    vision_ai: CloudflareClient,
) -> None:
    await _media_turn(message, bot, conn, ai, stickers, cfg, vision_ai)


async def handle_text(
    message: Message,
    bot: Bot,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    stickers: "stickers_module.StickerStore | None",
    cfg: Config,
) -> None:
    text = (message.text or "").strip()
    if not text:
        return
    await run_db(db.ensure_user, conn, message.from_user.id)

    # Дедуп повторных отправок (Telegram long-poll retry): если ровно такой же
    # ход уже записан за последнее окно — не пишем дубль и не отвечаем again.
    if await run_db(db.is_dup_user_text, conn, message.from_user.id, text):
        logger.info("дубль user-text пропущен user=%s text=%r", message.from_user.id, text[:60])
        return

    # Быстрый путь «запомни: X»: явная просьба что-то запомнить — факт пишется
    # сразу (без ожидания фонового витка кураторки), category='legacy' —
    # кураторка потом разложит по категориям. Ответ бота идёт как обычно
    # (модель видит фразу в истории и реагирует в своём характере).
    remember_match = _REMEMBER_RE.match(text)
    if remember_match:
        fact_text = remember_match.group(1).strip()
        await run_db(
            db.add_fact, conn, message.from_user.id, "legacy", fact_text,
            source_msg_id=None,
        )
        logger.info(
            "запомни user=%s: факт записан сразу %r", message.from_user.id, fact_text[:60]
        )

    messages = await _build_messages(message, cfg, conn, stickers, text)

    await run_db(db.add_message, conn, message.from_user.id, "user", "text", text)
    sent = await _stream_reply(
        bot, message.chat.id, ai, messages, cfg, stickers,
    )
    for chunk in sent:
        await run_db(db.add_message, conn, message.from_user.id, "assistant", "text", chunk)


# ---------------------------------------------------------------------------
# Регистрация
# ---------------------------------------------------------------------------

def setup(
    dp: Dispatcher,
    bot: Bot,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    cfg: Config,
    stickers: "stickers_module.StickerStore | None",
    vision_ai: CloudflareClient | None = None,
) -> None:
    # /start
    dp.message.register(cmd_start, Command("start"))

    # /persona с прикреплённым документом (caption начинается с /persona) — приоритет.
    dp.message.register(
        cmd_persona_document,
        Command("persona"),
        F.document,
    )
    # /persona текстом или /persona reset
    dp.message.register(cmd_persona_text, Command("persona"), F.text)

    # Callback от inline-кнопок выбора спикера (срабатывает только в FSM-состоянии
    # waiting_speaker — см. cmd_persona_document, где мы в него входим). Фильтр по
    # префиксу callback_data страхует от случайного срабатывания на чужих кнопках.
    dp.callback_query.register(
        on_speaker_chosen,
        StateFilter(PersonaChoice.waiting_speaker),
        lambda c: bool(c.data and c.data.startswith(_PERSONA_CB_PREFIX)),
    )

    # Фото приоритетнее текста.
    dp.message.register(handle_photo, F.photo)
    # Стикеры
    dp.message.register(handle_sticker, F.sticker)
    # Обычный текст
    dp.message.register(handle_text, F.text)

    # Прокидываем зависимости в хендлеры через kwargs (aiogram middleware).
    dp["conn"] = conn
    dp["bot"] = bot
    dp["ai"] = ai
    dp["cfg"] = cfg
    dp["stickers"] = stickers
    # vision_ai не задан явно — используем тот же клиент, что и для текста
    # (совпадает с поведением при VISION_PROVIDER == AI_PROVIDER в config.py).
    dp["vision_ai"] = vision_ai if vision_ai is not None else ai
