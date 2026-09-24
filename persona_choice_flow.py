"""Интеграция выбора спикера в handlers.py — готовый флоу для копирования.

Не зависящий от остального handlers.py модуль: показывает, как обработать
загрузку файла переписки, спросить кого из собеседников брать за персону
(через inline-кнопки), и затем извлечь стиль выбранного спикера.

Что нужно добавить в существующий handlers.py:
  1. Зарегистрировать FSM-состояние PersonaChoice.waiting_speaker.
  2. В обработчике загрузки файла (DocumentHandler) — вместо немедленного
     parse_export + summarize_style, вызвать ask_speaker_or_extract(...) из
     этого модуля: он сам решит, спрашивать или сразу извлекать.
  3. Зарегистрировать callback_query_handler на prefix "persona:" — он пойдёт
     в on_speaker_chosen.
  4. (Опционально) Добавить кнопку «Любой / самый активный» — она позволить
     сохранить старое поведение, когда юзер не хочет выбирать.

Зависимости:
  - aiogram 3.x (CallbackQuery, InlineKeyboardBuilder, FSMContext)
  - persona.extract_speakers, persona.parse_export_for_speaker,
    persona.parse_export, persona.summarize_style — из обновлённого persona.py.
  - db.set_persona (или ваш аналог) для сохранения результата.

Memory-стратегия: между получением файла и кликом по кнопке храним raw_bytes
в FSM Storage (keyed by user_id). MemoryStorage — ок для малого числа юзеров;
для прод — RedisStorage, чтобы пережить рестарт воркера. На больших файлах
(>1МБ) лучше кешировать во временной файловой таблице по user_id, а в FSM
держать только путь — иначе MemoryStorage раздуется.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from aiogram import Bot
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    CallbackQuery,
    InlineKeyboardButton,
    InlineKeyboardMarkup,
    Message,
)
from aiogram.utils.keyboard import InlineKeyboardBuilder
from aiogram.filters import StateFilter

import db
import persona
from cloudflare_client import CloudflareClient

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


# FSM-состояние: ждём, пока юзер ткнёт кнопку выбора спикера.
# Добавьте этот класс в свой существующий StatesGroup или используйте как есть.
class PersonaChoice(StatesGroup):
    waiting_speaker = State()


# Префикс для callback_data: "persona:<index>". Имя спикера НЕ кодируем в
# callback_data — оно может быть длинным/с эмодзи/со скобками (как «Диана
# Белоусова (my love)»). Вместо этого передаём индекс, а сами имена достаём
# из FSM Storage, куда сохранили список при отправке клавиатуры.
CALLBACK_PREFIX = "persona:"
# Специальный индекс для fallback-кнопки «самый активный» — старое поведение.
LEGACY_ANY_INDEX = -1


async def ask_speaker_or_extract(
    message: Message,
    state: FSMContext,
    raw_bytes: bytes,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    user_id: int,
) -> None:
    """Точка входа: вызывается из обработчика загрузки файла переписки.

    Логика:
      - extract_speakers находит 0 спикеров → fallback: parse_export (все
        реплики) + summarize_style(speaker_name=None). Старое поведение.
      - ровно 1 спикер → спрашивать не о чем, сразу parse_export_for_speaker
        + summarize_style(speaker_name=name).
      - 2+ спикеров → сохранить raw_bytes и список спикеров в FSM, отправить
        inline-клавиатуру. Дальнейший путь — on_speaker_chosen.
    """
    speakers = persona.extract_speakers(raw_bytes)

    if not speakers:
        # Plain text / нераспознанный формат — без авторской разметки вопрос
        # бессмысленный. Откатываемся на legacy-режим (модель сама выберет).
        logger.info("user=%s: спикеров не найдено, fallback на legacy-режим", user_id)
        await _extract_and_save_legacy(message, raw_bytes, conn, ai, user_id)
        return

    if len(speakers) == 1:
        # Один спикер — спрашивать не о чем.
        name = speakers[0]
        logger.info("user=%s: один спикер %r, без вопроса", user_id, name)
        await _extract_and_save_for_speaker(message, raw_bytes, name, conn, ai, user_id)
        return

    # 2+ спикеров — спросить. Кладём в FSM всё, что нужно для продолжения.
    await state.update_data(persona_raw_bytes=raw_bytes, persona_speakers=speakers)
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
            callback_data=f"{CALLBACK_PREFIX}{idx}",
        ))
    # Fallback-кнопка: старое поведение (модель сама выберет самого активного).
    builder.row(InlineKeyboardButton(
        text="Любой (самый активный)",
        callback_data=f"{CALLBACK_PREFIX}{LEGACY_ANY_INDEX}",
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

    Регистрация в handlers.setup:
        dp.callback_query.register(
            on_speaker_chosen,
            StateFilter(PersonaChoice.waiting_speaker),
            lambda c: c.data and c.data.startswith(CALLBACK_PREFIX),
        )
    """
    # Парсим индекс из callback_data. На любой malformed-data — отвечаем error
    # и остаёмся в ожидании (юзер может ткнуть другую кнопку).
    assert callback.data is not None  # проверено фильтром выше
    payload = callback.data[len(CALLBACK_PREFIX):]
    try:
        idx = int(payload)
    except ValueError:
        logger.warning("user=%s: невалидный callback_data=%r", callback.from_user.id, callback.data)
        await callback.answer("Что-то пошло не так, ткни ещё раз", show_alert=True)
        return

    data = await state.get_data()
    raw_bytes: bytes | None = data.get("persona_raw_bytes")
    speakers: list[str] | None = data.get("persona_speakers")

    if not raw_bytes or not speakers:
        # FSM-данные потерялись (перезагрузка / истёк TTL / юзер ткнул «старую»
        # кнопку из прошлого сообщения). Просим переотправить файл.
        await callback.answer("Сессия устарела — переотправь файл переписки", show_alert=True)
        await state.clear()
        return

    user_id = callback.from_user.id

    # Выходим из состояния ДО долгого AI-вызова — иначе если AI упадёт, юзер
    # зависнет в waiting_speaker и не сможет загрузить другой файл.
    await state.clear()
    await callback.answer()  # снять «часики» с кнопки

    if idx == LEGACY_ANY_INDEX:
        # Fallback: старое поведение — все реплики, модель сама выберет.
        await bot.send_message(user_id, "Окей, беру самого разговорчивого — посмотрим, что выйдет…")
        await _extract_and_save_legacy(callback.message, raw_bytes, conn, ai, user_id)
        return

    if idx < 0 or idx >= len(speakers):
        logger.warning("user=%s: индекс спикера %d вне диапазона %d", user_id, idx, len(speakers))
        await bot.send_message(user_id, "Что-то не сходится — переотправь файл переписки")
        return

    name = speakers[idx]
    await bot.send_message(user_id, f"Окей, строю персону на «{name}» — это займёт пару секунд…")
    await _extract_and_save_for_speaker(callback.message, raw_bytes, name, conn, ai, user_id)


# ---------------------------------------------------------------------------
# Внутренние хелперы — общая логика извлечения и сохранения.
# ---------------------------------------------------------------------------

async def _extract_and_save_for_speaker(
    message: Message,
    raw_bytes: bytes,
    speaker_name: str,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    user_id: int,
) -> None:
    """Извлекает стиль конкретного спикера и сохраняет персону в БД."""
    try:
        sample = persona.parse_export_for_speaker(raw_bytes, speaker_name)
        if not sample.strip():
            await message.answer(
                f"Не вышло вытащить реплики «{speaker_name}» — возможно, имя в "
                "файле записано иначе. Переотправь файл или выбери другого."
            )
            return
        summary = await persona.summarize_style(ai, sample, speaker_name=speaker_name)
    except Exception as exc:  # noqa: BLE001 — задача хендлера: не уронить бота
        logger.exception("user=%s: summarize_style упал", user_id)
        await message.answer(f"Не получилось собрать персону: {exc}")
        return

    await db.set_persona(conn, user_id, summary)
    await message.answer("Персона готова — теперь я буду говорить в этом стиле.")


async def _extract_and_save_legacy(
    message: Message,
    raw_bytes: bytes,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    user_id: int,
) -> None:
    """Legacy-путь: все реплики + модель сама выбирает самого активного.
    Используется как fallback, когда спикеров не нашлось или юзер ткнул
    «Любой (самый активный)»."""
    try:
        sample = persona.parse_export(raw_bytes)
        summary = await persona.summarize_style(ai, sample, speaker_name=None)
    except Exception as exc:  # noqa: BLE001
        logger.exception("user=%s: summarize_style (legacy) упал", user_id)
        await message.answer(f"Не получилось собрать персону: {exc}")
        return

    await db.set_persona(conn, user_id, summary)
    await message.answer("Персона готова — теперь я буду говорить в этом стиле.")


# ---------------------------------------------------------------------------
# Шаблон регистрации (для копирования в handlers.setup).
# ---------------------------------------------------------------------------

def register_persona_choice(dp, bot, conn, ai, cfg, sticker_store) -> None:
    """Регистрирует callback-обработчик выбора спикера.

    Кладём это рядом со своим handlers.setup и зовём из bot.py:

        from persona_choice_flow import register_persona_choice
        register_persona_choice(dp, conn=conn, ai=ai, bot=bot, cfg=cfg, sticker_store=sticker_store)

    Обработчик загрузки файла (DocumentHandler) в handlers.py нужно
    переписать: вместо parse_export + summarize_style сразу звать
    ask_speaker_or_extract(...) из этого модуля.

    callback_data префикс «persona:» не должен пересекаться с другими
    callback-префиксами в боте. Если пересекается — поменяйте CALLBACK_PREFIX.
    """
    dp.callback_query.register(
        lambda c, state: on_speaker_chosen(c, state, conn, ai, bot),
        StateFilter(PersonaChoice.waiting_speaker),
        lambda c: bool(c.data and c.data.startswith(CALLBACK_PREFIX)),
    )
