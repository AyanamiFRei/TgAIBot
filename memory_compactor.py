"""Автокомпакт памяти: фоновый цикл сжатия старой истории в краткое изложение.

Зачем: длинная история гоняется в промпт на каждый запрос → большой промпт →
глубокий reasoning-поток у glm-5.3-flash → ответы по минутам. Компактор
периодически сжимает старые сообщения моделью в краткую «память» (факты, темы,
решения — БД, таблица memories), в промпт идёт только память + свежий хвост.

Схема компактации:
  - триггер: непокрытых сообщений после compacted_until больше COMPACT_TRIGGER_MSGS;
  - на вход модели: прежняя память (если была) + сообщения в диапазоне
    (compacted_until, id последнего - COMPACT_KEEP_RECENT] — сжимаем всё, кроме
    свежего хвоста (он ещё живёт в обычном диалоге);
  - промпт: «выдели важное, кратко» — сами размышления компактатора не важны,
    гоняем с маленьким max_tokens и enable_thinking=false;
  - результат перезаписывает память, compacted_until двигается вперёд.

Порядок с Pronomina: модель зовётся через тот же CloudflareClient (base_url/
ключи те же). Сжатие — единственное место, где бот осознанно строит НЕ
диалоговый запрос: это фоновая задача, не реплика собеседнику.
"""
from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

import db
from aiogram import Bot
from cloudflare_client import CloudflareClient

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

COMPACT_PROMPT = (
    "Ты — редактор памяти. Ниже — переписка (роли user/assistant) и, возможно, "
    "прежняя краткая память. Составь ОБНОВЛЁННУЮ краткую память: только важное — "
    "факты о собеседнике (имя, чем занимается, проекты, предпочтения), ключевые "
    "темы и договорённости, важные события. Без воды, без пересказа каждого хода, "
    "без размышлений. Максимум ~1500 символов. Верни только текст памяти, без "
    "вступлений и пояснений."
)


async def _compact_once(
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    user_id: int,
    keep_recent: int,
) -> bool:
    """Сжимает непокрытую старую историю юзера в память. True — память обновлена.

    keep_recent — сколько ПОСЛЕДНИХ сообщений оставить вне памяти (свежий хвост,
    он продолжает жить в обычном диалоге без компактации).
    """
    last = await _run_db(db.last_row, conn, user_id)
    if last is None:
        return False
    last_id = last["id"]

    memory = await _run_db(db.get_memory, conn, user_id)
    compacted_until = memory["compacted_until"] if memory else 0

    # Непокрытых сообщений достаточно? (всё, что после compacted_until, минус
    # свежий хвост, который оставляем жить)
    pending = await _run_db(
        db.messages_to_compact, conn, user_id, compacted_until, last_id,
    )
    if len(pending) <= keep_recent:
        return False

    # Компактируем всё, кроме свежего хвоста.
    to_compact = pending[:-keep_recent] if keep_recent else pending
    if not to_compact:
        return False

    lines: list[str] = []
    for row in to_compact:
        who = "собеседник" if row["role"] == "user" else "ты"
        content = (row["content"] or "").strip()
        if content:
            lines.append(f"{who}: {content}")
    transcript = "\n".join(lines)
    if not transcript:
        return False

    user_prompt = COMPACT_PROMPT
    if memory is not None and memory["content"]:
        user_prompt += (
            "\n\nПрежняя краткая память (обнови её с учётом новой переписки):\n"
            + memory["content"]
        )
    user_prompt += "\n\nПереписка:\n" + transcript

    try:
        summary = await ai.chat(
            [
                {"role": "system", "content": "Ты — редактор памяти. Отвечай кратко, только важное, без размышлений."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,          # точность важнее живости: это фоновая память
            max_tokens=1024,
            model=None,               # основная модель; enable_thinking выключаем ниже
        )
    except Exception:  # noqa: BLE001
        logger.exception("компактация памяти user=%s не удалась", user_id)
        return False

    summary = summary.strip()
    if not summary:
        logger.warning("компактация user=%s: пустой ответ, память не обновляю", user_id)
        return False

    new_until = to_compact[-1]["id"]
    await _run_db(db.set_memory, conn, user_id, summary, new_until)
    logger.info(
        "память обновлена user=%s: %d сообщений → %d символов (compacted_until=%d)",
        user_id, len(to_compact), len(summary), new_until,
    )
    return True


async def _run_db(fn, *args, **kwargs):
    """Обёртка handlers.run_db: синхронный db-вызов в отдельном потоке."""
    import asyncio as _a
    try:
        to_thread = _a.to_thread
    except AttributeError:  # Python 3.8
        import functools
        loop = _a.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
    return await to_thread(fn, *args, **kwargs)


async def memory_compactor_loop(
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    keep_recent: int,
    wake_min: float,
) -> None:
    """Просыпается раз в wake_min минут и проверяет всех юзеров: у кого накопилась
    непокрытая история — сжимает её в память. Тихо спит при пустой БД."""
    logger.info("memory_compactor запущен (wake=%.1f min, keep_recent=%d)", wake_min, keep_recent)
    try:
        while True:
            try:
                activity = await _run_db(db.last_activity_per_user, conn)
                for user_id in activity:
                    try:
                        await _compact_once(conn, ai, user_id, keep_recent)
                    except Exception:  # noqa: BLE001
                        logger.exception("компакт user=%s упал, продолжаю", user_id)
            except Exception:  # noqa: BLE001
                logger.exception("memory_compactor виток упал")
            await asyncio.sleep(wake_min * 60.0)
    except asyncio.CancelledError:
        logger.info("memory_compactor отменён")
        raise
