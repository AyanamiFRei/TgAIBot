"""Proactive messaging — фоновый цикл, по которому бот иногда пишет первым.

Гибрид двух мотивов:
  - silence: последняя реплика была от юзера, он замолчал — бот «соскучился».
  - continuation: последняя реплика была от бота — ему есть что добавить позже.

Вероятность пинка растёт с тишиной: ниже min_silence_h → 0, рампа 0.2→0.8 на
[min, max], выше max → 0.85. Жёсткой отсечки «слишком давно» нет.

Дневной лимит (rolling 24h) и per-user троттл (min_gap_h) общие для обоих мотивов.
Бот строго 1:1 DM → chat_id == user_id для отправки. Без часового пояса: все
времена — epoch-секунды UTC (db хранит int(time.time())).
"""
from __future__ import annotations

import asyncio
import logging
import random
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import db
import prompts
from aiogram import Bot
from config import Config
from handlers import _stream_reply, run_db
from cloudflare_client import CloudflareClient
from stickers import StickerStore

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Шанс пинка
# ---------------------------------------------------------------------------

def poke_chance(silence_h: float, min_h: float, max_h: float) -> float:
    """Вероятность пнуть в зависимости от часов тишины.

    < min_h   → 0.0  (только что общались, рано)
    в [min,max] → рампа 0.20 → 0.80 (растёт линейно)
    > max_h   → 0.85 (активно, но не детерминированно — иначе каждый wake бил бы)
    """
    if silence_h < min_h:
        return 0.0
    if silence_h > max_h:
        return 0.85
    if max_h <= min_h:
        return 0.85
    frac = (silence_h - min_h) / (max_h - min_h)
    return 0.2 + 0.6 * frac


# ---------------------------------------------------------------------------
# Shim конфига для _stream_reply
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _ProactiveCfgView:
    """Минимальный «огрызок» конфига для _stream_reply: он трогает у cfg ТОЛЬКО
    .temperature и .max_tokens (handlers.py). Подкладываем сюда proactive-варианты,
    чтобы не менять сигнатуру _stream_reply и не расползаться по живым хендлерам.

    ВАЖНО: если внутри _stream_reply добавят чтение ещё полей cfg — сюда их тоже
    нужно будет прокинуть. Контракт в handlers._stream_reply."""
    temperature: float
    max_tokens: int


def _proactive_cfg_view(cfg: Config) -> _ProactiveCfgView:
    return _ProactiveCfgView(
        temperature=cfg.proactive_temperature,
        max_tokens=cfg.proactive_max_tokens,
    )


# ---------------------------------------------------------------------------
# Сборка сообщений и отправка
# ---------------------------------------------------------------------------

async def build_proactive_messages(
    user_id: int,
    cfg: Config,
    conn: "sqlite3.Connection",
    stickers: StickerStore | None,
    motive: str,
    silence_hours: float,
) -> list[dict]:
    """Аналог handlers._build_messages, но БЕЗ хвостового user-хода: system строит
    build_system_proactive (он сам говорит «ты пишешь первым»), + та же история.
    Никакого синтетического user-cue — система несёт ситуацию, первый токен
    модели — сам poking."""
    history = await run_db(
        db.get_recent_budget, conn, user_id, cfg.history_chars, cfg.history_window,
    )
    persona = await run_db(db.get_persona, conn, user_id)
    sticker_vibes = stickers.top_vibes(7) if stickers and not stickers.is_empty() else None

    messages: list[dict] = [
        {
            "role": "system",
            "content": prompts.build_system_proactive(
                persona.summary if persona else None, sticker_vibes, silence_hours, motive,
            ),
        }
    ]
    for row in history:
        role = row["role"]
        content = row["content"]
        if role == "user":
            messages.append({"role": "user", "content": content})
        else:
            messages.append({"role": "assistant", "content": content})
    # НЕТ trailing user-сообщения — модель должна начать сама.
    return messages


async def send_proactive(
    bot: Bot,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    cfg: Config,
    stickers: StickerStore | None,
    user_id: int,
    motive: str,
    silence_hours: float,
) -> None:
    """Собирает proactive-сообщения, стримит их в DM (chat_id=user_id) через
    _stream_reply, пишет пришедшие чанки в БД с kind='proactive'.

    emit_typing=False — не светим «печатает…» до первого чанка (юзер ничего не
    писал, индикатор на ровном месте выглядит как глюк; перед каждым реальным
    сообщением typing ставится всё равно). error_note=None — не шлём «сеть
    барахлит» при обрыве: proactive не должен вылезать с извинениями сам по себе."""
    messages = await build_proactive_messages(user_id, cfg, conn, stickers, motive, silence_hours)
    sent = await _stream_reply(
        bot, user_id, ai, messages, _proactive_cfg_view(cfg), stickers,
        emit_typing=False, error_note=None,
    )
    if not sent:
        logger.info(
            "proactive user=%s motive=%s: модель ничего не прислала, БД не пишу",
            user_id, motive,
        )
        return
    for chunk in sent:
        await run_db(db.add_message, conn, user_id, "assistant", "proactive", chunk)
    logger.info(
        "proactive отправлен user=%s motive=%s chunks=%d",
        user_id, motive, len(sent),
    )


# ---------------------------------------------------------------------------
# Фоновый цикл
# ---------------------------------------------------------------------------

async def proactive_loop(
    bot: Bot,
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    cfg: Config,
    stickers: StickerStore | None,
) -> None:
    """Просыпается раз в proactive_min_loop_min минут и проверяет всех юзеров: кому
    пора написать первым. При cfg.proactive=False — просто спит, ничего не делая
    (но живёт, чтобы можно было включить перезапуском и не пересоздавать задачу)."""
    logger.info(
        "proactive_loop запущен (enabled=%s, wake=%.2f min)",
        cfg.proactive, cfg.proactive_min_loop_min,
    )
    try:
        while True:
            if not cfg.proactive:
                await asyncio.sleep(cfg.proactive_min_loop_min * 60.0)
                continue

            now = int(time.time())
            try:
                activity = await run_db(db.last_activity_per_user, conn)
            except Exception:  # noqa: BLE001
                logger.exception("last_activity_per_user не выполнен, пропускаю виток")
                await asyncio.sleep(cfg.proactive_min_loop_min * 60.0)
                continue

            for user_id, last_ts in activity.items():
                try:
                    # Дёшево — без запросов: только что общались → мимо.
                    silence_h = (now - last_ts) / 3600.0
                    if silence_h < cfg.proactive_min_silence_h:
                        continue

                    last = await run_db(db.last_row, conn, user_id)
                    if last is None:
                        continue
                    motive = "silence" if last["role"] == "user" else "continuation"

                    # Дневной лимит (rolling 24h) — общий для обоих мотивов.
                    if (await run_db(db.proactive_count_24h, conn, user_id)) >= cfg.proactive_max_per_day:
                        continue

                    # Per-user троттл между пинками.
                    last_pro = await run_db(db.last_proactive_ts, conn, user_id)
                    if last_pro is not None and (now - last_pro) / 3600.0 < cfg.proactive_min_gap_h:
                        continue

                    # Рулетка — только после всех дешёвых проверок.
                    chance = poke_chance(
                        silence_h, cfg.proactive_min_silence_h, cfg.proactive_max_silence_h,
                    )
                    if random.random() >= chance:
                        continue

                    logger.info(
                        "proactive срабатывает user=%s silence=%.2fh motive=%s p=%.2f",
                        user_id, silence_h, motive, chance,
                    )
                    await send_proactive(
                        bot, conn, ai, cfg, stickers, user_id, motive, silence_h,
                    )
                except Exception:  # noqa: BLE001
                    logger.exception("proactive для user=%s упал, продолжаю", user_id)

            await asyncio.sleep(cfg.proactive_min_loop_min * 60.0)
    except asyncio.CancelledError:
        logger.info("proactive_loop отменён")
        raise
