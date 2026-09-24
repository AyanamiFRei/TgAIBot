"""Точка входа Telegram-бота-собеседника на Cloudflare Workers AI.

Запуск:
    venv/Scripts/python.exe bot.py
(или через run.ps1 / run.bat)
"""
from __future__ import annotations

import asyncio
import logging
import sys

from aiogram import Bot, Dispatcher
from aiogram.client.default import DefaultBotProperties
from aiogram.fsm.storage.memory import MemoryStorage

import db
import handlers
import proactive
import stickers as stickers_module
from config import get_config
from cloudflare_client import build_ai_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("tg-ai-chat")


async def main() -> None:
    cfg = get_config()
    bot = Bot(
        token=cfg.bot_token,
        default=DefaultBotProperties(parse_mode=None),  # plain текст, как живой человек
    )

    conn = db.connect(cfg.db_full_path)
    # Текстовый провайдер выбирается в .env через AI_PROVIDER=cloudflare|mistral —
    # build_ai_client сам подставит нужные base_url/api_key/model.
    ai = build_ai_client(cfg)
    # Vision-провайдер (фото/стикеры) переключается отдельно через VISION_PROVIDER
    # и может отличаться от текстового. Если совпадает — переиспользуем тот же
    # клиент (не плодим лишний httpx.AsyncClient и лишнее соединение на закрытие).
    if cfg.vision_provider == cfg.ai_provider:
        vision_ai = ai
    else:
        vision_ai = build_ai_client(cfg, provider=cfg.vision_provider)

    # Стикеры, которые бот сможет присылать. Грузим из публичного пака (если задан
    # STICKER_PACK). Без пака — StickerStore остаётся пустым, бот работает без стикеров.
    sticker_store = stickers_module.StickerStore()
    await sticker_store.load_from_pack(bot, cfg.sticker_pack)

    # MemoryStorage — явно, для читаемости. Нужно для FSM-состояния выбора спикера
    # в /persona (см. handlers.PersonaChoice). Aiogram по умолчанию и так создал бы
    # MemoryStorage, но явное указание делает зависимость от in-memory хранения
    # заметной: на рестарте воркера незавершённый выбор спикера сбрасывается — юзер
    # просто переотправит файл. На проде с долгоживущими сессиями стоит RedisStorage.
    dp = Dispatcher(storage=MemoryStorage())
    handlers.setup(dp, bot, conn, ai, cfg, sticker_store, vision_ai=vision_ai)

    model_by_provider = {
        "cloudflare": cfg.cloudflare_model,
        "mistral": cfg.mistral_model,
        "nvidia": cfg.nvidia_model,
    }
    active_model = model_by_provider[cfg.ai_provider]
    vision_model = model_by_provider[cfg.vision_provider]
    logger.info(
        "Бот запущен. Текст: %s (%s) | vision: %s (%s) | стикеры: %s",
        cfg.ai_provider, active_model, cfg.vision_provider, vision_model,
        "выкл" if sticker_store.is_empty() else f"{len(sticker_store.available_vibes)} виб",
    )
    # Фоновый цикл proactive-сообщений (бот иногда пишет первым). Opt-in — внутри
    # цикла сам проверяет cfg.proactive; когда выключено, просто спит. Отменяем в
    # finally ДО conn.close(): внутри цикла может висеть pending run_db по conn.
    proactive_task = asyncio.create_task(
        proactive.proactive_loop(bot, conn, ai, cfg, sticker_store)
    )
    try:
        await dp.start_polling(bot)
    finally:
        proactive_task.cancel()
        try:
            await proactive_task
        except (asyncio.CancelledError, Exception):  # noqa: BLE001, PERF203
            pass
        await ai.aclose()
        if vision_ai is not ai:
            await vision_ai.aclose()
        conn.close()
        logger.info("Бот остановлен.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
    except RuntimeError as exc:
        # config.get_config кидает понятную ошибку без токена.
        logger.error("%s", exc)
        sys.exit(1)
