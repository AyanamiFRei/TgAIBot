"""Точка входа Telegram-бота-собеседника на Cloudflare Workers AI.

Запуск:
    venv/Scripts/python.exe bot.py
(или через run.ps1 / run.bat)
"""
from __future__ import annotations

import asyncio
import logging
import os
import socket
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

    # Watchdog: если event loop завис (polling молчит, запрос висит без ответа),
    # systemd сам перезапустит сервис — процесс не должен висеть в epoll молча.
    # WORKING watchdog: SD_NOTIFY=1 в .env (на VPS), локально выключено —
    # sd_notify требует libsystemd, которого нет на Windows/локальном Python.
    if os.getenv("SD_NOTIFY") == "1" and hasattr(socket, "AF_UNIX"):
        try:
            import sdnotify  # type: ignore[import-not-found]
            _sd = sdnotify.SystemdNotifier()
            # WATCHDOG_USEC приходит из systemd в окружении процесса.
            try:
                wd_usec = int(os.getenv("WATCHDOG_USEC", "180000000"))
            except ValueError:
                wd_usec = 180000000
            # пинг дважды за период watchdog'а
            interval = max(wd_usec / 1e6 / 2, 5.0)

            async def _watchdog() -> None:
                while True:
                    _sd.notify("WATCHDOG=1")
                    await asyncio.sleep(interval)

            asyncio.create_task(_watchdog())
            _sd.notify("READY=1")
            logger.info("systemd watchdog включён (пинг каждые %.0f с)", interval)
        except Exception:  # noqa: BLE001
            logger.warning("sd_notify недоступен, watchdog выключен: %s", sys.exc_info()[1])
    else:
        logger.info("systemd watchdog выключен (SD_NOTIFY не задан)")

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
        if cfg.webhook_mode:
            await _run_webhook(bot, dp, cfg)
        else:
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


async def _run_webhook(bot: Bot, dp: Dispatcher, cfg: Config) -> None:
    """Приём апдейтов вебхуком вместо polling.

    Зачем: long-polling в aiogram 3.13 трижды зависал намертво (эпизоды 25-26.09:
    процесс жив, watchdog пингует, но getUpdates-соединение висит внутри aiohttp
    без таймаута — апдейты не идут, systemd этого не видит). С вебхуком Telegram
    сам присылает апдейты POST-запросами: зависший NIM-запрос блокирует один
    апдейт (handle_in_background=True), а не весь приём сообщений.

    Безопасность: secret_token в заголовке X-Telegram-Bot-Api-Secret-Token
    (Telegram подставляет его сам; всё остальное — 403). HTTPS — self-signed
    сертификат, публичная часть которого загружается в Telegram через
    set_webhook (официально поддерживаемый сценарий для IP-хостов).
    """
    from aiohttp import web
    from aiogram.webhook.aiohttp_server import (
        SimpleRequestHandler,
        setup_application,
    )

    app = web.Application()
    webhook_handler = SimpleRequestHandler(
        dispatcher=dp,
        bot=bot,
        handle_in_background=True,  # не держим Telegram ждущим долгий NIM-ответ
        secret_token=cfg.webhook_secret,
    )
    webhook_handler.register(app, path=cfg.webhook_path)
    setup_application(app, dp)

    runner = web.AppRunner(app)
    await runner.setup()
    site_kwargs = {}
    if cfg.webhook_cert_path:
        import ssl
        ssl_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_ctx.load_cert_chain(cfg.webhook_cert_path, cfg.webhook_key_path)
        site_kwargs["ssl_context"] = ssl_ctx
    site = web.TCPSite(
        runner,
        host=cfg.webhook_listen_host,
        port=cfg.webhook_port,
        **site_kwargs,
    )
    await site.start()
    logger.info(
        "Webhook-сервер запущен: %s:%s%s (secret: %s, cert: %s)",
        cfg.webhook_listen_host, cfg.webhook_port, cfg.webhook_path,
        "задан" if cfg.webhook_secret else "НЕТ",
        "да" if cfg.webhook_cert_path else "нет",
    )

    # Регистрируем вебхук в Telegram: URL + (для self-signed) публичный сертификат.
    # drop_pending_updates=False — накопившиеся за офлайн апдейты обрабатываем.
    webhook_url = cfg.webhook_public_url or (
        f"https://{cfg.webhook_host_ip}:{cfg.webhook_port}{cfg.webhook_path}"
    )
    cert_file = None
    if cfg.webhook_cert_path:
        from aiogram.types import InputFile
        cert_file = InputFile(open(cfg.webhook_cert_path, "rb"))
    try:
        await bot.set_webhook(
            webhook_url,
            certificate=cert_file,
            secret_token=cfg.webhook_secret or None,
            allowed_updates=dp.resolve_used_update_types(),
            drop_pending_updates=False,
        )
        logger.info("set_webhook OK: %s", webhook_url)
    finally:
        if cert_file:
            cert_file.close()

    # Висим, пока не придёт сигнал остановки (systemd SIGTERM → grpcace через
    # loop.add_signal_handler в polling-режиме нет — тут ставим свой хендлер).
    import signal
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for sig in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(sig, stop_event.set)
        except NotImplementedError:  # Windows
            pass
    logger.info("Run webhook for bot @%s", (await bot.me()).username)
    await stop_event.wait()

    # Снятие вебхука при остановке не делаем: при следующем старте он
    # перезапишется, а Telegram и так ретраит недоставленные апдейты.
    await runner.cleanup()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except (KeyboardInterrupt, SystemExit):
        pass
    except RuntimeError as exc:
        # config.get_config кидает понятную ошибку без токена.
        logger.error("%s", exc)
        sys.exit(1)
