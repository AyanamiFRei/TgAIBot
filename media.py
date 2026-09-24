"""Скачивание медиа из Telegram для мультимодального сообщения Gemma.

Фото или миниатюра стикера преобразуется в PNG data URI и входит прямо в
текущий запрос вместе с историей и подписью пользователя.
"""
from __future__ import annotations

import base64
import io
import logging
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from aiogram import Bot
    from aiogram.types import Message

logger = logging.getLogger(__name__)

# Ужимаем картинку до этого размера по большей стороне — экономим токены и трафик.
MAX_SIDE = 768


async def download_to_bytes(bot: "Bot", file_id: str) -> bytes:
    buf = await bot.download(file_id, destination=io.BytesIO())
    # bot.download возвращает BinaryIO (BytesIO, seek=0)
    return buf.getvalue()  # type: ignore[union-attr]


def to_png_base64(data: bytes) -> str | None:
    """Конвертируем любые байты-картинку в base64 data-URI PNG. Не картинка — None."""
    try:
        img = Image.open(io.BytesIO(data))
    except Exception:  # noqa: BLE001
        logger.warning("не удалось открыть как картинку (%d байт)", len(data))
        return None
    img.load()
    if img.mode not in ("RGB", "L", "P"):
        img = img.convert("RGB")
    img.thumbnail((MAX_SIDE, MAX_SIDE))
    out = io.BytesIO()
    img.save(out, format="PNG")
    b64 = base64.b64encode(out.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def image_content_block(image_data_uri: str, provider: str) -> dict:
    """Формат content-блока с картинкой для Chat Completions API — зависит от
    провайдера:
      - cloudflare и nvidia (OpenAI-совместимый формат): {"type": "image_url",
        "image_url": {"url": data_uri}}
      - mistral: их документация и примеры используют image_url КАК СТРОКУ
        (без вложенного объекта {"url": ...}): {"type": "image_url",
        "image_url": data_uri}
    Если провайдер сменится/добавится новый — эта функция единственное место,
    где нужно поправить формат."""
    if provider == "mistral":
        return {"type": "image_url", "image_url": image_data_uri}
    return {"type": "image_url", "image_url": {"url": image_data_uri}}


async def prepare_media(
    bot: "Bot", message: "Message"
) -> tuple[str | None, str, str | None]:
    """Готовим медиа-сообщение для единой мультимодальной модели.

    Возвращает:
      image_data_uri | None  — картинка в base64 data URI (None, если не вышло)
      kind                   — 'фото' | 'стикер'
      caption_text | None    — дополнительный текст от юзера (caption к фото/подпись к стикеру),
                               None если подписи не было
    """
    if message.photo:
        biggest = message.photo[-1]  # PhotoSize — наибольший размер
        try:
            raw = await download_to_bytes(bot, biggest.file_id)
        except Exception:  # noqa: BLE001
            logger.warning("не удалось скачать фото")
            raw = b""
        data_uri = to_png_base64(raw) if raw else None
        caption = (message.caption or "").strip()
        return data_uri, "фото", (caption or None)

    if message.sticker:
        st = message.sticker
        # Берём статичный thumbnail (webp/jpg) — он надёжно читается PIL.
        # Анимированные (.tgs) и видео (.webm) стикеры напрямую PIL не откроет.
        target_file_id: str | None = None
        if st.thumbnail is not None:
            target_file_id = st.thumbnail.file_id
        data_uri = None
        if target_file_id is not None:
            try:
                raw = await download_to_bytes(bot, target_file_id)
            except Exception:  # noqa: BLE001
                logger.warning("не удалось скачать thumbnail стикера")
                raw = b""
            data_uri = to_png_base64(raw) if raw else None
        # Стикер реагирует «эмоцией»: э gentle user_text = emoji стикера (если есть).
        caption_text = None
        if st.emoji:
            caption_text = f"эмодзи стикера: {st.emoji}"
        elif st.set_name:
            caption_text = f"набор «{st.set_name}»"
        return data_uri, "стикер", caption_text

    return None, "", None
