"""Стикеры, которые БОТ умеет отправлять.

Источник — публичный стикерпак Telegram. При старте бот дёргает
`bot.get_sticker_set(STICKER_PACK_NAME)`, забирает у каждого стикера его `file_id`
(он стабилен и переиспользуется в `send_sticker`) и сажает стикер на «вибу» по его
эму (авторазметка). Дальше модель в ответе вставляет маркер `[sticker: виб]`,
handlers.parse_stickers вырезает маркеры и отправляет стикеры через file_id.

Если пак не задан или не загрузился — модуль просто пустой, бот работает как
обычно (без стикеров). Это graceful: ничего не падает.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aiogram import Bot

logger = logging.getLogger(__name__)

# Карта «emoji -> человеческая виба». Используется при авто-разметке стикеров
# из пака по их Sticker.emoji. Если emoji не в карте — стикер не привязывается
# к вибе и модель не сможет его вызвать маркером. Чем точнее карта покрывает
# emoji целевого пака, тем разнообразнее и осмысленнее стикеры у бота.
#
# Вибы — короткие русские слова-настроения, которые в системном промпте
# перечисляются модели как «доступные вибы». Модель берёт подходящую по контексту.
EMOJI_TO_VIBE: dict[str, str] = {
    # улыбка / радость
    "😀": "радость", "😄": "радость", "😁": "радость", "😊": "радость",
    "🙂": "радость", "😉": "подмигивание", "🙃": "ирония",
    # озорство / довольство собой (отличаем от обычной радости)
    "😏": "самодовольство", "🤓": "умник",
    # смех
    "😂": "смех", "🤣": "смех", "😆": "смех", "😹": "смех",
    # неловкость / пот-лицо
    "😅": "неловко", "😬": "неловко", "🫠": "неловко",
    # удивление / недоумение
    "😮": "ого", "😯": "ого", "😱": "ого", "😨": "ого",
    "😳": "смущение", "🥴": "смущение",
    # задумчивость / скепсис
    "🤔": "хм", "🤨": "хм", "😕": "недоумение", "🙁": "недоумение",
    # раздражение / пренебрежение
    "🙄": "закат-глаз", "😒": "недовольство", "😤": "раздражение",
    "😠": "злость", "😡": "ярость",
    # грусть / вина / подавленность (разные оттенки)
    "😢": "грусть", "😭": "горькие-слёзы", "🥺": "грусть",
    "😔": "уныние", "😟": "тревога", "😞": "разочарование",
    "😓": "испарина", "😰": "паника", "😥": "облегчение-со-вздохом",
    "😧": "мучение", "🤕": "пострадал", "💔": "разбитое-сердце",
    # усталость / сон
    "😴": "сон", "💤": "сон",
    # фейспалм / незнаю / привет
    "🤦": "фейспалм", "🤦‍♂️": "фейспалм", "🤦‍♀️": "фейспалм",
    "🤷": "незнаю", "🤷‍♂️": "незнаю", "🤷‍♀️": "незнаю",
    "👋": "привет", "✊": "решимость",
    # одобрение / позитив
    "👍": "ок", "👌": "ок", "🆗": "ок",
    "🔥": "огонь", "💪": "класс",
    # сердце / любовь
    "❤": "сердце", "❤️": "сердце", "💜": "сердце", "💖": "сердце",
    "💕": "сердце", "😍": "сердце", "🥰": "сердце", "😘": "сердце",
    # всякое
    "🍵": "чифир", "🤝": "класс",
}


@dataclass
class StickerEntry:
    vibe: str
    file_id: str
    emoji: str


class StickerStore:
    """Хранилищеcción виба -> [StickerEntry]. Бот берёт стикер по вибе."""

    def __init__(self) -> None:
        # vibe -> list[StickerEntry]  (может быть несколько стикеров на одну вибу)
        self._by_vibe: dict[str, list[StickerEntry]] = {}
        # все стикеры пака (для логов / fallback)
        self._all: list[StickerEntry] = []

    @property
    def available_vibes(self) -> list[str]:
        return sorted(self._by_vibe.keys())

    def is_empty(self) -> bool:
        return not self._by_vibe

    async def load_from_pack(self, bot: "Bot", pack_name: str) -> None:
        """Загружаем стикеры из публичного пака при старте. Ничего не выбрасывает —
        если пак недоступен, фиксируем в лог и остаёмся пустым."""
        if not pack_name:
            logger.info("Стикерпак не задан (STICKER_PACK пуст) — стикеры выключены")
            return
        try:
            sticker_set = await bot.get_sticker_set(pack_name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не загрузить стикерпак %r: %s — стикеры выключены", pack_name, exc)
            return

        count = 0
        for st in sticker_set.stickers:
            emoji = st.emoji or ""
            #/video/animated тоже можно слать через file_id — Telegram сам их отрисует.
            vibe = EMOJI_TO_VIBE.get(emoji.strip())
            entry = StickerEntry(vibe=vibe or "", file_id=st.file_id, emoji=emoji)
            self._all.append(entry)
            if vibe:
                self._by_vibe.setdefault(vibe, []).append(entry)
            count += 1

        logger.info(
            "Стикеры: загружено %d из пака %r, размечено виб: %s",
            count, pack_name,
            ", ".join(f"{v}({len(es)})" for v, es in sorted(self._by_vibe.items())) or "—",
        )

    def get_file_id(self, vibe: str) -> str | None:
        """file_id первого (или случайного) стикера с этой вибой.
        Возвращает None, если виба нет — handlers просто пропустит стикер."""
        entries = self._by_vibe.get(vibe)
        if not entries:
            return None
        # Если на вибу несколько — берём случайный, чтобы не было монотонно одного.
        return random.choice(entries).file_id

    def describe_for_prompt(self) -> str:
        """Строка с доступными вибами — для вставки в системный промпт модели."""
        if not self._by_vibe:
            return ""
        return ", ".join(self.available_vibes)

    def top_vibes(self, n: int = 7) -> list[str]:
        """Топ-N виб по числу стикеров в пачке — те настроения, которые представлены
        наиболее часто. Их и подаём модели в промпт (компактный осмысленный список),
        а не все 20+ виб сразу. Если модель всё же вставит вибу вне топа — бот найдёт
        её через get_file_id по полному словарю, стикер уйдёт."""
        sized = sorted(self._by_vibe.items(), key=lambda kv: len(kv[1]), reverse=True)
        return [v for v, _ in sized[:n]]
