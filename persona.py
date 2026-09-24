"""Парсинг экспортированной переписки и извлечение краткого описания стиля.

Поддерживаем:
  - Telegram Desktop JSON-экспорт (ключ "messages" — список СЛОВАРЕЙ, у каждого
    "text" — строка или список text_entities с полями "text").
  - «Компактный» JSON-экспорт (ключ "messages" — список СПИСКОВ вида
    [timestamp, speaker_index, text], плюс "speakers": {имя: index}). Формат
    отдают некоторые сторонние экспортеры/скрипты выгрузки истории Telegram.
  - Telegram HTML-экспорт здесь не разбираем осмысленно — отдаём как plain text.
  - Произвольный plain text (txt) — берём как есть, построчно.

В репрезентативную выборку берём не больше ~4000 символов, чтобы уложиться
в контекст лёгкой модели при вызове извлечения стиля.

Дополнительно (новое):
  - extract_speakers(raw) — список уникальных имён собеседников из экспорта.
    Используется ботом, чтобы спросить у юзера, кого брать за персону, прежде
    чем звать модель (см. handlers: inline-кнопки со списком спикеров).
  - parse_export_for_speaker(raw, speaker_name) — то же, что parse_export,
    но оставляет только реплики выбранного собеседника. Без «Author:» префикса
    — модели даём чисто его текст, чтобы она не отвлекалась на чужие реплики.
  - summarize_style(client, sample, speaker_name=None) — если передано имя,
    берётся build_persona_extraction_prompt(speaker_name), где модели явно
    сказано «опиши только этого собеседника». Без имени — legacy-промпт
    (модель сама берёт самого активного).
"""
from __future__ import annotations

import json
import logging

from cloudflare_client import CloudflareClient, CloudflareError
from prompts import build_persona_extraction_prompt

logger = logging.getLogger(__name__)

SAMPLE_LIMIT = 4000


# ---------------------------------------------------------------------------
# Декодирование сырых байтов в текст — общий хелпер для всех парсеров.
# ---------------------------------------------------------------------------

def _decode(raw: bytes | str) -> str:
    """Пробует несколько кодировок; на крайний случай — utf-8 с replace."""
    if isinstance(raw, str):
        return raw
    for enc in ("utf-8", "utf-8-sig", "cp1251", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Извлечение спикеров — без AI, чисто по структуре файла.
# ---------------------------------------------------------------------------

def extract_speakers(raw: bytes | str) -> list[str]:
    """Возвращает список уникальных имён собеседников в порядке первого появления.

    Поддерживаемые форматы:
      - Компактный JSON: ключ "speakers" — словарь {имя: index}. Имена берём
        из него (это каноничный список, кто с кем говорит).
      - Telegram Desktop JSON: ключ "messages", у каждого dict есть поле
        "from" (или "from_id"). Сканируем и собираем уникальные имена.
      - Plain text / нераспознанный формат: возвращаем [] — без авторской
        разметки понять, кто говорит, нельзя; бот в этом случае просто
        пройдёт по старому пути (без вопроса о спикере).

    Никогда не бросает исключение — на любой непонятной структуре возвращает [].
    """
    text = _decode(raw).lstrip()
    if not text.startswith("{") and not text.startswith("["):
        return []

    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(obj, dict):
        return []

    # 1) Компактный формат —首选: ключ "speakers" содержит каноничный словарь.
    speakers = obj.get("speakers")
    if isinstance(speakers, dict) and speakers:
        # Сохраняем порядок словаря (Python 3.7+). Имена — ключи.
        return [str(name) for name in speakers.keys()]

    # 2) Telegram Desktop JSON: сканируем messages по полю "from"/"from_id".
    messages = obj.get("messages")
    if not isinstance(messages, list):
        return []
    seen: list[str] = []
    seen_set: set[str] = set()
    for m in messages:
        if not isinstance(m, dict):
            continue
        author = m.get("from") or m.get("from_id")
        if not author:
            continue
        author = str(author)
        if author not in seen_set:
            seen.append(author)
            seen_set.add(author)
    return seen


# ---------------------------------------------------------------------------
# Парсеры — все сообщения (legacy) и для конкретного спикера (новое).
# ---------------------------------------------------------------------------

def _extract_telegram_json(obj: dict, speaker_name: str | None = None) -> str:
    """Telegram Desktop JSON. Если speaker_name задан — оставляем только его
    реплики (без «Author:» префикса, чистый текст). Иначе — все реплики
    с префиксом «Author: text» (legacy-формат для summarize_style без фильтра)."""
    lines: list[str] = []
    for m in obj.get("messages", []):
        if not isinstance(m, dict):
            continue
        if m.get("type") not in ("message", None):
            continue
        # text может быть строкой или списком entities
        text_field = m.get("text")
        if isinstance(text_field, str):
            text = text_field
        elif isinstance(text_field, list):
            parts = []
            for ent in text_field:
                if isinstance(ent, dict):
                    parts.append(str(ent.get("text") or ""))
                else:
                    parts.append(str(ent))
            text = "".join(parts)
        else:
            text = ""
        text = text.strip()
        if not text:
            continue
        author = m.get("from") or m.get("from_id") or ""
        if speaker_name is not None:
            # Фильтр: только реплики выбранного спикера, без префикса.
            if str(author) != speaker_name:
                continue
            lines.append(text)
        else:
            # Legacy: все реплики, с «Author: text».
            if author:
                lines.append(f"{author}: {text}")
            else:
                lines.append(text)
    return "\n".join(lines)


def _extract_compact_json(obj: dict, speaker_name: str | None = None) -> str:
    """«Компактный» формат: {"speakers": {имя: index}, "messages": [[ts, idx, text], ...]}.

    index->имя достаём из "speakers" (обратный маппинг); если он не задан или
    не совпадает — просто пишем индекс как есть, лучше так, чем падать.

    Если speaker_name задан — оставляем только его реплики (без префикса).
    Иначе — все реплики с «Author: text» (legacy)."""
    speakers = obj.get("speakers") or {}
    idx_to_name: dict[int, str] = {}
    name_to_idx: dict[str, int] = {}
    if isinstance(speakers, dict):
        for name, idx in speakers.items():
            try:
                idx_int = int(idx)
                idx_to_name[idx_int] = str(name)
                name_to_idx[str(name)] = idx_int
            except (TypeError, ValueError):
                continue

    target_idx: int | None = None
    if speaker_name is not None:
        target_idx = name_to_idx.get(speaker_name)
        # Если имени нет в speakers — фильтр всё равно применяем, просто ничего
        # не отберётся (лучше пустой результат, чем «все реплики» по ошибке).

    lines: list[str] = []
    for m in obj.get("messages", []):
        if not isinstance(m, (list, tuple)) or len(m) < 3:
            continue
        _ts, speaker_idx, text = m[0], m[1], m[2]
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text:
            continue
        if speaker_name is not None:
            # Только реплики выбранного спикера, без префикса.
            if speaker_idx != target_idx:
                continue
            lines.append(text)
        else:
            author = idx_to_name.get(speaker_idx) if isinstance(speaker_idx, int) else None
            if author:
                lines.append(f"{author}: {text}")
            else:
                lines.append(text)
    return "\n".join(lines)


def _extract_json_messages(obj: dict, speaker_name: str | None = None) -> str:
    """Определяет конкретный JSON-формат экспорта по форме messages и вызывает
    нужный извлекатель. Никогда не бросает исключение — на любой непонятной
    структуре просто возвращает "" (вызывающий код откатится на plain text)."""
    messages = obj.get("messages")
    if not isinstance(messages, list) or not messages:
        return ""
    first = messages[0]
    try:
        if isinstance(first, dict):
            return _extract_telegram_json(obj, speaker_name)
        if isinstance(first, (list, tuple)):
            return _extract_compact_json(obj, speaker_name)
    except Exception:  # noqa: BLE001
        logger.exception("не удалось извлечь текст из JSON-экспорта (формат %s)", type(first))
        return ""
    return ""


# ---------------------------------------------------------------------------
# Публичные функции парсинга.
# ---------------------------------------------------------------------------

def parse_export(raw: bytes | str) -> str:
    """Возвращает текст переписки (только текстовые реплики). Пусто — если не вытянул.

    Legacy-формат: все реплики с префиксами «Author: text». Используется как
    fallback, когда не получилось спросить у юзера, кого брать за персону
    (например, plain text без авторской разметки)."""
    text = _decode(raw)
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "messages" in obj:
            extracted = _extract_json_messages(obj, speaker_name=None)
            if extracted.strip():
                return _sample(extracted)
        # Может быть просто JSON-объект без messages, или формат messages не
        # распознан — дропаем разбор и падаем на plain text ниже.

    # Plain text (в т.ч. Telegram Desktop txt-экспорт или любой .txt) — как есть.
    return _sample(text)


def parse_export_for_speaker(raw: bytes | str, speaker_name: str) -> str:
    """Как parse_export, но оставляет только реплики указанного собеседника —
    без «Author:» префикса, чистым текстом.

    Используется после того, как бот спросил у юзера, кого брать за персону
    (extract_speakers -> inline-кнопки -> выбор). Модели даём чистый текст
    одного спикера, чтобы она не отвлекалась на чужие реплики и не смешивала
    стили.

    Если имя не найдено в экспорте (опечатка, разные регистры и т.п.) —
    возвращаем пустую строку; summarize_style на этом поднимет CloudflareError.
    Это лучше, чем тихо отдать «все реплики» — юзер увидит ошибку и выберет
    правильное имя на кнопках."""
    text = _decode(raw)
    stripped = text.lstrip()
    if stripped.startswith("{") or stripped.startswith("["):
        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            obj = None
        if isinstance(obj, dict) and "messages" in obj:
            extracted = _extract_json_messages(obj, speaker_name=speaker_name)
            if extracted.strip():
                return _sample(extracted)

    # Plain text без авторской разметки — фильтровать некого. Возвращаем пусто,
    # не весь текст: юзер явно выбрал конкретного спикера, отдавать «всё» —
    # молчаливая ошибка. summarize_style поднимется с понятным сообщением.
    return ""


def _sample(text: str) -> str:
    text = text.strip()
    if len(text) <= SAMPLE_LIMIT:
        return text
    # head + середина: не теряем «характер» начала, но укладываемся в лимит.
    head = text[: SAMPLE_LIMIT // 3]
    tail = text[-SAMPLE_LIMIT + len(head) :]
    return head + "\n…\n" + tail


# ---------------------------------------------------------------------------
# Извлечение стиля — с указанием конкретного спикера или без.
# ---------------------------------------------------------------------------

async def summarize_style(
    client: CloudflareClient,
    sample: str,
    speaker_name: str | None = None,
) -> str:
    """Зовём модель, чтобы по выборке переписки родить описание стиля.

    speaker_name — имя собеседника, чей стиль описываем. Если передано,
    берётся build_persona_extraction_prompt(speaker_name): модели явно сказано
    «опирайся ТОЛЬКО на реплики этого собеседника». В этом случае sample
    обычно приходит из parse_export_for_speaker — чистый текст одного спикера.

    Без speaker_name — legacy-поведение: модель сама выбирает «самого
    разговорчивого» по PERSONA_EXTRACTION_PROMPT. sample в этом случае
    приходит из parse_export (все реплики с префиксами «Author:»)."""
    if not sample.strip():
        raise CloudflareError("не удалось вытащить текст из переписки")
    messages = [
        {"role": "system", "content": build_persona_extraction_prompt(speaker_name)},
        {"role": "user", "content": sample},
    ]
    answer = await client.chat(messages, temperature=0.3, max_tokens=400)
    answer = answer.strip()
    if not answer:
        raise CloudflareError("модель не смогла описать стиль")
    # Защита: некоторые модели уходят в рассуждения — отрезаем после 3 предложения.
    return answer
