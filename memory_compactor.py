"""Долговременная память фактами + фоновая кураторка (замена плоского автокомпакта).

Зачем: прежняя память — одна плоская сводка (~1500 симв), регенерируемая целиком
каждые 5 минут из пересказа переписки; факты тонули в пересказах и терялись при
каждой перегенерации. Новая схема (референс — ChatGPT Dreaming V3):
  - память хранится ФАКТАМИ (db.memory_facts: один факт — одна строка с категорией);
  - кураторка инкрементальная: текущие факты + только непокрытая переписка
    (не весь транскрипт) → модель возвращает СТРОГИЙ JSON-дифф (add/update/drop);
  - дифф применяется SQL'ем точечно — факты не теряются, ответ модели крошечный;
  - compacted_until двигается вперёд, свежий хвост продолжает жить в диалоге.

Быстрый путь «запомни: X» — в handlers.handle_text: факт пишется сразу, без
ожидания фонового витка.

build_memory_block — хелпер инжекции фактов в диалог (используют db.
get_recent_budget → handlers._build_messages и proactive.build_proactive_messages).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import TYPE_CHECKING

import db
from cloudflare_client import CloudflareClient

if TYPE_CHECKING:
    import sqlite3

logger = logging.getLogger(__name__)

# Категории фактов: identity (имя, чем занимается), preferences (вкусы, привычки),
# projects (проекты, цели), relationships (отношения, динамика), ongoing (текущие
# темы, договорённости), events (важные события, обещания), legacy (старая
# flat-сводка после миграции — кураторка раскладывает её по категориям).
CATEGORIES = ("identity", "preferences", "projects", "relationships", "ongoing", "events")

CURATOR_PROMPT = (
    "Ты — редактор памяти. Ниже — текущая память (факты с id) и свежая переписка. "
    "Реши, что изменить в памяти на основе переписки. Верни СТРОГИЙ JSON без "
    "размышлений и пояснений:\n"
    '{"add": [{"category": "...", "content": "..."}], '
    '"update": [{"id": N, "content": "..."}], "drop": [id, ...]}\n\n'
    "Категории: identity (имя, пол, чем занимается), preferences (вкусы, привычки, "
    "как общается), projects (проекты, цели, планы), relationships (отношение к "
    "собеседнику, динамика), ongoing (текущие темы, договорённости, висящие вопросы), "
    "events (важные события, даты, обещания).\n\n"
    "Правила:\n"
    "- add — только новая ДОЛГОВРЕМЕННАЯ информация: факт о собеседнике, а не пересказ хода "
    "(«обсудили дропы» — не факт; «делает игру Look in Book, карточные бои» — факт)\n"
    "- update — уточни/заостри существующий факт (id обязателен)\n"
    "- drop — факт устарел или неверен (id в списке)\n"
    "- если legacy-факт есть — разложи его содержимое по категориям (add) и пометь сам legacy-факт на удаление (drop)\n"
    "- ничего не менять — верни все списки пустыми\n"
    "- факты кратко, по-русски, без воды"
)


# ---------------------------------------------------------------------------
# Инжекция фактов в диалог
# ---------------------------------------------------------------------------

# Русские заголовки категорий для блока памяти.
_CATEGORY_TITLES = {
    "identity": "О нём",
    "preferences": "Вкусы и привычки",
    "projects": "Проекты",
    "relationships": "Отношения",
    "ongoing": "Текущее",
    "events": "События",
    "legacy": "Из старой памяти",
}


def build_memory_block(facts: list) -> str:
    """Компактный блок памяти по категориям для инжекции в диалог.

    facts — строки из db.get_facts (id, category, content). Факты группируются
    по категориям, внутри категории — через «; ». Размер ~600-800 символов на
    юзера — только факты, без пересказа болтовни.
    """
    if not facts:
        return ""
    by_category: dict[str, list[str]] = {}
    for f in facts:
        cat = f["category"] if not isinstance(f, dict) else f["category"]
        content = f["content"] if not isinstance(f, dict) else f["content"]
        if content and content.strip():
            by_category.setdefault(cat, []).append(content.strip())

    lines: list[str] = []
    # Порядок категорий: CATEGORIES, потом legacy в конце (если ещё не разложена).
    for cat in list(CATEGORIES) + ["legacy"]:
        items = by_category.get(cat)
        if not items:
            continue
        title = _CATEGORY_TITLES.get(cat, cat)
        lines.append(f"{title}: " + "; ".join(items))
    if not lines:
        return ""
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Кураторка: инкрементальный дифф памяти
# ---------------------------------------------------------------------------

async def _run_db(fn, *args, **kwargs):
    """Обёртка handlers.run_db: синхронный db-вызов в отдельном потоке."""
    import asyncio as _a
    try:
        to_thread = _a.to_thread
    except AttributeError:  # Python 3.8: to_thread ещё нет
        import functools
        loop = _a.get_event_loop()
        return await loop.run_in_executor(None, functools.partial(fn, *args, **kwargs))
    return await to_thread(fn, *args, **kwargs)


def _parse_diff(raw: str) -> dict | None:
    """Парсит JSON-дифф из ответа модели. None — невалидный (виток пропускается,
    факты не теряются — они в БД). Терпим markdown-обёртку ```json ... ```."""
    text = raw.strip()
    if text.startswith("```"):
        # срезаем обёртку: первая строка ```json, последняя ```
        lines = text.splitlines()
        if len(lines) >= 2 and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None
    # Нормализуем: пустые/отсутствующие списки → []
    result = {"add": [], "update": [], "drop": []}
    adds = data.get("add")
    if isinstance(adds, list):
        for a in adds:
            if isinstance(a, dict) and isinstance(a.get("content"), str) and a["content"].strip():
                result["add"].append({
                    "category": str(a.get("category") or "legacy").strip(),
                    "content": a["content"].strip(),
                })
    updates = data.get("update")
    if isinstance(updates, list):
        for u in updates:
            if (isinstance(u, dict) and isinstance(u.get("id"), int)
                    and isinstance(u.get("content"), str) and u["content"].strip()):
                result["update"].append({"id": u["id"], "content": u["content"].strip()})
    drops = data.get("drop")
    if isinstance(drops, list):
        for d in drops:
            if isinstance(d, int):
                result["drop"].append(d)
    return result


async def curate_user_memory(
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    user_id: int,
    keep_recent: int,
    reasoning_effort: str | None = "low",
) -> bool:
    """Инкрементальная кураторка памяти юзера. True — память обновлена.

    keep_recent — сколько ПОСЛЕДНИХ сообщений оставить вне кураторки (свежий
    хвост, он продолжает жить в обычном диалоге). reasoning_effort — 'low' для
    NIM glm-5.3-flash (без размышлений; проверено: NIM уважает reasoning_effort).
    """
    last = await _run_db(db.last_row, conn, user_id)
    if last is None:
        return False
    last_id = last["id"]

    memory = await _run_db(db.get_memory, conn, user_id)
    compacted_until = memory["compacted_until"] if memory else 0

    # Непокрытых сообщений достаточно? (после compacted_until, минус свежий хвост)
    pending = await _run_db(
        db.messages_to_compact, conn, user_id, compacted_until, last_id,
    )
    if len(pending) <= keep_recent:
        return False

    # Кураторим всё, кроме свежего хвоста.
    to_compact = pending[:-keep_recent] if keep_recent else pending
    if not to_compact:
        return False

    facts = await _run_db(db.get_facts, conn, user_id)

    # Транскрипт только непокрытой части (НЕ весь, НЕ с нуля).
    lines: list[str] = []
    for row in to_compact:
        who = "собеседник" if row["role"] == "user" else "ты"
        content = (row["content"] or "").strip()
        if content:
            lines.append(f"{who}: {content}")
    transcript = "\n".join(lines)
    if not transcript:
        return False

    user_prompt = CURATOR_PROMPT
    if facts:
        facts_lines = [
            f"[{f['id']}] ({f['category']}) {f['content']}" for f in facts
        ]
        user_prompt += "\n\nТекущая память (факты с id):\n" + "\n".join(facts_lines)
    user_prompt += "\n\nСвежая переписка:\n" + transcript

    try:
        raw = await ai.chat(
            [
                {"role": "system", "content": "Ты — редактор памяти. Отвечай только строгим JSON без размышлений."},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,          # точность важнее живости: фоновая память
            max_tokens=1024,
            model=None,               # основная модель
            reasoning_effort=reasoning_effort,  # 'low' — без размышлений
        )
    except Exception:  # noqa: BLE001
        logger.exception("кураторка памяти user=%s не удалась", user_id)
        return False

    diff = _parse_diff(raw)
    if diff is None:
        logger.warning(
            "кураторка user=%s: невалидный JSON в ответе, виток пропущен (факты не тронуты): %r",
            user_id, raw[:200],
        )
        return False

    # Применяем дифф SQL'ем точечно.
    added = updated = dropped = 0
    for a in diff["add"]:
        await _run_db(db.add_fact, conn, user_id, a["category"], a["content"])
        added += 1
    for u in diff["update"]:
        await _run_db(db.update_fact, conn, u["id"], u["content"])
        updated += 1
    if diff["drop"]:
        await _run_db(db.drop_facts, conn, diff["drop"])
        dropped = len(diff["drop"])

    new_until = to_compact[-1]["id"]
    logger.info(
        "кураторка user=%s: +%d/обновлено %d/удалено %d фактов, компакт-покрытие до id=%d",
        user_id, added, updated, dropped, new_until,
    )
    return True


# ---------------------------------------------------------------------------
# Фоновый цикл
# ---------------------------------------------------------------------------

async def memory_compactor_loop(
    conn: "sqlite3.Connection",
    ai: CloudflareClient,
    keep_recent: int,
    wake_min: float,
    reasoning_effort: str | None = "low",
) -> None:
    """Просыпается раз в wake_min минут и проверяет всех юзеров: у кого накопилась
    непокрытая история — кураторит память инкрементальным диффом."""
    logger.info(
        "memory_compactor запущен (wake=%.1f min, keep_recent=%d, effort=%s)",
        wake_min, keep_recent, reasoning_effort or "нет",
    )
    try:
        while True:
            try:
                activity = await _run_db(db.last_activity_per_user, conn)
                for user_id in activity:
                    try:
                        await curate_user_memory(
                            conn, ai, user_id, keep_recent, reasoning_effort,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception("кураторка user=%s упала, продолжаю", user_id)
            except Exception:  # noqa: BLE001
                logger.exception("memory_compactor виток упал")
            await asyncio.sleep(wake_min * 60.0)
    except asyncio.CancelledError:
        logger.info("memory_compactor отменён")
        raise
