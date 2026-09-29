"""Хранилище: история переписки и активный persona per user. sqlite3 stdlib.

Соединение одно, открывается в startup бота. Запросы идут синхронно —
оборачиваем вызовы в asyncio.to_thread, чтобы не блокировать event loop.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass

logger = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id    INTEGER PRIMARY KEY,
    created_at INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id    INTEGER NOT NULL,
    role       TEXT    NOT NULL,   -- 'user' | 'assistant'
    kind       TEXT    NOT NULL,   -- 'text' | 'image' | 'sticker'
    content    TEXT    NOT NULL,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

CREATE INDEX IF NOT EXISTS idx_messages_user_time
    ON messages (user_id, id);

CREATE TABLE IF NOT EXISTS personas (
    user_id    INTEGER PRIMARY KEY,
    name       TEXT    NOT NULL,
    summary    TEXT    NOT NULL,
    source_raw TEXT,
    updated_at INTEGER NOT NULL
);

-- Автокомпакт памяти: старые сообщения сжимаются моделью в краткие факты.
-- compacted_until = id последнего сообщения, которое ВОШЛО в память.
CREATE TABLE IF NOT EXISTS memories (
    user_id         INTEGER PRIMARY KEY,
    content         TEXT    NOT NULL,   -- краткое изложение важного (память)
    compacted_until INTEGER NOT NULL,   -- id последнего вошедшего в память сообщения
    updated_at      INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

-- Долговременная память фактами (замена плоской memories-сводки): один факт —
-- одна строка с категорией. Кураторка обновляет точечно (add/update/drop),
-- факты не теряются при перегенерации. category='legacy' — старая flat-сводка
-- после миграции, кураторка раскладывает её по категориям на первом витке.
CREATE TABLE IF NOT EXISTS memory_facts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       INTEGER NOT NULL,
    category      TEXT    NOT NULL,  -- identity|preferences|projects|relationships|ongoing|events|legacy
    content       TEXT    NOT NULL,  -- один факт, кратко
    source_msg_id INTEGER,           -- id сообщения-источника (provenance), может быть NULL
    created_at    INTEGER NOT NULL,
    updated_at    INTEGER NOT NULL,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);

CREATE INDEX IF NOT EXISTS idx_facts_user ON memory_facts (user_id, category);
"""


@dataclass
class Persona:
    name: str
    summary: str


def connect(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    # Разделяем одно соединение между event-loop (хендлеры) и фоновым proactive-циклом.
    # Запросы идут через asyncio.to_thread, поэтому конкурентные writes возможны
    # (хендлер пишет реплику, а proactive同期 пишет свой чанк). busy_timeout даёт
    # sqlite до 5с подождать освобождения блокировки вместо моментального
    # «database is locked», что нормально для низконагруженного личного бота.
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.executescript(SCHEMA)
    _migrate_flat_memory(conn)
    conn.commit()
    return conn


def _migrate_flat_memory(conn: sqlite3.Connection) -> None:
    """Миграция: старая flat-сводка из memories → один legacy-факт в memory_facts.

    Сводку НЕ парсим (она смесь фактов и пересказов — NLP-парсинг хрупок):
    записываем целиком одним фактом category='legacy', кураторка на первом витке
    сама выделит факты и разложит по категориям. compacted_until остаётся в
    memories и продолжает фильтровать хвост истории тем же кодом. Idempotent:
    legacy-факт на юзера один (проверка перед вставкой)."""
    rows = conn.execute(
        "SELECT m.user_id, m.content FROM memories m "
        "WHERE m.content != '' AND NOT EXISTS ("
        "  SELECT 1 FROM memory_facts f "
        "  WHERE f.user_id = m.user_id AND f.category = 'legacy')"
    ).fetchall()
    now = int(time.time())
    for row in rows:
        conn.execute(
            "INSERT INTO memory_facts (user_id, category, content, source_msg_id, created_at, updated_at) "
            "VALUES (?, 'legacy', ?, NULL, ?, ?)",
            (row["user_id"], row["content"], now, now),
        )
        logger.debug("миграция: flat-память user=%s → legacy-факт", row["user_id"])
    if rows:
        conn.commit()


def ensure_user(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute(
        "INSERT OR IGNORE INTO users (user_id, created_at) VALUES (?, ?)",
        (user_id, int(time.time())),
    )
    conn.commit()


def is_dup_user_text(
    conn: sqlite3.Connection,
    user_id: int,
    content: str,
    window_s: int = 30,
) -> bool:
    """Был ли точно такой же текстовый ход от user_id за последние window_s секунд?

    Telegram в режиме long-polling (особенно когда бот подолгу отвечает стримом)
    иногда доносит одно и то же сообщение дважды — aiogram отдаёт его в хендлер
    повторно. Без дедупа дубль попадает в БД и в будущий контекст модели: она видит
    «одно и то же сообщение шесть раз подряд» и закономерно жалуется
    «хватит уже repeating одно и то же». Эту же проверку ставим перед записью/стримом
    в handle_text/_media_turn — дубль не пишем и не отвечаем.

    Окно 30с: больше — рискуем убить быструю осмысленную двойную отправку («да» «да»
    подряд реально бывает), меньше — не покроем запоздалый retry. 30с — баланс.
    """
    cur = conn.execute(
        "SELECT 1 FROM messages "
        "WHERE user_id = ? AND role = 'user' AND content = ? "
        "AND created_at >= ? LIMIT 1",
        (user_id, content, int(time.time()) - window_s),
    )
    return cur.fetchone() is not None


def add_message(
    conn: sqlite3.Connection,
    user_id: int,
    role: str,
    kind: str,
    content: str,
) -> None:
    conn.execute(
        "INSERT INTO messages (user_id, role, kind, content, created_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (user_id, role, kind, content, int(time.time())),
    )
    conn.commit()


def get_recent_budget(
    conn: sqlite3.Connection,
    user_id: int,
    max_chars: int,
    hard_limit: int = 400,
    checkpoint_granularity: int = 2048,
) -> list[sqlite3.Row]:
    """Последние реплики диалога в хронологическом порядке, в бюджете по длине.

    Долговременная память фактами: если у юзера есть факты (memory_facts), они
    возвращаются ПЕРВОЙ строкой с role='memory' и content=компактный блок по
    категориям (_build_messages превращает её в system-блок), а дальше — только
    свежий хвост (сообщения после compacted_until) в бюджете max_chars. Промпт
    остаётся небольшим: старая переписка не гоняется на каждый запрос, только
    её факт-блок → короткий reasoning-поток и быстрые ответы.

    Кеш: подрезка хвоста идёт с гистерезисом в checkpoint_granularity символов.
    Без него граница обрезки ползла бы с каждой новой репликой — как только
    старая реплика выпадает спереди, префикс [system + история] меняется с
    самого начала и кеш промпта сбрасывается целиком. С гистерезисом следующая
    подрезка случается только через ~checkpoint_granularity символов новой
    переписки, а между ними префикс стабилен и попадает в кеш.
    """
    cur = conn.execute(
        "SELECT role, kind, content FROM messages "
        "WHERE user_id = ? AND id > COALESCE("
        "  (SELECT compacted_until FROM memories WHERE user_id = ?), 0) "
        "ORDER BY id DESC LIMIT ?",
        (user_id, user_id, hard_limit),
    )
    rows = list(cur.fetchall())
    rows.reverse()

    total = 0
    kept_start = 0
    for i in range(len(rows) - 1, -1, -1):
        content = rows[i]["content"]
        total += len(content) if content else 0
        if total > max_chars:
            kept_start = i + 1
            break

    if kept_start and 0 < checkpoint_granularity < max_chars:
        # Гистерезис: подрезаем не впритык к max_chars, а с запасом в checkpoint
        # символов. Впритык граница ползла бы с каждой новой репликой (каждая
        # подрезка — снова +1 реплика спереди) — префикс [system + история]
        # сбрасывался бы почти на каждом запросе. С запасом следующая подрезка
        # случится только через ~checkpoint символов новой переписки, а между
        # ними префикс стабилен и попадает в кеш. Плата — checkpoint символов
        # недоступной истории, что дешевле регулярной переобработки префикса.
        budget = max_chars - checkpoint_granularity
        total_tail = 0
        for i in range(len(rows) - 1, kept_start - 1, -1):
            total_tail += len(rows[i]["content"]) if rows[i]["content"] else 0
            if total_tail > budget:
                kept_start = i + 1
                break

    result: list = []
    facts = get_facts(conn, user_id)
    if facts:
        # Факты первой строкой — компактный блок по категориям (build_memory_block
        # в memory_compactor.py собирает текст; здесь — единый content). Синтетический
        # row с role='memory', _build_messages превратит его в system-блок.
        from memory_compactor import build_memory_block
        result.append({
            "role": "memory",
            "kind": "memory",
            "content": build_memory_block(facts),
        })
    result.extend(rows[kept_start:])
    return result


def get_persona(conn: sqlite3.Connection, user_id: int) -> Persona | None:
    cur = conn.execute(
        "SELECT name, summary FROM personas WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return Persona(name=row["name"], summary=row["summary"])


def set_persona(
    conn: sqlite3.Connection,
    user_id: int,
    name: str,
    summary: str,
    source_raw: str | None = None,
) -> None:
    conn.execute(
        "INSERT INTO personas (user_id, name, summary, source_raw, updated_at) "
        "VALUES (?, ?, ?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET "
        "  name = excluded.name, "
        "  summary = excluded.summary, "
        "  source_raw = excluded.source_raw, "
        "  updated_at = excluded.updated_at",
        (user_id, name, summary, source_raw, int(time.time())),
    )
    conn.commit()


def delete_persona(conn: sqlite3.Connection, user_id: int) -> None:
    conn.execute("DELETE FROM personas WHERE user_id = ?", (user_id,))
    conn.commit()


# --- Proactive messaging: помощь фоновому циклу «бот иногда пишет первым» ---

def last_activity_per_user(conn: sqlite3.Connection) -> dict[int, int]:
    """{user_id: created_at самой свежей реплики} для всех юзеров с сообщениями.
    GROUP BY user_id — full table scan, но для персонального бота (десятки-сотни
    юзеров) дёшево; джоинить с users не нужно — нас интересуют лишь те, у кого
    уже есть хоть какая-то переписка. Юзеры без сообщений сюда не попадают."""
    cur = conn.execute(
        "SELECT user_id, MAX(created_at) AS last_ts FROM messages GROUP BY user_id"
    )
    return {int(r["user_id"]): int(r["last_ts"]) for r in cur.fetchall()}


def last_row(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    """Самая свежая реплика юзера (по id DESC). Ключи: id, role, kind, content, created_at.
    Нужна чтобы понять, был ли последний ход от юзера (motive 'silence') или от
    бота (motive 'continuation')."""
    cur = conn.execute(
        "SELECT id, role, kind, content, created_at FROM messages "
        "WHERE user_id = ? ORDER BY id DESC LIMIT 1",
        (user_id,),
    )
    return cur.fetchone()


def proactive_count_24h(conn: sqlite3.Connection, user_id: int) -> int:
    """Сколько proactive-реплик бот УЖЕ прислал юзеру за rolling 24 часа.
    created_at хранится в epoch-секундах UTC — часовой пояс не нужен."""
    cur = conn.execute(
        "SELECT COUNT(*) AS n FROM messages "
        "WHERE user_id = ? AND role = 'assistant' AND kind = 'proactive' "
        "AND created_at >= ?",
        (user_id, int(time.time()) - 86400),
    )
    return int(cur.fetchone()["n"])


def last_proactive_ts(conn: sqlite3.Connection, user_id: int) -> int | None:
    """created_at последней proactive-реплики бота этому юзеру, или None если их
    ещё не было. Для per-user троттла proactive_min_gap_h между пинками."""
    cur = conn.execute(
        "SELECT created_at FROM messages "
        "WHERE user_id = ? AND role = 'assistant' AND kind = 'proactive' "
        "ORDER BY id DESC LIMIT 1",
        (user_id,),
    )
    row = cur.fetchone()
    return int(row["created_at"]) if row else None


# --- Автокомпакт памяти: сжатие старой истории в краткое изложение ---

def get_memory(conn: sqlite3.Connection, user_id: int) -> sqlite3.Row | None:
    """Память юзера (content + compacted_until) или None, если компактации не было."""
    cur = conn.execute(
        "SELECT content, compacted_until FROM memories WHERE user_id = ?",
        (user_id,),
    )
    return cur.fetchone()


def set_memory(
    conn: sqlite3.Connection,
    user_id: int,
    content: str,
    compacted_until: int,
) -> None:
    """Записать/обновить память: краткое изложение важного + id последнего
    сообщения, которое вошло в неё. ON CONFLICT — памяти одна на юзера."""
    conn.execute(
        "INSERT INTO memories (user_id, content, compacted_until, updated_at) "
        "VALUES (?, ?, ?, ?) "
        "ON CONFLICT(user_id) DO UPDATE SET "
        "  content = excluded.content, "
        "  compacted_until = excluded.compacted_until, "
        "  updated_at = excluded.updated_at",
        (user_id, content, compacted_until, int(time.time())),
    )
    conn.commit()


def messages_to_compact(
    conn: sqlite3.Connection,
    user_id: int,
    from_id_exclusive: int | None,
    until_id_inclusive: int,
) -> list[sqlite3.Row]:
    """Сообщения в диапазоне (from_id_exclusive, until_id_inclusive] в
    хронологическом порядке — то, что пойдёт в компактацию."""
    cur = conn.execute(
        "SELECT id, role, kind, content FROM messages "
        "WHERE user_id = ? AND id > COALESCE(?, 0) AND id <= ? "
        "ORDER BY id ASC",
        (user_id, from_id_exclusive, until_id_inclusive),
    )
    return list(cur.fetchall())


# --- Долговременная память фактами (замена плоской memories-сводки) ---

def get_facts(conn: sqlite3.Connection, user_id: int) -> list[sqlite3.Row]:
    """Все факты юзера по категориям (id, category, content, updated_at)."""
    cur = conn.execute(
        "SELECT id, category, content, updated_at FROM memory_facts "
        "WHERE user_id = ? ORDER BY category ASC, updated_at ASC",
        (user_id,),
    )
    return list(cur.fetchall())


def add_fact(
    conn: sqlite3.Connection,
    user_id: int,
    category: str,
    content: str,
    source_msg_id: int | None = None,
) -> int:
    """Добавить факт (одну строку). Возвращает id нового факта."""
    now = int(time.time())
    cur = conn.execute(
        "INSERT INTO memory_facts (user_id, category, content, source_msg_id, created_at, updated_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, category, content, source_msg_id, now, now),
    )
    conn.commit()
    return int(cur.lastrowid)


def update_fact(conn: sqlite3.Connection, fact_id: int, content: str) -> None:
    """Заострить/уточнить существующий факт (id обязателен — кураторка диффует)."""
    conn.execute(
        "UPDATE memory_facts SET content = ?, updated_at = ? WHERE id = ?",
        (content, int(time.time()), fact_id),
    )
    conn.commit()


def drop_facts(conn: sqlite3.Connection, fact_ids: list[int]) -> None:
    """Удалить устаревшие/неверные факты по списку id."""
    if not fact_ids:
        return
    placeholders = ",".join("?" for _ in fact_ids)
    conn.execute(
        f"DELETE FROM memory_facts WHERE id IN ({placeholders})",
        tuple(fact_ids),
    )
    conn.commit()
