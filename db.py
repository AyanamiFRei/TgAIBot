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

-- Трёхуровневая гибридная память: промежуточный слой «дайджест текущей сессии».
-- Вместо жёсткого обрезания контекста по N сообщений: сообщения, вытесненные из
-- дословного окна, прогрессивно сжимаются в rolling-дайджест (тема + суть +
-- открытые вопросы). При длинной паузе (>SESSION_TIMEOUT_H) дайджест переезжает
-- в prev_digest, и новая сессия начинается с чистого буфера (но prev_digest
-- остаётся доступен — юзер может ответить на старую тему после перерыва).
CREATE TABLE IF NOT EXISTS session_state (
    user_id         INTEGER PRIMARY KEY,
    digest          TEXT    NOT NULL DEFAULT '',  -- rolling-дайджест текущей сессии
    topic           TEXT    NOT NULL DEFAULT '',  -- текущая тема (1-2 предложения)
    unresolved      TEXT    NOT NULL DEFAULT '',  -- открытые вопросы / задачи
    digest_until    INTEGER NOT NULL DEFAULT 0,   -- id последнего сообщения, покрытого дайджестом
    session_start   INTEGER NOT NULL DEFAULT 0,   -- id первого сообщения текущей сессии
    prev_digest     TEXT    NOT NULL DEFAULT '',  -- дайджест предыдущей сессии (для мягкого перехода)
    updated_at      INTEGER NOT NULL DEFAULT 0,
    FOREIGN KEY (user_id) REFERENCES users (user_id)
);
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

    Трёхуровневая гибридная память:
      1. Долговременные факты (memory_facts) → role='memory'
      2. Дайджест предыдущей сессии (session_state.prev_digest) → role='session_prev'
      3. Дайджест текущей сессии (session_state.digest/topic/unresolved) → role='session_current'
      4. Дословный хвост (свежие сообщения после digest_until или compacted_until)

    Парная защита: контекст никогда не обрезается на ответе ассистента без
    предшествующего вопроса юзера — если первая реплика в окне 'assistant',
    она тоже отбрасывается.

    hard_limit — потолок строк из SQL (защита от OOM на огромных историях).
    Размер окна контролируется max_chars (символьный бюджет), а НЕ числом сообщений.
    """
    # Нижняя граница: сообщения ДО этого id уже покрыты долговременной памятью.
    compacted_until_row = conn.execute(
        "SELECT compacted_until FROM memories WHERE user_id = ?",
        (user_id,),
    ).fetchone()
    compacted_until = compacted_until_row["compacted_until"] if compacted_until_row else 0

    # Сессионная граница: сообщения до digest_until покрыты дайджестом сессии.
    session = get_session_state(conn, user_id)
    digest_until = session["digest_until"] if session else 0

    # Начинаем выборку с наибольшей из двух границ — не дублируем информацию.
    lower_bound = max(compacted_until, digest_until)

    cur = conn.execute(
        "SELECT role, kind, content FROM messages "
        "WHERE user_id = ? AND id > ? "
        "ORDER BY id DESC LIMIT ?",
        (user_id, lower_bound, hard_limit),
    )
    rows = list(cur.fetchall())

    # Гарантия минимального дословного хвоста: если lower_bound отсёк слишком много
    # (например, сразу после финализации сессии или компактации), модель всё равно
    # должна видеть последние 6 реплик — иначе она не поймёт, на что именно юзер
    # отвечает после паузы («Да, во сколько?» на вопрос о встрече).
    if len(rows) < 6:
        cur_fallback = conn.execute(
            "SELECT role, kind, content FROM messages "
            "WHERE user_id = ? "
            "ORDER BY id DESC LIMIT 6",
            (user_id,),
        )
        fallback_rows = list(cur_fallback.fetchall())
        if len(fallback_rows) > len(rows):
            rows = fallback_rows

    rows.reverse()

    # --- Символьный бюджет ---
    total = 0
    kept_start = 0
    for i in range(len(rows) - 1, -1, -1):
        content = rows[i]["content"]
        total += len(content) if content else 0
        if total > max_chars:
            kept_start = i + 1
            break

    # Гистерезис: подрезаем не впритык к max_chars, а с запасом в checkpoint
    # символов, чтобы стабилизировать префикс для KV-кеша промпта.
    if kept_start and 0 < checkpoint_granularity < max_chars:
        budget = max_chars - checkpoint_granularity
        total_tail = 0
        for i in range(len(rows) - 1, kept_start - 1, -1):
            total_tail += len(rows[i]["content"]) if rows[i]["content"] else 0
            if total_tail > budget:
                kept_start = i + 1
                break

    trimmed = rows[kept_start:]

    # --- Парная защита: не начинаем с оторванного assistant-ответа ---
    if trimmed and trimmed[0]["role"] == "assistant":
        trimmed = trimmed[1:]

    # --- Собираем результат с инжекцией всех уровней памяти ---
    result: list = []

    # Уровень 1: долговременные факты.
    facts = get_facts(conn, user_id)
    if facts:
        from memory_compactor import build_memory_block
        block = build_memory_block(facts)
        if block:
            result.append({
                "role": "memory",
                "kind": "memory",
                "content": block,
            })

    # Уровень 2: дайджест предыдущей сессии (мягкий переход после паузы).
    if session and session["prev_digest"]:
        result.append({
            "role": "session_prev",
            "kind": "session",
            "content": session["prev_digest"],
        })

    # Уровень 3: дайджест текущей сессии (тема + суть + открытые вопросы).
    if session:
        parts: list[str] = []
        if session["topic"]:
            parts.append(f"Тема: {session['topic']}")
        if session["digest"]:
            parts.append(session["digest"])
        if session["unresolved"]:
            parts.append(f"Открытые вопросы: {session['unresolved']}")
        session_block = "\n".join(parts)
        if session_block.strip():
            result.append({
                "role": "session_current",
                "kind": "session",
                "content": session_block,
            })

    # Дословный хвост.
    result.extend(trimmed)
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


# --- Дайджест сессии (промежуточный уровень трёхуровневой памяти) ---

def get_session_state(conn: sqlite3.Connection, user_id: int) -> dict | None:
    """Текущее состояние сессии юзера (digest, topic, unresolved, prev_digest…).
    None — сессии ещё не было (новый юзер / до миграции)."""
    cur = conn.execute(
        "SELECT digest, topic, unresolved, digest_until, session_start, "
        "prev_digest, updated_at FROM session_state WHERE user_id = ?",
        (user_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "digest": row["digest"],
        "topic": row["topic"],
        "unresolved": row["unresolved"],
        "digest_until": row["digest_until"],
        "session_start": row["session_start"],
        "prev_digest": row["prev_digest"],
        "updated_at": row["updated_at"],
    }


def upsert_session_state(
    conn: sqlite3.Connection,
    user_id: int,
    *,
    digest: str,
    topic: str,
    unresolved: str,
    digest_until: int,
    session_start: int,
    prev_digest: str | None = None,
) -> None:
    """Обновить или создать запись сессии.  prev_digest=None — не трогать."""
    now = int(time.time())
    existing = get_session_state(conn, user_id)
    if existing is None:
        conn.execute(
            "INSERT INTO session_state "
            "(user_id, digest, topic, unresolved, digest_until, session_start, prev_digest, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, digest, topic, unresolved, digest_until, session_start,
             prev_digest or "", now),
        )
    else:
        if prev_digest is None:
            conn.execute(
                "UPDATE session_state SET digest=?, topic=?, unresolved=?, "
                "digest_until=?, session_start=?, updated_at=? WHERE user_id=?",
                (digest, topic, unresolved, digest_until, session_start, now, user_id),
            )
        else:
            conn.execute(
                "UPDATE session_state SET digest=?, topic=?, unresolved=?, "
                "digest_until=?, session_start=?, prev_digest=?, updated_at=? "
                "WHERE user_id=?",
                (digest, topic, unresolved, digest_until, session_start,
                 prev_digest, now, user_id),
            )
    conn.commit()


def finalize_session(conn: sqlite3.Connection, user_id: int) -> None:
    """Завершить сессию: дайджест + тему + нерешённое переносим в prev_digest,
    текущие поля обнуляем. Вызывается при обнаружении длинной паузы."""
    state = get_session_state(conn, user_id)
    if state is None or (not state["digest"] and not state["topic"]):
        return
    # Собираем prev_digest из текущей сессии.
    parts: list[str] = []
    if state["topic"]:
        parts.append(f"Тема: {state['topic']}")
    if state["digest"]:
        parts.append(state["digest"])
    if state["unresolved"]:
        parts.append(f"Нерешённое: {state['unresolved']}")
    prev = "; ".join(parts) if parts else ""
    now = int(time.time())
    conn.execute(
        "UPDATE session_state SET digest='', topic='', unresolved='', "
        "digest_until=0, session_start=0, prev_digest=?, updated_at=? "
        "WHERE user_id=?",
        (prev, now, user_id),
    )
    conn.commit()


def last_user_msg_ts(conn: sqlite3.Connection, user_id: int) -> int | None:
    """created_at последнего сообщения юзера (роль 'user'), или None."""
    cur = conn.execute(
        "SELECT created_at FROM messages WHERE user_id = ? AND role = 'user' "
        "ORDER BY id DESC LIMIT 1",
        (user_id,),
    )
    row = cur.fetchone()
    return int(row["created_at"]) if row else None


def last_any_msg_ts(conn: sqlite3.Connection, user_id: int) -> int | None:
    """created_at самого свежего сообщения юзера (любая роль), или None."""
    cur = conn.execute(
        "SELECT created_at FROM messages WHERE user_id = ? "
        "ORDER BY id DESC LIMIT 1",
        (user_id,),
    )
    row = cur.fetchone()
    return int(row["created_at"]) if row else None
