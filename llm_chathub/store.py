"""SQLite persistence: conversations, messages, RAG documents and chunks.

A fresh connection is opened per operation. That keeps the store usable from the
ComfyUI execution threads and from the aiohttp routes at the same time without
sharing a connection object. WAL mode keeps concurrent reads cheap.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
import uuid

from . import config

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id            TEXT PRIMARY KEY,
    title         TEXT NOT NULL DEFAULT 'Nouveau chat',
    created_at    REAL NOT NULL,
    updated_at    REAL NOT NULL,
    provider      TEXT NOT NULL DEFAULT '',
    model         TEXT NOT NULL DEFAULT '',
    system_prompt TEXT NOT NULL DEFAULT '',
    reference     TEXT NOT NULL DEFAULT '',
    rag_enabled   INTEGER NOT NULL DEFAULT 0,
    rag_folder    TEXT NOT NULL DEFAULT '',
    rag_top_k     INTEGER NOT NULL DEFAULT 4,
    history_limit INTEGER NOT NULL DEFAULT 20,
    temperature   REAL NOT NULL DEFAULT 0.7,
    max_tokens    INTEGER NOT NULL DEFAULT 1024
);

CREATE TABLE IF NOT EXISTS messages (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    seq             INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL DEFAULT '',
    images          TEXT NOT NULL DEFAULT '[]',
    created_at      REAL NOT NULL,
    edited          INTEGER NOT NULL DEFAULT 0,
    meta            TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages (conversation_id, seq);

CREATE TABLE IF NOT EXISTS documents (
    id         TEXT PRIMARY KEY,
    path       TEXT NOT NULL,
    folder     TEXT NOT NULL DEFAULT '',
    size       INTEGER NOT NULL DEFAULT 0,
    mtime      REAL NOT NULL DEFAULT 0,
    added_at   REAL NOT NULL,
    provider   TEXT NOT NULL DEFAULT '',
    model      TEXT NOT NULL DEFAULT '',
    chunks     INTEGER NOT NULL DEFAULT 0,
    embedded   INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS chunks (
    id          TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    text        TEXT NOT NULL,
    tokens      TEXT NOT NULL DEFAULT '{}',
    embedding   BLOB,
    dim         INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks (document_id);

CREATE TABLE IF NOT EXISTS kv (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_initialized = False


def _now() -> float:
    return time.time()


def new_id(prefix: str = "") -> str:
    return (prefix + uuid.uuid4().hex[:12]) if prefix else uuid.uuid4().hex


@contextlib.contextmanager
def connect():
    config.ensure_dirs()
    connection = sqlite3.connect(config.db_path(), timeout=30)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        yield connection
        connection.commit()
    finally:
        connection.close()


def init_db() -> None:
    global _initialized
    if _initialized:
        return
    with connect() as connection:
        connection.executescript(SCHEMA)
    _initialized = True


def _row_to_conversation(row: sqlite3.Row) -> dict:
    data = dict(row)
    data["rag_enabled"] = bool(data.get("rag_enabled"))
    return data


def _row_to_message(row: sqlite3.Row) -> dict:
    data = dict(row)
    try:
        data["images"] = json.loads(data.get("images") or "[]")
    except Exception:
        data["images"] = []
    try:
        data["meta"] = json.loads(data.get("meta") or "{}")
    except Exception:
        data["meta"] = {}
    data["edited"] = bool(data.get("edited"))
    return data


# --------------------------------------------------------------------------- conversations


def list_conversations() -> list[dict]:
    init_db()
    with connect() as connection:
        rows = connection.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) AS message_count "
            "FROM conversations c ORDER BY c.updated_at DESC"
        ).fetchall()
    return [_row_to_conversation(row) for row in rows]


def get_conversation(conversation_id: str) -> dict | None:
    init_db()
    with connect() as connection:
        row = connection.execute(
            "SELECT c.*, (SELECT COUNT(*) FROM messages m WHERE m.conversation_id = c.id) "
            "AS message_count FROM conversations c WHERE c.id = ?",
            (conversation_id,),
        ).fetchone()
    return _row_to_conversation(row) if row else None


def create_conversation(**fields) -> dict:
    init_db()
    now = _now()
    defaults = config.resolve_provider(fields.get("provider"))
    conversation_id = fields.get("id") or new_id("conv_")
    record = {
        "id": conversation_id,
        "title": (fields.get("title") or "").strip() or "Nouveau chat",
        "created_at": now,
        "updated_at": now,
        "provider": fields.get("provider") or defaults["provider"],
        "model": fields.get("model") or defaults["model"],
        "system_prompt": fields.get("system_prompt") or defaults["system_prompt"],
        "reference": fields.get("reference") or "",
        "rag_enabled": 1 if fields.get("rag_enabled", defaults["rag_enabled"]) else 0,
        "rag_folder": fields.get("rag_folder") or defaults["rag_folder"],
        "rag_top_k": int(fields.get("rag_top_k", defaults["rag_top_k"])),
        "history_limit": int(fields.get("history_limit", defaults["history_limit"])),
        "temperature": float(fields.get("temperature", defaults["temperature"])),
        "max_tokens": int(fields.get("max_tokens", defaults["max_tokens"])),
    }
    with connect() as connection:
        connection.execute(
            "INSERT INTO conversations (id, title, created_at, updated_at, provider, model, "
            "system_prompt, reference, rag_enabled, rag_folder, rag_top_k, history_limit, "
            "temperature, max_tokens) VALUES (:id, :title, :created_at, :updated_at, :provider, "
            ":model, :system_prompt, :reference, :rag_enabled, :rag_folder, :rag_top_k, "
            ":history_limit, :temperature, :max_tokens)",
            record,
        )
    return get_conversation(conversation_id)


CONVERSATION_FIELDS = {
    "title",
    "provider",
    "model",
    "system_prompt",
    "reference",
    "rag_enabled",
    "rag_folder",
    "rag_top_k",
    "history_limit",
    "temperature",
    "max_tokens",
}


def update_conversation(conversation_id: str, **fields) -> dict | None:
    init_db()
    updates = {key: value for key, value in fields.items() if key in CONVERSATION_FIELDS}
    if "rag_enabled" in updates:
        updates["rag_enabled"] = 1 if updates["rag_enabled"] else 0
    updates["updated_at"] = _now()
    assignments = ", ".join(f"{key} = :{key}" for key in updates)
    updates["id"] = conversation_id
    with connect() as connection:
        connection.execute(
            f"UPDATE conversations SET {assignments} WHERE id = :id", updates
        )
    return get_conversation(conversation_id)


def touch_conversation(conversation_id: str) -> None:
    with connect() as connection:
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id)
        )


def delete_conversation(conversation_id: str) -> None:
    init_db()
    with connect() as connection:
        connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        connection.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))


# --------------------------------------------------------------------------- messages


def list_messages(conversation_id: str) -> list[dict]:
    init_db()
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM messages WHERE conversation_id = ? ORDER BY seq ASC",
            (conversation_id,),
        ).fetchall()
    return [_row_to_message(row) for row in rows]


def get_message(message_id: str) -> dict | None:
    init_db()
    with connect() as connection:
        row = connection.execute("SELECT * FROM messages WHERE id = ?", (message_id,)).fetchone()
    return _row_to_message(row) if row else None


def add_message(
    conversation_id: str,
    role: str,
    content: str,
    images: list[str] | None = None,
    meta: dict | None = None,
) -> dict:
    init_db()
    with connect() as connection:
        row = connection.execute(
            "SELECT COALESCE(MAX(seq), -1) AS last FROM messages WHERE conversation_id = ?",
            (conversation_id,),
        ).fetchone()
        seq = int(row["last"]) + 1
        message_id = new_id("msg_")
        connection.execute(
            "INSERT INTO messages (id, conversation_id, seq, role, content, images, created_at, "
            "edited, meta) VALUES (?, ?, ?, ?, ?, ?, ?, 0, ?)",
            (
                message_id,
                conversation_id,
                seq,
                role,
                content or "",
                json.dumps(images or []),
                _now(),
                json.dumps(meta or {}),
            ),
        )
        connection.execute(
            "UPDATE conversations SET updated_at = ? WHERE id = ?", (_now(), conversation_id)
        )
    return get_message(message_id)


def update_message(message_id: str, content: str, images: list[str] | None = None) -> dict | None:
    init_db()
    with connect() as connection:
        if images is None:
            connection.execute(
                "UPDATE messages SET content = ?, edited = 1 WHERE id = ?", (content, message_id)
            )
        else:
            connection.execute(
                "UPDATE messages SET content = ?, images = ?, edited = 1 WHERE id = ?",
                (content, json.dumps(images), message_id),
            )
    return get_message(message_id)


def delete_message(message_id: str) -> None:
    init_db()
    with connect() as connection:
        connection.execute("DELETE FROM messages WHERE id = ?", (message_id,))


def delete_from_message(conversation_id: str, message_id: str, include: bool = True) -> int:
    """Drop a message and everything after it (used by edit / regenerate)."""
    init_db()
    with connect() as connection:
        row = connection.execute(
            "SELECT seq FROM messages WHERE id = ? AND conversation_id = ?",
            (message_id, conversation_id),
        ).fetchone()
        if not row:
            return 0
        operator = ">=" if include else ">"
        cursor = connection.execute(
            f"DELETE FROM messages WHERE conversation_id = ? AND seq {operator} ?",
            (conversation_id, int(row["seq"])),
        )
        return cursor.rowcount or 0


def truncate_messages(conversation_id: str, keep: int) -> None:
    """Keep only the first ``keep`` messages of the conversation."""
    init_db()
    if keep <= 0:
        with connect() as connection:
            connection.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        return
    with connect() as connection:
        connection.execute(
            "DELETE FROM messages WHERE conversation_id = ? AND seq >= ?",
            (conversation_id, keep),
        )


# --------------------------------------------------------------------------- documents / chunks


def upsert_document(**fields) -> dict:
    init_db()
    document_id = fields.get("id") or new_id("doc_")
    with connect() as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        connection.execute(
            "INSERT INTO documents (id, path, folder, size, mtime, added_at, provider, model, "
            "chunks, embedded) VALUES (:id, :path, :folder, :size, :mtime, :added_at, :provider, "
            ":model, :chunks, :embedded)",
            {
                "id": document_id,
                "path": fields.get("path", ""),
                "folder": fields.get("folder", ""),
                "size": int(fields.get("size", 0)),
                "mtime": float(fields.get("mtime", 0)),
                "added_at": _now(),
                "provider": fields.get("provider", ""),
                "model": fields.get("model", ""),
                "chunks": int(fields.get("chunks", 0)),
                "embedded": 1 if fields.get("embedded") else 0,
            },
        )
    return {"id": document_id, **fields}


def add_chunks(document_id: str, chunks: list[dict]) -> None:
    if not chunks:
        return
    init_db()
    rows = [
        (
            new_id("chk_"),
            document_id,
            index,
            chunk["text"],
            json.dumps(chunk.get("tokens") or {}),
            chunk.get("embedding"),
            int(chunk.get("dim") or 0),
        )
        for index, chunk in enumerate(chunks)
    ]
    with connect() as connection:
        connection.executemany(
            "INSERT INTO chunks (id, document_id, seq, text, tokens, embedding, dim) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            rows,
        )


def list_documents() -> list[dict]:
    init_db()
    with connect() as connection:
        rows = connection.execute(
            "SELECT * FROM documents ORDER BY added_at DESC"
        ).fetchall()
    return [dict(row) for row in rows]


def delete_document(document_id: str) -> None:
    init_db()
    with connect() as connection:
        connection.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        connection.execute("DELETE FROM documents WHERE id = ?", (document_id,))


def clear_index() -> None:
    init_db()
    with connect() as connection:
        connection.execute("DELETE FROM chunks")
        connection.execute("DELETE FROM documents")


def index_stats() -> dict:
    init_db()
    with connect() as connection:
        documents = connection.execute("SELECT COUNT(*) AS n FROM documents").fetchone()["n"]
        chunks = connection.execute("SELECT COUNT(*) AS n FROM chunks").fetchone()["n"]
        embedded = connection.execute(
            "SELECT COUNT(*) AS n FROM chunks WHERE embedding IS NOT NULL"
        ).fetchone()["n"]
        folder = connection.execute(
            "SELECT folder FROM documents ORDER BY added_at DESC LIMIT 1"
        ).fetchone()
    return {
        "documents": documents,
        "chunks": chunks,
        "embedded": embedded,
        "folder": folder["folder"] if folder else "",
    }


def iter_chunk_rows(with_tokens: bool = True):
    """Stream chunk rows for retrieval without loading embeddings as objects."""
    init_db()
    columns = "id, document_id, text, embedding, dim" + (", tokens" if with_tokens else "")
    with connect() as connection:
        cursor = connection.execute(
            f"SELECT {columns} FROM chunks ORDER BY document_id, seq"
        )
        for row in cursor:
            yield dict(row)


def document_paths() -> dict[str, str]:
    init_db()
    with connect() as connection:
        rows = connection.execute("SELECT id, path FROM documents").fetchall()
    return {row["id"]: row["path"] for row in rows}


# --------------------------------------------------------------------------- kv


def kv_get(key: str, default=None):
    init_db()
    with connect() as connection:
        row = connection.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
    if not row:
        return default
    try:
        return json.loads(row["value"])
    except Exception:
        return default


def kv_set(key: str, value) -> None:
    init_db()
    with connect() as connection:
        connection.execute(
            "INSERT INTO kv (key, value) VALUES (?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (key, json.dumps(value)),
        )
