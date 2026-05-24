"""
SQLite-backed cognitive memory store.

Schema:
  - memories: core entry storage (all MemoryEntry fields)
  - memory_tags: normalized tag associations (many-to-many)
  - memory_fts: FTS5 virtual table on title + content for full-text search
  - memory_relations: typed relationships between memories (future phase)

Design decisions:
  - WAL mode for concurrent reads
  - Foreign keys enforced
  - Schema versioning for future migrations
  - No decay columns. No importance column. Those don't belong here.
"""

import sqlite3
import json
import time
from typing import Optional

from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance

SCHEMA_VERSION = 1


def _memory_type_to_str(mt: MemoryType) -> str:
    return mt.value


def _str_to_memory_type(s: str) -> MemoryType:
    return MemoryType(s)


def _provenance_to_str(p: Provenance) -> str:
    return p.value


def _str_to_provenance(s: str) -> Provenance:
    return Provenance(s)


def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
    """Convert a database row to a MemoryEntry."""
    return MemoryEntry(
        type=_str_to_memory_type(row["type"]),
        title=row["title"],
        content=row["content"],
        id=row["id"],
        tags=json.loads(row["tags"]) if row["tags"] else [],
        confidence=row["confidence"],
        provenance=_str_to_provenance(row["provenance"]),
        contradicted_by=row["contradicted_by"],
        distilled_to=row["distilled_to"],
        created_at=row["created_at"],
        updated_at=row["updated_at"],
        access_count=row["access_count"],
        last_accessed=row["last_accessed"],
        source_session_id=row["source_session_id"],
    )


class CognitiveMemoryStore:
    """SQLite-backed store for typed cognitive memories.

    Usage:
        store = CognitiveMemoryStore(":memory:")           # in-memory
        store = CognitiveMemoryStore("~/.hermes/cognitive_memory.db")  # persistent
    """

    def __init__(self, db_path: str = ":memory:"):
        self._db_path = db_path
        self._conn = sqlite3.connect(db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self) -> None:
        """Create tables if they don't exist. Idempotent."""
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                type TEXT NOT NULL,
                title TEXT NOT NULL,
                content TEXT NOT NULL,
                tags TEXT NOT NULL DEFAULT '[]',
                confidence REAL NOT NULL DEFAULT 1.0,
                provenance TEXT NOT NULL DEFAULT 'user_explicit',
                contradicted_by TEXT,
                distilled_to TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                access_count INTEGER NOT NULL DEFAULT 0,
                last_accessed REAL,
                source_session_id TEXT
            );

            CREATE TABLE IF NOT EXISTS memory_tags (
                memory_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (memory_id, tag),
                FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
            );

            CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                title, content, content='memories', content_rowid='rowid'
            );

            -- Triggers to keep FTS in sync
            CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
                INSERT INTO memory_fts(rowid, title, content)
                VALUES (new.rowid, new.title, new.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, title, content)
                VALUES ('delete', old.rowid, old.title, old.content);
            END;

            CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
                INSERT INTO memory_fts(memory_fts, rowid, title, content)
                VALUES ('delete', old.rowid, old.title, old.content);
                INSERT INTO memory_fts(rowid, title, content)
                VALUES (new.rowid, new.title, new.content);
            END;

            -- Schema version tracking
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
        """)

        # Set schema version if not already set
        self._conn.execute(
            "INSERT OR IGNORE INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )

    def close(self) -> None:
        """Close the database connection."""
        self._conn.close()

    # ── CRUD ────────────────────────────────────────────────────────────────

    def insert(self, entry: MemoryEntry) -> MemoryEntry:
        """Insert a new memory entry. Returns the entry with generated ID if needed.

        If entry.id is empty, a new ULID is generated.
        """
        from agent.cognitive_memory.types import generate_ulid

        if not entry.id:
            entry.id = generate_ulid()

        now = time.time()
        entry.created_at = now
        entry.updated_at = now

        self._conn.execute(
            """
            INSERT INTO memories (
                id, type, title, content, tags, confidence, provenance,
                contradicted_by, distilled_to, created_at, updated_at,
                access_count, last_accessed, source_session_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                entry.id,
                _memory_type_to_str(entry.type),
                entry.title,
                entry.content,
                json.dumps(entry.tags),
                entry.confidence,
                _provenance_to_str(entry.provenance),
                entry.contradicted_by,
                entry.distilled_to,
                entry.created_at,
                entry.updated_at,
                entry.access_count,
                entry.last_accessed,
                entry.source_session_id,
            ),
        )
        self._conn.commit()

        # Store tag associations
        if entry.tags:
            self._conn.executemany(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                [(entry.id, tag) for tag in entry.tags],
            )
            self._conn.commit()

        return entry

    def get_by_id(self, entry_id: str) -> Optional[MemoryEntry]:
        """Get a memory by ID. Returns None if not found."""
        row = self._conn.execute(
            "SELECT * FROM memories WHERE id = ?", (entry_id,)
        ).fetchone()
        if row is None:
            return None
        return _row_to_entry(row)

    def update(self, entry: MemoryEntry) -> None:
        """Update an existing memory entry. Must have a valid ID."""
        entry.updated_at = time.time()

        self._conn.execute(
            """
            UPDATE memories SET
                type = ?, title = ?, content = ?, tags = ?, confidence = ?,
                provenance = ?, contradicted_by = ?, distilled_to = ?,
                updated_at = ?, access_count = ?, last_accessed = ?,
                source_session_id = ?
            WHERE id = ?
            """,
            (
                _memory_type_to_str(entry.type),
                entry.title,
                entry.content,
                json.dumps(entry.tags),
                entry.confidence,
                _provenance_to_str(entry.provenance),
                entry.contradicted_by,
                entry.distilled_to,
                entry.updated_at,
                entry.access_count,
                entry.last_accessed,
                entry.source_session_id,
                entry.id,
            ),
        )
        self._conn.commit()

        # Rebuild tag associations: delete old, insert new
        self._conn.execute("DELETE FROM memory_tags WHERE memory_id = ?", (entry.id,))
        if entry.tags:
            self._conn.executemany(
                "INSERT OR IGNORE INTO memory_tags (memory_id, tag) VALUES (?, ?)",
                [(entry.id, tag) for tag in entry.tags],
            )
        self._conn.commit()

    def delete(self, entry_id: str) -> bool:
        """Delete a memory by ID. Returns True if deleted, False if not found."""
        # Tag cleanup handled by ON DELETE CASCADE foreign key
        cursor = self._conn.execute("DELETE FROM memories WHERE id = ?", (entry_id,))
        self._conn.commit()
        return cursor.rowcount > 0

    # ── Listing ─────────────────────────────────────────────────────────────

    def list_all(self, limit: int = 50, offset: int = 0) -> list[MemoryEntry]:
        """List all memories, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM memories ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    def list_by_type(
        self, mem_type: MemoryType, limit: int = 50, offset: int = 0
    ) -> list[MemoryEntry]:
        """List memories filtered by type, newest first."""
        rows = self._conn.execute(
            "SELECT * FROM memories WHERE type = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (_memory_type_to_str(mem_type), limit, offset),
        ).fetchall()
        return [_row_to_entry(r) for r in rows]

    # ── FTS Search ──────────────────────────────────────────────────────────

    def search_fts(
        self,
        query: str,
        type_filter: Optional[MemoryType] = None,
        limit: int = 20,
    ) -> list[MemoryEntry]:
        """Full-text search on title and content using FTS5.

        Returns entries ranked by BM25 relevance.
        """
        # Sanitize: wrap terms in double quotes for phrase matching,
        # but allow partial matching by appending *
        terms = query.strip().split()
        if not terms:
            return []

        # Build FTS5 query with prefix matching on last term
        fts_query_parts = []
        for i, term in enumerate(terms[:-1]):
            fts_query_parts.append(f'"{term}"')
        if terms:
            last = terms[-1]
            fts_query_parts.append(f'"{last}"*')

        fts_query = " ".join(fts_query_parts)

        if type_filter:
            rows = self._conn.execute(
                """
                SELECT m.* FROM memories m
                JOIN memory_fts fts ON m.rowid = fts.rowid
                WHERE memory_fts MATCH ? AND m.type = ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, _memory_type_to_str(type_filter), limit),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT m.* FROM memories m
                JOIN memory_fts fts ON m.rowid = fts.rowid
                WHERE memory_fts MATCH ?
                ORDER BY rank
                LIMIT ?
                """,
                (fts_query, limit),
            ).fetchall()

        return [_row_to_entry(r) for r in rows]

    # ── Stats ───────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Return health statistics about the memory store."""
        total = self._conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]

        type_counts = {}
        for mt in MemoryType:
            count = self._conn.execute(
                "SELECT COUNT(*) FROM memories WHERE type = ?",
                (_memory_type_to_str(mt),),
            ).fetchone()[0]
            type_counts[mt.value] = count

        schema_ver = self._conn.execute(
            "SELECT value FROM schema_meta WHERE key = 'schema_version'"
        ).fetchone()
        schema_version = int(schema_ver[0]) if schema_ver else 0

        return {
            "memory_count": total,
            "type_counts": type_counts,
            "schema_version": schema_version,
        }
