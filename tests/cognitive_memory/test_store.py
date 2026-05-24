"""
Tests for agent/cognitive_memory/store.py — SQLite-backed CognitiveMemoryStore.

TDD RED phase: define behavior before implementation exists.
Uses temporary databases; never touches production data.
"""

import os
import tempfile
import pytest
from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore


@pytest.fixture
def store():
    """Create a fresh in-memory store for each test."""
    db = CognitiveMemoryStore(":memory:")
    yield db
    db.close()


@pytest.fixture
def store_on_disk():
    """Create a temporary file-backed store."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    db = CognitiveMemoryStore(path)
    yield db
    db.close()
    try:
        os.unlink(path)
    except OSError:
        pass


class TestStoreCreation:
    """Database initialization and schema setup."""

    def test_creates_in_memory_database(self, store):
        """In-memory store initializes without error."""
        assert store is not None
        stats = store.get_stats()
        assert stats["memory_count"] == 0

    def test_creates_file_database(self, store_on_disk):
        """File-backed store creates the database file."""
        assert store_on_disk is not None
        stats = store_on_disk.get_stats()
        assert stats["memory_count"] == 0

    def test_schema_version_is_set(self, store):
        """Schema version is recorded after initialization."""
        stats = store.get_stats()
        assert "schema_version" in stats
        assert stats["schema_version"] > 0

    def test_wal_mode_enabled(self, store_on_disk):
        """WAL mode is enabled for concurrent access (file-backed only).

        In-memory databases use 'memory' journal mode, which is the equivalent
        for in-memory operation. WAL only applies to file-backed stores.
        """
        cursor = store_on_disk._conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode.lower() == "wal"

    def test_foreign_keys_enabled(self, store):
        """Foreign key constraints are enforced."""
        cursor = store._conn.execute("PRAGMA foreign_keys")
        assert cursor.fetchone()[0] == 1


class TestInsert:
    """Inserting memories into the store."""

    def test_insert_single_memory(self, store):
        """A memory can be inserted and retrieved by ID."""
        entry = MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="Test memory",
            content="This is a test.",
        )
        stored = store.insert(entry)
        assert stored.id == entry.id
        assert store.get_stats()["memory_count"] == 1

        retrieved = store.get_by_id(entry.id)
        assert retrieved is not None
        assert retrieved.title == "Test memory"
        assert retrieved.type == MemoryType.SEMANTIC

    def test_insert_preserves_all_fields(self, store):
        """All MemoryEntry fields survive a round-trip through the store."""
        entry = MemoryEntry(
            type=MemoryType.PATTERN,
            title="Round trip test",
            content="All fields should survive.",
            tags=["testing", "roundtrip"],
            confidence=0.85,
            provenance=Provenance.CONVERSATION_EXTRACTED,
            source_session_id="session-42",
        )
        stored = store.insert(entry)
        retrieved = store.get_by_id(stored.id)

        assert retrieved.type == MemoryType.PATTERN
        assert retrieved.title == "Round trip test"
        assert retrieved.content == "All fields should survive."
        assert set(retrieved.tags) == {"testing", "roundtrip"}
        assert retrieved.confidence == 0.85
        assert retrieved.provenance == Provenance.CONVERSATION_EXTRACTED
        assert retrieved.source_session_id == "session-42"
        assert retrieved.access_count == 0
        assert retrieved.contradicted_by is None
        assert retrieved.distilled_to is None

    def test_insert_generates_id_if_missing(self, store):
        """If an entry has no ID, one is generated."""
        entry = MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="No ID",
            content="Should get an ID.",
        )
        entry.id = ""  # Explicitly empty
        stored = store.insert(entry)
        assert stored.id
        assert len(stored.id) == 26

    def test_insert_with_tags(self, store):
        """Tags are stored and retrieved correctly."""
        entry = MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="Tagged",
            content="Has tags.",
            tags=["Python", "  testing ", "python"],  # Mixed case, whitespace, duplicate
        )
        stored = store.insert(entry)
        retrieved = store.get_by_id(stored.id)
        # Tags should be normalized
        assert "python" in retrieved.tags
        assert "testing" in retrieved.tags
        assert len(retrieved.tags) == 2


class TestRetrieve:
    """Getting memories by ID and listing."""

    def test_get_by_id_not_found(self, store):
        """None returned for nonexistent ID."""
        assert store.get_by_id("nonexistent") is None

    def test_list_all(self, store):
        """List returns all memories, newest first."""
        e1 = store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="A", content="a"))
        e2 = store.insert(MemoryEntry(type=MemoryType.PROCEDURAL, title="B", content="b"))
        e3 = store.insert(MemoryEntry(type=MemoryType.PATTERN, title="C", content="c"))

        results = store.list_all(limit=10)
        assert len(results) == 3
        assert results[0].id == e3.id  # newest first

    def test_list_all_respects_limit(self, store):
        """Limit parameter caps results."""
        for i in range(10):
            store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title=str(i), content=str(i)))

        results = store.list_all(limit=3)
        assert len(results) == 3

    def test_list_by_type(self, store):
        """List filtered by memory type."""
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="S1", content="s"))
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="S2", content="s"))
        store.insert(MemoryEntry(type=MemoryType.PROCEDURAL, title="P1", content="p"))

        semantic = store.list_by_type(MemoryType.SEMANTIC)
        assert len(semantic) == 2
        procedural = store.list_by_type(MemoryType.PROCEDURAL)
        assert len(procedural) == 1
        pattern = store.list_by_type(MemoryType.PATTERN)
        assert len(pattern) == 0


class TestUpdate:
    """Updating existing memories."""

    def test_update_title_and_content(self, store):
        """Title and content can be updated."""
        entry = store.insert(
            MemoryEntry(type=MemoryType.SEMANTIC, title="Old", content="old content")
        )
        entry.title = "New title"
        entry.content = "new content"
        store.update(entry)

        retrieved = store.get_by_id(entry.id)
        assert retrieved.title == "New title"
        assert retrieved.content == "new content"

    def test_update_confidence(self, store):
        """Confidence can be adjusted."""
        entry = store.insert(
            MemoryEntry(type=MemoryType.SEMANTIC, title="T", content="c", confidence=0.5)
        )
        entry.confidence = 0.9
        store.update(entry)

        retrieved = store.get_by_id(entry.id)
        assert retrieved.confidence == 0.9

    def test_update_tags(self, store):
        """Tags can be updated."""
        entry = store.insert(
            MemoryEntry(type=MemoryType.SEMANTIC, title="T", content="c", tags=["old"])
        )
        entry.tags = ["new", "updated"]
        store.update(entry)

        retrieved = store.get_by_id(entry.id)
        assert set(retrieved.tags) == {"new", "updated"}

    def test_update_mark_contradicted(self, store):
        """Contradiction marker persists."""
        entry = store.insert(
            MemoryEntry(type=MemoryType.SEMANTIC, title="T", content="c")
        )
        entry.mark_contradicted("conflicting-id-123")
        store.update(entry)

        retrieved = store.get_by_id(entry.id)
        assert retrieved.contradicted_by == "conflicting-id-123"
        assert retrieved.confidence == 0.3

    def test_update_mark_distilled(self, store):
        """Distillation marker persists."""
        entry = store.insert(
            MemoryEntry(type=MemoryType.PATTERN, title="T", content="c")
        )
        entry.mark_distilled("my-skill")
        store.update(entry)

        retrieved = store.get_by_id(entry.id)
        assert retrieved.distilled_to == "my-skill"


class TestDelete:
    """Deleting memories."""

    def test_delete_existing(self, store):
        """Existing memory can be deleted."""
        entry = store.insert(
            MemoryEntry(type=MemoryType.SEMANTIC, title="T", content="c")
        )
        assert store.delete(entry.id) is True
        assert store.get_by_id(entry.id) is None

    def test_delete_nonexistent(self, store):
        """Deleting nonexistent returns False."""
        assert store.delete("nonexistent") is False

    def test_delete_cascades_tags(self, store):
        """Deleting a memory also removes its tag associations."""
        entry = store.insert(
            MemoryEntry(type=MemoryType.SEMANTIC, title="T", content="c", tags=["deleteme"])
        )
        store.delete(entry.id)

        # Verify tags are cleaned up
        cursor = store._conn.execute(
            "SELECT COUNT(*) FROM memory_tags WHERE memory_id = ?", (entry.id,)
        )
        assert cursor.fetchone()[0] == 0


class TestSearch:
    """FTS5 full-text search."""

    def test_fts_search_finds_exact_match(self, store):
        """FTS finds memories by title/content text."""
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="Python", content="Python programming"))
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="Java", content="Java programming"))
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="Rust", content="Rust systems"))

        results = store.search_fts("python")
        assert len(results) == 1
        assert results[0].title == "Python"

    def test_fts_search_partial_match(self, store):
        """FTS finds partial word matches."""
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="Deploy", content="deployment pipeline"))
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="Test", content="testing framework"))

        results = store.search_fts("deploy")
        assert len(results) >= 1

    def test_fts_search_no_match(self, store):
        """FTS returns empty when nothing matches."""
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="X", content="y"))
        results = store.search_fts("zzznotfoundzzz")
        assert results == []

    def test_fts_search_with_type_filter(self, store):
        """FTS search respects type filter."""
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="Python config", content="c"))
        store.insert(MemoryEntry(type=MemoryType.PROCEDURAL, title="Python build", content="c"))

        results = store.search_fts("python", type_filter=MemoryType.SEMANTIC)
        assert len(results) == 1
        assert results[0].type == MemoryType.SEMANTIC

    def test_fts_search_respects_limit(self, store):
        """FTS search respects result limit."""
        for i in range(10):
            store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title=f"Memory {i}", content="common"))

        results = store.search_fts("common", limit=3)
        assert len(results) == 3


class TestStats:
    """Health statistics and diagnostics."""

    def test_get_stats_counts(self, store):
        """Stats reflect actual database state."""
        store.insert(MemoryEntry(type=MemoryType.SEMANTIC, title="S", content="s"))
        store.insert(MemoryEntry(type=MemoryType.PROCEDURAL, title="P", content="p"))
        store.insert(MemoryEntry(type=MemoryType.PATTERN, title="Pt", content="pt"))

        stats = store.get_stats()
        assert stats["memory_count"] == 3
        assert "type_counts" in stats
        assert stats["type_counts"]["semantic"] == 1
        assert stats["type_counts"]["procedural"] == 1
        assert stats["type_counts"]["pattern"] == 1

    def test_get_stats_on_empty_store(self, store):
        """Stats on empty store show zeros."""
        stats = store.get_stats()
        assert stats["memory_count"] == 0
        assert stats["type_counts"]["semantic"] == 0
        assert stats["type_counts"]["procedural"] == 0
        assert stats["type_counts"]["pattern"] == 0
