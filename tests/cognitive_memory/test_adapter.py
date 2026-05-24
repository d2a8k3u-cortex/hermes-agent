"""
Tests for agent/cognitive_memory/adapter.py — dual-write adapter that hooks
into the existing MemoryStore to also write to the cognitive store.

TDD RED phase: define behavior before implementation exists.
"""

import os
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.adapter import CognitiveMemoryAdapter


@pytest.fixture
def cognitive_store():
    """Fresh in-memory cognitive store."""
    store = CognitiveMemoryStore(":memory:")
    yield store
    store.close()


@pytest.fixture
def adapter(cognitive_store):
    """Adapter wrapping a cognitive store."""
    return CognitiveMemoryAdapter(cognitive_store)


@pytest.fixture
def mock_memory_store():
    """Mock existing file-based MemoryStore."""
    mock = MagicMock()
    mock._entries_for = lambda target: (
        [] if target == "memory" else []
    )
    mock.memory_entries = []
    mock.user_entries = []
    return mock


class TestAdapterCreation:
    """Adapter initialization."""

    def test_creates_with_cognitive_store(self, cognitive_store):
        """Adapter wraps a cognitive store."""
        adapter = CognitiveMemoryAdapter(cognitive_store)
        assert adapter is not None

    def test_disabled_without_cognitive_store(self):
        """Adapter returns None or disabled when no store provided."""
        adapter = CognitiveMemoryAdapter(None)
        assert adapter.enabled is False

    def test_enabled_with_valid_store(self, cognitive_store):
        """Adapter is enabled when a store is provided."""
        adapter = CognitiveMemoryAdapter(cognitive_store)
        assert adapter.enabled is True


class TestMirrorAdd:
    """Mirroring MemoryStore.add() calls to cognitive store."""

    def test_mirror_add_writes_to_cognitive(self, cognitive_store, mock_memory_store):
        """When MemoryStore.add() is called, entry appears in cognitive store."""
        adapter = CognitiveMemoryAdapter(cognitive_store)

        adapter.mirror_add("memory", "Container hostname: cortex. ARM64 Docker container.")

        stats = cognitive_store.get_stats()
        assert stats["memory_count"] == 1

        # Check the mirrored entry
        entries = cognitive_store.list_all()
        assert len(entries) == 1
        assert "cortex" in entries[0].content
        assert entries[0].provenance == Provenance.USER_EXPLICIT
        assert entries[0].confidence == 1.0

    def test_mirror_add_user_target(self, cognitive_store):
        """User entries are mirrored as semantic memories."""
        adapter = CognitiveMemoryAdapter(cognitive_store)

        adapter.mirror_add("user", "David prefers to be called Cortex.")

        entries = cognitive_store.list_all()
        assert len(entries) == 1
        assert "Cortex" in entries[0].content
        assert entries[0].type == MemoryType.PATTERN  # User preferences → pattern

    def test_mirror_add_classifies_type(self, cognitive_store):
        """Entries are classified by type on mirror."""
        adapter = CognitiveMemoryAdapter(cognitive_store)

        # Procedural content
        adapter.mirror_add("memory", "Build with: npm run build && npm test")
        entries = cognitive_store.list_all()
        assert entries[0].type == MemoryType.PROCEDURAL

    def test_mirror_add_uses_title_from_first_sentence(self, cognitive_store):
        """Mirrored entries derive title from content."""
        adapter = CognitiveMemoryAdapter(cognitive_store)

        adapter.mirror_add("memory", "Project uses pytest for testing. It runs with xdist.")

        entries = cognitive_store.list_all()
        assert entries[0].title == "Project uses pytest for testing"

    def test_mirror_add_skips_when_disabled(self):
        """No-op when adapter is disabled."""
        adapter = CognitiveMemoryAdapter(None)  # disabled
        result = adapter.mirror_add("memory", "test")
        assert adapter.enabled is False
        # Should not raise or error

    def test_mirror_add_returns_entry_id(self, cognitive_store):
        """mirror_add returns the ULID of the created entry."""
        adapter = CognitiveMemoryAdapter(cognitive_store)
        entry_id = adapter.mirror_add("memory", "Test content for ID check.")
        assert entry_id is not None
        assert len(entry_id) == 26


class TestMirrorReplace:
    """Mirroring MemoryStore.replace() calls."""

    def test_mirror_replace_updates_cognitive(self, cognitive_store):
        """Replace updates the mirrored entry in cognitive store."""
        adapter = CognitiveMemoryAdapter(cognitive_store)

        # First add, then replace
        old_id = adapter.mirror_add("memory", "Build uses Make. Old approach.")
        adapter.mirror_replace("memory", "Make", "Build uses Ninja now. Updated approach.")

        entries = cognitive_store.list_all()
        assert len(entries) == 1
        assert "Ninja" in entries[0].content
        assert "Make" not in entries[0].content

    def test_mirror_replace_no_match(self, cognitive_store):
        """If no matching entry found in cognitive store, replace is a no-op."""
        adapter = CognitiveMemoryAdapter(cognitive_store)
        # No prior add, replace should not create anything
        adapter.mirror_replace("memory", "nonexistent", "replacement")

        stats = cognitive_store.get_stats()
        assert stats["memory_count"] == 0


class TestMirrorRemove:
    """Mirroring MemoryStore.remove() calls."""

    def test_mirror_remove_deletes_from_cognitive(self, cognitive_store):
        """Remove deletes the mirrored entry."""
        adapter = CognitiveMemoryAdapter(cognitive_store)

        adapter.mirror_add("memory", "This entry will be removed.")
        assert cognitive_store.get_stats()["memory_count"] == 1

        adapter.mirror_remove("memory", "will be removed")
        assert cognitive_store.get_stats()["memory_count"] == 0

    def test_mirror_remove_no_match(self, cognitive_store):
        """No-op when no matching entry."""
        adapter = CognitiveMemoryAdapter(cognitive_store)
        adapter.mirror_remove("memory", "nonexistent")
        assert cognitive_store.get_stats()["memory_count"] == 0


class TestRealMemoryStoreIntegration:
    """Integration tests with actual MemoryStore (not mock)."""

    @pytest.fixture
    def temp_memory_dir(self, monkeypatch):
        """Patch get_memory_dir to use a temp directory."""
        import tools.memory_tool

        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setattr(tools.memory_tool, "get_memory_dir", lambda: Path(tmpdir))
            yield tmpdir

    def test_dual_write_on_add(self, temp_memory_dir, cognitive_store):
        """Adding to MemoryStore also writes to cognitive store."""
        from tools.memory_tool import MemoryStore

        file_store = MemoryStore()
        adapter = CognitiveMemoryAdapter(cognitive_store)

        # The actual dual-write happens by injecting adapter calls
        # into MemoryStore.add() — this test validates the adapter itself
        content = "Production uses Kubernetes on ARM64."
        entry_id = adapter.mirror_add("memory", content)

        retrieved = cognitive_store.get_by_id(entry_id)
        assert retrieved is not None
        assert "Kubernetes" in retrieved.content
