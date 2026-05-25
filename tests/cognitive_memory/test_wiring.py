"""
Test that cognitive memory wiring works end-to-end — writes are mirrored,
injection context is produced, and the default path is unchanged when disabled.
"""

import tempfile
import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock, PropertyMock


class TestEndToEndCognitiveWiring:
    """End-to-end: MemoryStore writes mirror to cognitive store."""

    def test_add_mirrors_to_cognitive(self):
        """MemoryStore.add() writes to cognitive store when adapter is wired."""
        from tools.memory_tool import MemoryStore
        from agent.cognitive_memory.store import CognitiveMemoryStore
        from agent.cognitive_memory.adapter import CognitiveMemoryAdapter
        from agent.cognitive_memory.types import MemoryType

        # Setup
        cog_store = CognitiveMemoryStore(":memory:")
        adapter = CognitiveMemoryAdapter(cog_store)

        with tempfile.TemporaryDirectory() as tmpdir:
            memory_path = Path(tmpdir) / "MEMORY.md"
            user_path = Path(tmpdir) / "USER.md"

            file_store = MemoryStore()
            file_store._cognitive_adapter = adapter

            # Patch the file paths to use temp dir
            file_store._path_for = lambda target: user_path if target == "user" else memory_path
            # Bypass file lock in tests
            file_store._file_lock = lambda path: contextlib_null()

            # Add via file store
            result = file_store.add("memory", "Project uses pytest with xdist for parallel testing.")
            assert result["success"] is True

            # Verify it appeared in cognitive store
            entries = cog_store.list_all()
            assert len(entries) >= 1
            assert any("pytest" in e.content for e in entries)

            cog_store.close()

    def test_replace_mirrors_to_cognitive(self):
        """MemoryStore.replace() updates the cognitive entry."""
        from tools.memory_tool import MemoryStore
        from agent.cognitive_memory.store import CognitiveMemoryStore
        from agent.cognitive_memory.adapter import CognitiveMemoryAdapter

        cog_store = CognitiveMemoryStore(":memory:")
        adapter = CognitiveMemoryAdapter(cog_store)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_store = MemoryStore()
            file_store._cognitive_adapter = adapter
            file_store._path_for = lambda t: Path(tmpdir) / (f"{'USER' if t == 'user' else 'MEMORY'}.md")
            file_store._file_lock = lambda path: contextlib_null()

            # Add, then replace
            file_store.add("memory", "Build uses Make.")
            file_store.replace("memory", "Make", "Build uses Ninja now.")

            entries = cog_store.list_all()
            assert len(entries) == 1
            assert "Ninja" in entries[0].content

            cog_store.close()

    def test_remove_mirrors_to_cognitive(self):
        """MemoryStore.remove() deletes the cognitive entry."""
        from tools.memory_tool import MemoryStore
        from agent.cognitive_memory.store import CognitiveMemoryStore
        from agent.cognitive_memory.adapter import CognitiveMemoryAdapter

        cog_store = CognitiveMemoryStore(":memory:")
        adapter = CognitiveMemoryAdapter(cog_store)

        with tempfile.TemporaryDirectory() as tmpdir:
            file_store = MemoryStore()
            file_store._cognitive_adapter = adapter
            file_store._path_for = lambda t: Path(tmpdir) / (f"{'USER' if t == 'user' else 'MEMORY'}.md")
            file_store._file_lock = lambda path: contextlib_null()

            file_store.add("memory", "This will be removed.")
            file_store.remove("memory", "will be removed")

            entries = cog_store.list_all()
            assert len(entries) == 0

            cog_store.close()

    def test_disabled_by_default(self):
        """Without cognitive_memory.enabled, nothing changes."""
        from tools.memory_tool import MemoryStore

        file_store = MemoryStore()
        assert file_store._cognitive_enabled is False
        assert file_store._cognitive_adapter is None


class TestCognitiveInjectionContext:
    """Injection context is produced when cognitive engine is active."""

    def test_injection_returns_context_for_query(self):
        """A query against the cognitive store returns formatted context."""
        from agent.cognitive_memory.store import CognitiveMemoryStore
        from agent.cognitive_memory.types import MemoryType, MemoryEntry
        from agent.cognitive_memory.injection import inject_memory_context, PerTurnDedupTracker

        store = CognitiveMemoryStore(":memory:")
        store.insert(MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="Pytest config",
            content="Project uses pytest with xdist for parallel testing.",
        ))

        context = inject_memory_context(
            store, "How do I run tests?", PerTurnDedupTracker()
        )
        assert len(context) > 0
        assert "pytest" in context.lower()

        store.close()

    def test_injection_empty_for_no_match(self):
        """No context when nothing matches."""
        from agent.cognitive_memory.store import CognitiveMemoryStore
        from agent.cognitive_memory.injection import inject_memory_context, PerTurnDedupTracker

        store = CognitiveMemoryStore(":memory:")
        context = inject_memory_context(store, "", PerTurnDedupTracker())
        assert context == ""
        store.close()


def contextlib_null():
    """Minimal null context manager to bypass file locks in tests."""
    from contextlib import contextmanager
    @contextmanager
    def _null():
        yield
    return _null()
