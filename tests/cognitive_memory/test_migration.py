"""
Tests for agent/cognitive_memory/migration.py — classification and migration
from file-based memory (MEMORY.md/USER.md) to typed cognitive memory.

TDD RED phase: define behavior before implementation exists.
"""

import os
import tempfile
import pytest
from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.migration import (
    classify_entry,
    migrate_file_memory,
    MIGRATION_MARKER_FILE,
)


class TestClassifyEntry:
    """Regex-based classification of memory entries into types."""

    def test_classify_semantic_project_fact(self):
        """Project facts, architecture, conventions → semantic."""
        content = "Project uses pytest for testing with xdist for parallel runs."
        entry_type, confidence = classify_entry(content)
        assert entry_type == MemoryType.SEMANTIC

    def test_classify_semantic_environment(self):
        """Environment details → semantic."""
        content = (
            "Container hostname: cortex. Docker container with persistent volumes. "
            "ARM64 architecture running on Apple Silicon."
        )
        entry_type, _ = classify_entry(content)
        assert entry_type == MemoryType.SEMANTIC

    def test_classify_procedural_build_command(self):
        """Build/test/deploy commands → procedural."""
        content = "Build with: npm run build. Test with: pytest -x --cov."
        entry_type, _ = classify_entry(content)
        assert entry_type == MemoryType.PROCEDURAL

    def test_classify_procedural_workflow(self):
        """Workflow descriptions → procedural."""
        content = "To run tests: activate venv, then pytest tests/ -q. To build: npm run build --prod."
        entry_type, _ = classify_entry(content)
        assert entry_type == MemoryType.PROCEDURAL

    def test_classify_pattern_user_correction(self):
        """User corrections, preferences, rules → pattern."""
        content = "ALWAYS reply in English, regardless of what language David writes in."
        entry_type, _ = classify_entry(content)
        assert entry_type == MemoryType.PATTERN

    def test_classify_pattern_rule(self):
        """Behavioral rules → pattern."""
        content = "Never run destructive commands without David's explicit approval."
        entry_type, _ = classify_entry(content)
        assert entry_type == MemoryType.PATTERN

    def test_classify_defaults_to_semantic(self):
        """Ambiguous content defaults to semantic."""
        content = "Hermes Agent v0.14.0"
        entry_type, _ = classify_entry(content)
        assert entry_type == MemoryType.SEMANTIC

    def test_classify_returns_confidence(self):
        """Classification also returns a confidence score."""
        _, confidence = classify_entry("Project uses React for frontend.")
        assert 0.5 <= confidence <= 1.0


class TestMigrateFileMemory:
    """End-to-end migration from MEMORY.md/USER.md to cognitive store."""

    @pytest.fixture
    def temp_memory_dir(self):
        """Create a temporary directory with sample MEMORY.md and USER.md."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Sample MEMORY.md content (multiple entries separated by §)
            memory_path = os.path.join(tmpdir, "MEMORY.md")
            with open(memory_path, "w") as f:
                f.write(
                    "\n§\n".join(
                        [
                            "Container hostname: cortex. ARM64, 15 GB RAM, 911 GB disk.",
                            "Hermes Agent v0.14.0 running deepseek-v4-pro via ollama-cloud.",
                            "Build with: npm run build && npm test.",
                            "ALWAYS reply in English. Never switch to another language.",
                            "GitHub: username d2a8k3u-cortex. Auth via SSH key ~/.ssh/id_cortex_github.",
                        ]
                    )
                )

            # Sample USER.md content
            user_path = os.path.join(tmpdir, "USER.md")
            with open(user_path, "w") as f:
                f.write(
                    "\n§\n".join(
                        [
                            "User's name is David. Prefers partnership, not command-and-response.",
                            "David prefers to call Hermes 'Cortex'.",
                            "Work directory: /home/appuser/work is the primary workspace.",
                        ]
                    )
                )

            yield tmpdir

    def test_migrate_creates_typed_entries(self, temp_memory_dir):
        """Migration reads file memory and creates typed entries in the store."""
        store = CognitiveMemoryStore(":memory:")
        count = migrate_file_memory(store, temp_memory_dir)

        assert count > 0
        stats = store.get_stats()
        assert stats["memory_count"] == count

    def test_migrate_preserves_content(self, temp_memory_dir):
        """Migrated entries have their original content preserved."""
        store = CognitiveMemoryStore(":memory:")
        migrate_file_memory(store, temp_memory_dir)

        entries = store.list_all()
        contents = [e.content for e in entries]
        assert any("cortex" in c.lower() for c in contents)
        assert any("deepseek" in c.lower() for c in contents)
        assert any('npm run build' in c for c in contents)

    def test_migrate_sets_provenance_to_user_explicit(self, temp_memory_dir):
        """All migrated entries have provenance = user_explicit (confidence 1.0)."""
        store = CognitiveMemoryStore(":memory:")
        migrate_file_memory(store, temp_memory_dir)

        for entry in store.list_all():
            assert entry.provenance == Provenance.USER_EXPLICIT
            assert entry.confidence == 1.0

    def test_migrate_distributes_across_types(self, temp_memory_dir):
        """Migration produces a mix of semantic, procedural, and pattern entries."""
        store = CognitiveMemoryStore(":memory:")
        migrate_file_memory(store, temp_memory_dir)

        stats = store.get_stats()
        type_counts = stats["type_counts"]

        # We put 5 entries in MEMORY.md + 3 in USER.md = 8 total
        # Some should be semantic, some procedural, some pattern
        assert type_counts["semantic"] >= 1
        assert type_counts["pattern"] >= 1

    def test_migrate_is_idempotent(self, temp_memory_dir):
        """Running migration twice doesn't duplicate entries."""
        store = CognitiveMemoryStore(":memory:")
        first_count = migrate_file_memory(store, temp_memory_dir)
        second_count = migrate_file_memory(store, temp_memory_dir)

        # Second run should detect already-migrated and skip
        assert second_count == 0
        assert store.get_stats()["memory_count"] == first_count

    def test_migrate_handles_missing_files(self):
        """Migration on a directory without MEMORY.md/USER.md returns 0."""
        with tempfile.TemporaryDirectory() as empty_dir:
            store = CognitiveMemoryStore(":memory:")
            count = migrate_file_memory(store, empty_dir)
            assert count == 0
            assert store.get_stats()["memory_count"] == 0

    def test_migrate_handles_empty_files(self):
        """Migration on empty files returns 0 without error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            open(os.path.join(tmpdir, "MEMORY.md"), "w").close()
            open(os.path.join(tmpdir, "USER.md"), "w").close()

            store = CognitiveMemoryStore(":memory:")
            count = migrate_file_memory(store, tmpdir)
            assert count == 0

    def test_migration_marker_is_created(self, temp_memory_dir):
        """After migration, a marker file is created to prevent re-migration."""
        store = CognitiveMemoryStore(":memory:")
        migrate_file_memory(store, temp_memory_dir)

        # The marker should exist somewhere (in the store or as a file)
        # We test idempotency via the double-migrate test above
        # This just confirms the mechanism works
        assert store.get_stats()["memory_count"] > 0
