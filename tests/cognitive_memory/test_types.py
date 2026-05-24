"""
Tests for agent/cognitive_memory/types.py — MemoryType, MemoryEntry, Provenance, ULID.

TDD RED phase: these tests define the expected behavior before implementation exists.
"""

import time
import pytest
from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance, generate_ulid


class TestMemoryType:
    """MemoryType enum — exactly 3 types, no more."""

    def test_has_exactly_three_types(self):
        """Only semantic, procedural, and pattern types exist."""
        members = list(MemoryType)
        assert len(members) == 3, f"Expected 3 types, got {len(members)}"
        names = {m.value for m in members}
        assert names == {"semantic", "procedural", "pattern"}

    def test_string_conversion(self):
        """MemoryType values are strings."""
        assert MemoryType.SEMANTIC.value == "semantic"
        assert MemoryType.PROCEDURAL.value == "procedural"
        assert MemoryType.PATTERN.value == "pattern"


class TestProvenance:
    """Provenance enum — tracks why a memory was stored."""

    def test_has_all_provenance_sources(self):
        """Four provenance sources exist."""
        members = list(Provenance)
        assert len(members) == 4
        names = {m.value for m in members}
        assert names == {
            "user_explicit",
            "conversation_extracted",
            "pattern_detected",
            "skill_distilled",
        }

    def test_default_confidence_by_provenance(self):
        """Each provenance source has a sensible default confidence."""
        # user_explicit: user chose to store this → 1.0
        # conversation_extracted: heuristically extracted → 0.7
        # pattern_detected: algorithm found cluster → 0.6
        # skill_distilled: elevated from memory → 1.0 (already proven)
        from agent.cognitive_memory.types import DEFAULT_CONFIDENCE

        assert DEFAULT_CONFIDENCE[Provenance.USER_EXPLICIT] == 1.0
        assert DEFAULT_CONFIDENCE[Provenance.CONVERSATION_EXTRACTED] == 0.7
        assert DEFAULT_CONFIDENCE[Provenance.PATTERN_DETECTED] == 0.6
        assert DEFAULT_CONFIDENCE[Provenance.SKILL_DISTILLED] == 1.0


class TestGenerateULID:
    """ULID generation — unique, sortable, timestamp-based."""

    def test_generates_unique_ids(self):
        """Each call produces a different ULID."""
        ids = {generate_ulid() for _ in range(100)}
        assert len(ids) == 100, "ULIDs must be unique"

    def test_ulid_format(self):
        """ULID is a 26-character string."""
        ulid = generate_ulid()
        assert isinstance(ulid, str)
        assert len(ulid) == 26

    def test_ulid_is_sortable_by_time(self):
        """ULIDs generated later sort after earlier ones."""
        ids = [generate_ulid() for _ in range(10)]
        time.sleep(0.001)  # Ensure timestamp difference
        later = generate_ulid()
        assert all(uid < later for uid in ids), "Later ULIDs must sort after earlier ones"


class TestMemoryEntry:
    """MemoryEntry — the core data structure for typed memory."""

    def test_create_minimal_entry(self):
        """Minimal entry: type, title, content. Rest gets defaults."""
        entry = MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="Project uses pytest",
            content="This project uses pytest for testing with xdist for parallel runs.",
        )
        assert entry.type == MemoryType.SEMANTIC
        assert entry.title == "Project uses pytest"
        assert "pytest" in entry.content
        assert isinstance(entry.id, str)
        assert len(entry.id) == 26  # ULID
        assert entry.tags == []
        assert entry.confidence == 1.0  # default provenance = user_explicit
        assert entry.provenance == Provenance.USER_EXPLICIT
        assert entry.contradicted_by is None
        assert entry.distilled_to is None
        assert isinstance(entry.created_at, float)
        # Both times set via default_factory=time.time, may differ by microseconds
        assert abs(entry.updated_at - entry.created_at) < 0.01
        assert entry.access_count == 0
        assert entry.last_accessed is None
        assert entry.source_session_id is None

    def test_create_with_all_fields(self):
        """Full entry with explicit provenance and metadata."""
        entry = MemoryEntry(
            type=MemoryType.PATTERN,
            title="Always use pytest -x for fast failure",
            content="When running tests, use -x flag to stop on first failure.",
            tags=["testing", "pytest", "convention"],
            confidence=0.8,
            provenance=Provenance.CONVERSATION_EXTRACTED,
            source_session_id="abc123",
        )
        assert entry.type == MemoryType.PATTERN
        assert entry.tags == ["testing", "pytest", "convention"]
        assert entry.confidence == 0.8
        assert entry.provenance == Provenance.CONVERSATION_EXTRACTED
        assert entry.source_session_id == "abc123"

    def test_confidence_clamped_to_range(self):
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValueError, match="confidence"):
            MemoryEntry(
                type=MemoryType.SEMANTIC,
                title="test",
                content="test",
                confidence=1.5,
            )
        with pytest.raises(ValueError, match="confidence"):
            MemoryEntry(
                type=MemoryType.SEMANTIC,
                title="test",
                content="test",
                confidence=-0.1,
            )

    def test_tags_are_deduplicated_and_normalized(self):
        """Tags are stored lowercase, no duplicates, whitespace stripped."""
        entry = MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="test",
            content="test",
            tags=["  Python ", "python", "PYTHON", " testing "],
        )
        assert entry.tags == ["python", "testing"]

    def test_mark_contradicted(self):
        """An entry can be marked as contradicted by another entry."""
        entry = MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="Build uses Make",
            content="The build system is Make.",
        )
        entry.mark_contradicted("ulid_of_newer_entry")
        assert entry.contradicted_by == "ulid_of_newer_entry"
        assert entry.confidence == 0.3  # Dropped to low confidence

    def test_mark_distilled(self):
        """An entry can be marked as distilled into a skill."""
        entry = MemoryEntry(
            type=MemoryType.PATTERN,
            title="Run tests with coverage",
            content="Always run pytest --cov.",
        )
        entry.mark_distilled("testing-with-coverage")
        assert entry.distilled_to == "testing-with-coverage"
        # Confidence unchanged — distillation is elevation, not doubt

    def test_register_access(self):
        """Access tracking: count increments, last_accessed updates."""
        entry = MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="test",
            content="test",
        )
        assert entry.access_count == 0
        assert entry.last_accessed is None

        entry.register_access()
        assert entry.access_count == 1
        assert entry.last_accessed is not None

        entry.register_access()
        assert entry.access_count == 2

    def test_equality_by_id(self):
        """Two entries are equal if they have the same ID."""
        e1 = MemoryEntry(type=MemoryType.SEMANTIC, title="a", content="a")
        e2 = MemoryEntry(type=MemoryType.PROCEDURAL, title="b", content="b")
        e2.id = e1.id  # Force same ID for test
        assert e1 == e2
        assert hash(e1) == hash(e2)

    def test_repr_is_useful(self):
        """repr should show type, title, and confidence."""
        entry = MemoryEntry(
            type=MemoryType.PATTERN,
            title="Do not use sudo without asking",
            content="sudo is dangerous.",
            confidence=0.9,
        )
        r = repr(entry)
        assert "pattern" in r.lower()
        assert "Do not use sudo" in r
        assert "0.9" in r
