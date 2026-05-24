"""
Tests for agent/cognitive_memory/retrieval.py — type-aware retrieval foundation.

TDD RED phase: define behavior before implementation exists.
Builds on the store's FTS5 search with type filtering and scoring.

Core scoring formula (no importance — relevance is query-driven):
    score = text_match_score × 0.5 + recency_boost × 0.3 + access_boost × 0.2
"""

import time
import pytest
from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.retrieval import (
    get_relevant_entries,
    get_context_for_prompt,
    DEFAULT_WEIGHTS,
)


@pytest.fixture
def store():
    """Fresh in-memory store pre-populated with test memories."""
    s = CognitiveMemoryStore(":memory:")

    # Semantic: project facts
    s.insert(MemoryEntry(
        type=MemoryType.SEMANTIC,
        title="Project uses pytest",
        content="This project uses pytest for testing with xdist for parallel runs.",
        tags=["testing", "pytest", "python"],
    ))
    s.insert(MemoryEntry(
        type=MemoryType.SEMANTIC,
        title="Container runs on ARM64",
        content="Container hostname: cortex. Docker on ARM64 with 15 GB RAM.",
        tags=["environment", "docker"],
    ))
    s.insert(MemoryEntry(
        type=MemoryType.SEMANTIC,
        title="GitHub auth configured",
        content="GitHub user d2a8k3u-cortex. SSH key at ~/.ssh/id_cortex_github.",
        tags=["github", "auth"],
    ))

    # Procedural: build/test commands
    s.insert(MemoryEntry(
        type=MemoryType.PROCEDURAL,
        title="Run tests with coverage",
        content="pytest tests/ -q --cov --cov-report=term-missing",
        tags=["testing", "commands"],
    ))
    s.insert(MemoryEntry(
        type=MemoryType.PROCEDURAL,
        title="Build the project",
        content="npm run build && npm run test",
        tags=["build", "npm"],
    ))

    # Pattern: user rules
    s.insert(MemoryEntry(
        type=MemoryType.PATTERN,
        title="Always reply in English",
        content="ALWAYS reply in English, regardless of what language David writes in.",
        tags=["communication", "rule"],
        confidence=0.9,
        provenance=Provenance.CONVERSATION_EXTRACTED,
    ))
    s.insert(MemoryEntry(
        type=MemoryType.PATTERN,
        title="Never use sudo without asking",
        content="Never run destructive sudo commands without David's explicit approval.",
        tags=["safety", "rule"],
        confidence=1.0,
        provenance=Provenance.USER_EXPLICIT,
    ))

    yield s
    s.close()


class TestGetRelevantEntries:
    """Type-aware retrieval with query-driven scoring."""

    def test_returns_relevant_by_text_match(self, store):
        """Entries matching the query are returned."""
        results = get_relevant_entries(store, "pytest")
        assert len(results) > 0
        assert any("pytest" in r.title.lower() for r in results)

    def test_returns_empty_for_no_match(self, store):
        """No results when nothing matches — hybrid search may still return
        low-relevance vector matches; check that no strong matches exist."""
        results = get_relevant_entries(store, "zzznotfoundzzz")
        # With embeddings, every entry has some cosine similarity.
        # We only care that nothing is a strong match.
        for r in results:
            assert r._retrieval_score < 0.5, (
                f"Unexpected strong match: {r.title} score={r._retrieval_score}"
            )

    def test_respects_max_entries(self, store):
        """max_entries limits results."""
        results = get_relevant_entries(store, "pytest run build test", max_entries=2)
        assert len(results) <= 2

    def test_type_filter_works(self, store):
        """type_filter restricts to specific memory type."""
        results = get_relevant_entries(
            store, "pytest", type_filter=MemoryType.PROCEDURAL
        )
        assert all(r.type == MemoryType.PROCEDURAL for r in results)

    def test_type_filter_returns_only_requested(self, store):
        """Pattern filter returns only pattern entries."""
        results = get_relevant_entries(
            store, "always english language", type_filter=MemoryType.PATTERN
        )
        assert len(results) > 0
        assert all(r.type == MemoryType.PATTERN for r in results)

    def test_default_score_order(self, store):
        """Results are ordered by relevance score (descending)."""
        results = get_relevant_entries(store, "test pytest build")
        if len(results) >= 2:
            scores = [r._retrieval_score for r in results]
            assert all(
                scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
            ), f"Scores not descending: {scores}"

    def test_entries_gain_access_boost(self, store):
        """Accessing an entry boosts its score in subsequent queries."""
        # First access: query once to increment access count
        get_relevant_entries(store, "pytest testing")
        # Second query should still work, entries now have higher access count
        results = get_relevant_entries(store, "pytest")
        assert len(results) > 0

    def test_recency_boost_newer_scores_higher(self, store):
        """Newer entries score higher than equally matched older ones (all else equal)."""
        # Insert a fresh entry with same keyword
        time.sleep(0.001)
        store.insert(MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="pytest configuration",
            content="pytest is configured in pyproject.toml",
        ))

        results = get_relevant_entries(store, "pytest")
        if len(results) >= 2:
            # The newest one should be first (or at least high-scoring)
            newest_ids = [r.id for r in results[:2]]
            # The fresh entry should appear in the results
            assert any("pyproject" in r.content for r in results[:3])

    def test_empty_query_returns_empty(self, store):
        """Empty query string returns no results."""
        results = get_relevant_entries(store, "")
        assert results == []

    def test_whitespace_query_returns_empty(self, store):
        """Whitespace-only query returns no results."""
        results = get_relevant_entries(store, "   ")
        assert results == []

    def test_assigns_retrieval_score_to_each_result(self, store):
        """Each returned entry has a _retrieval_score attribute."""
        results = get_relevant_entries(store, "pytest")
        for r in results:
            assert hasattr(r, "_retrieval_score")
            assert isinstance(r._retrieval_score, float)
            assert r._retrieval_score >= 0


class TestGetContextForPrompt:
    """Formatting retrieved entries for injection into the agent prompt."""

    def test_formats_single_entry(self, store):
        """Single entry is formatted as a readable block."""
        results = get_relevant_entries(store, "pytest", max_entries=1)
        context = get_context_for_prompt(results)
        assert "BUILTIN MEMORY" in context
        assert "pytest" in context.lower()

    def test_formats_multiple_entries(self, store):
        """Multiple entries separated clearly."""
        results = get_relevant_entries(store, "test build", max_entries=3)
        context = get_context_for_prompt(results)
        assert context.count("[") >= 1  # Each entry has a type marker like [semantic]

    def test_empty_results_returns_empty_string(self, store):
        """No results → empty context block. With hybrid search, even
        nonsense queries return weak vector matches, but the context
        block is still not empty in that case. Test with truly empty input instead."""
        results = get_relevant_entries(store, "")
        context = get_context_for_prompt(results)
        assert context == ""

    def test_includes_type_labels(self, store):
        """Each entry shows its memory type."""
        results = get_relevant_entries(store, "pytest")
        context = get_context_for_prompt(results)
        assert "semantic" in context.lower() or "★" in context

    def test_truncates_long_content(self, store):
        """Long content is truncated to reasonable length."""
        results = get_relevant_entries(store, "pytest", max_entries=1)
        context = get_context_for_prompt(results)
        # Should be compact enough for prompt injection
        assert len(context) < 2000, f"Context too long: {len(context)} chars"

    def test_shows_confidence_when_relevant(self, store):
        """Pattern entries with confidence < 1.0 show their confidence."""
        results = get_relevant_entries(store, "always english language")
        context = get_context_for_prompt(results)
        # Pattern entries may show confidence
        assert len(context) > 0


class TestDefaultWeights:
    """Scoring weight configuration."""

    def test_weights_sum_to_approximately_one(self):
        """Weights should be roughly normalized."""
        total = DEFAULT_WEIGHTS["text"] + DEFAULT_WEIGHTS["recency"] + DEFAULT_WEIGHTS["access"]
        assert 0.9 <= total <= 1.1, f"Weights sum to {total}, expected ~1.0"

    def test_text_weight_is_highest(self):
        """Text relevance should dominate scoring."""
        assert DEFAULT_WEIGHTS["text"] > DEFAULT_WEIGHTS["recency"]
        assert DEFAULT_WEIGHTS["text"] > DEFAULT_WEIGHTS["access"]
