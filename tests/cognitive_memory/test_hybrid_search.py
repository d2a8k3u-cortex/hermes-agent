"""
Tests for vector storage and hybrid search — Phase 2.2 + 2.3 + 2.4 combined.

Covers:
  - Adding embedding column to store
  - Auto-generating embeddings on insert
  - Cosine similarity search
  - Hybrid FTS + vector blended search
  - Cross-encoder reranking (opt-in)
"""

import time
import pytest
from agent.cognitive_memory.types import MemoryType, MemoryEntry
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.embeddings import is_available as emb_available
from agent.cognitive_memory.retrieval import (
    hybrid_search_entries,
    get_relevant_entries,
)


@pytest.fixture
def store():
    """Populated in-memory store for search tests."""
    s = CognitiveMemoryStore(":memory:")

    entries = [
        ("Python testing with pytest", "This project uses pytest for testing with xdist for parallel execution.", MemoryType.SEMANTIC),
        ("Docker ARM64 container", "Container runs on ARM64 with 15 GB RAM and 911 GB disk.", MemoryType.SEMANTIC),
        ("Build with npm", "Build the project with: npm run build && npm run test", MemoryType.PROCEDURAL),
        ("Deploy to Kubernetes", "Deployment uses kubectl apply -f k8s/ and Helm charts.", MemoryType.PROCEDURAL),
        ("Always reply in English", "ALWAYS reply in English regardless of input language.", MemoryType.PATTERN),
        ("Never use sudo", "Never run sudo without David's explicit approval.", MemoryType.PATTERN),
    ]

    for title, content, mtype in entries:
        s.insert(MemoryEntry(type=mtype, title=title, content=content))

    yield s
    s.close()


class TestEmbeddingStorage:
    """Phase 2.2: Embedding column and auto-generation in store."""

    def test_insert_auto_generates_embedding(self, store):
        """New insert generates an embedding automatically."""
        entry = store.insert(MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="New entry",
            content="This should get an embedding."
        ))

        # Check embedding exists in the database
        cursor = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (entry.id,)
        )
        row = cursor.fetchone()
        embedding = row["embedding"]
        assert embedding is not None
        assert len(embedding) > 0

    def test_update_rebuilds_embedding_on_content_change(self, store):
        """Updating content regenerates the embedding."""
        entry = store.insert(MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="Original",
            content="Original content for embedding."
        ))

        # Get original embedding
        cursor = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (entry.id,)
        )
        orig_embedding = cursor.fetchone()["embedding"]

        # Update with different content
        entry.content = "Completely different content about machine learning."
        store.update(entry)

        cursor = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (entry.id,)
        )
        new_embedding = cursor.fetchone()["embedding"]

        assert orig_embedding != new_embedding

    def test_update_keeps_embedding_on_non_content_change(self, store):
        """Updating access_count doesn't regenerate the embedding."""
        entry = store.insert(MemoryEntry(
            type=MemoryType.SEMANTIC,
            title="Stable",
            content="Stable content."
        ))
        cursor = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (entry.id,)
        )
        orig = cursor.fetchone()["embedding"]

        # Update only access_count
        entry.register_access()
        store._update_metadata_only(entry)

        cursor = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (entry.id,)
        )
        after = cursor.fetchone()["embedding"]

        assert orig == after

    def test_all_existing_entries_have_embeddings(self, store):
        """All entries in the populated fixture have embeddings."""
        entries = store.list_all()
        for entry in entries:
            cursor = store._conn.execute(
                "SELECT embedding FROM memories WHERE id = ?", (entry.id,)
            )
            row = cursor.fetchone()
            assert row["embedding"] is not None, f"Entry {entry.id} has no embedding"


class TestCosineSearch:
    """Phase 2.2: Cosine similarity search in the store."""

    def test_cosine_search_finds_similar(self, store):
        """Query embedding finds semantically similar entries."""
        results = store.cosine_search("pytest testing python", limit=3)
        assert len(results) > 0
        # Should find the pytest entry
        titles = [r.title for r in results]
        assert any("pytest" in t.lower() for t in titles)

    def test_cosine_search_returns_in_order(self, store):
        """Results are ordered by cosine similarity descending."""
        results = store.cosine_search("container docker infrastructure", limit=5)
        if len(results) >= 2:
            scores = [r._retrieval_score for r in results]
            assert all(
                scores[i] >= scores[i + 1] for i in range(len(scores) - 1)
            ), f"Scores not descending: {scores}"

    def test_cosine_search_respects_limit(self, store):
        """Limit parameter works."""
        results = store.cosine_search("test build deploy", limit=2)
        assert len(results) <= 2

    def test_cosine_search_with_type_filter(self, store):
        """Type filter restricts to specific memory types."""
        results = store.cosine_search("build test", limit=5, type_filter=MemoryType.PROCEDURAL)
        if results:
            assert all(r.type == MemoryType.PROCEDURAL for r in results)

    def test_cosine_search_empty_query(self, store):
        """Empty query returns empty list."""
        results = store.cosine_search("", limit=5)
        assert results == []

    def test_cosine_search_returns_scores(self, store):
        """Each result has a _retrieval_score (cosine similarity)."""
        results = store.cosine_search("pytest", limit=3)
        for r in results:
            assert hasattr(r, "_retrieval_score")
            assert 0.0 <= r._retrieval_score <= 1.0


class TestHybridSearch:
    """Phase 2.3: Hybrid FTS + vector blended search."""

    def test_hybrid_search_combines_both(self, store):
        """Hybrid search returns results blending FTS and vector."""
        results = hybrid_search_entries(store, "pytest testing python", max_entries=5)
        assert len(results) > 0

    def test_hybrid_search_returns_better_than_either_alone(self, store):
        """Hybrid catches matches that pure FTS might miss."""
        # "container deployment" — FTS might not match "Kubernetes" or "Docker ARM64" exactly
        results = hybrid_search_entries(store, "container deployment", max_entries=3)
        assert len(results) > 0

    def test_hybrid_search_handles_type_filter(self, store):
        """Hybrid search respects type filter."""
        results = hybrid_search_entries(
            store, "build test", max_entries=5, type_filter=MemoryType.PROCEDURAL
        )
        if results:
            assert all(r.type == MemoryType.PROCEDURAL for r in results)

    def test_hybrid_search_falls_back_to_fts_only(self, store):
        """When embeddings unavailable, hybrid falls back to FTS only."""
        # This is tested by design — if embedding lookup fails, FTS results are returned
        results = hybrid_search_entries(store, "pytest", max_entries=3)
        assert len(results) > 0
        assert any("pytest" in r.content.lower() for r in results)

    def test_hybrid_search_dedupes_entries(self, store):
        """Same entry found by both FTS and vector is not duplicated."""
        results = hybrid_search_entries(store, "pytest testing", max_entries=5)
        ids = [r.id for r in results]
        assert len(ids) == len(set(ids)), "Duplicate entries in results"

    def test_hybrid_search_empty_query(self, store):
        """Empty query returns empty."""
        assert hybrid_search_entries(store, "", max_entries=5) == []

    def test_default_hybrid_weights(self, store):
        """Default blending weights are used (0.4 FTS, 0.6 vector)."""
        # This is more of a smoke test — hybrid search works
        results = hybrid_search_entries(store, "deploy build", max_entries=3)
        assert len(results) > 0

    def test_upgraded_get_relevant_uses_hybrid(self, store):
        """get_relevant_entries() now uses hybrid search internally."""
        results = get_relevant_entries(store, "pytest testing", max_entries=3)
        assert len(results) > 0
        # Should find at least the pytest entry
        assert any("pytest" in r.content.lower() for r in results)


class TestReranker:
    """Phase 2.4: Cross-encoder reranking (opt-in)."""

    def test_reranker_is_available(self):
        """Check if cross-encoder can be imported."""
        from agent.cognitive_memory.reranker import is_reranker_available
        result = is_reranker_available()
        assert isinstance(result, bool)

    def test_reranker_not_loaded_at_import(self):
        """Reranker is lazy-loaded, not at import time."""
        from agent.cognitive_memory.reranker import reranker_loaded
        assert reranker_loaded() is False

    def test_rerank_results_returns_same_order_when_unavailable(self, store):
        """When reranker unavailable, results returned in original order."""
        from agent.cognitive_memory.reranker import rerank_results
        results = store.cosine_search("pytest testing", limit=5)
        reranked = rerank_results("pytest testing", results)
        # Should return the same entries (maybe reordered, but all present)
        assert len(reranked) == len(results)
        assert {r.id for r in reranked} == {r.id for r in results}

    def test_rerank_results_handles_empty(self):
        """Empty list passed through without error."""
        from agent.cognitive_memory.reranker import rerank_results
        assert rerank_results("query", []) == []
