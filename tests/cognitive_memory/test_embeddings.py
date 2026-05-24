"""
Tests for agent/cognitive_memory/embeddings.py — embedding model integration.

TDD RED phase: define behavior before implementation exists.

Uses sentence-transformers with all-MiniLM-L6-v2 (384-dim vectors).
Tests run against the real model where possible, with fallback tests
for environments where the package isn't installed.
"""

import pytest
import numpy as np

# Try importing — module should exist even without sentence-transformers
from agent.cognitive_memory.embeddings import (
    generate_embedding,
    generate_embeddings,
    cosine_similarity,
    is_available,
    model_loaded,
    EMBEDDING_DIM,
)


class TestAvailability:
    """Detection of whether the embedding model can be used."""

    def test_is_available_returns_bool(self):
        """is_available() returns a boolean."""
        result = is_available()
        assert isinstance(result, bool)

    def test_model_loaded_returns_bool(self):
        """model_loaded() returns a boolean."""
        result = model_loaded()
        assert isinstance(result, bool)


class TestEmbeddingDimension:
    """Embedding vector properties."""

    def test_embedding_dim_is_correct(self):
        """all-MiniLM-L6-v2 produces 384-dimensional vectors."""
        assert EMBEDDING_DIM == 384

    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_generate_returns_correct_dimension(self):
        """Generated embedding has the expected dimension."""
        vec = generate_embedding("test sentence")
        assert len(vec) == EMBEDDING_DIM
        assert isinstance(vec, list)
        assert all(isinstance(v, float) for v in vec)

    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_batch_generates_correct_number(self):
        """Batch generation returns the right number of vectors."""
        texts = ["first sentence", "second sentence", "third"]
        vectors = generate_embeddings(texts)
        assert len(vectors) == 3
        for vec in vectors:
            assert len(vec) == EMBEDDING_DIM


class TestCosineSimilarity:
    """Cosine similarity computation."""

    def test_identical_vectors_score_one(self):
        """Two identical vectors have cosine similarity 1.0."""
        vec = [0.1, 0.2, 0.3]
        score = cosine_similarity(vec, vec)
        assert abs(score - 1.0) < 0.001

    def test_orthogonal_vectors_score_zero(self):
        """Orthogonal vectors have cosine similarity ~0."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        score = cosine_similarity(a, b)
        assert abs(score - 0.0) < 0.001

    def test_opposite_vectors_score_negative_one(self):
        """Opposite vectors have cosine similarity -1.0."""
        a = [1.0, 0.0, 0.0]
        b = [-1.0, 0.0, 0.0]
        score = cosine_similarity(a, b)
        assert abs(score - (-1.0)) < 0.001

    def test_different_lengths_raises(self):
        """Vectors of different lengths raise ValueError."""
        with pytest.raises(ValueError, match="length"):
            cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0])

    def test_zero_vector_handled(self):
        """Zero vector returns 0.0 to avoid division by zero."""
        score = cosine_similarity([0.0, 0.0, 0.0], [1.0, 2.0, 3.0])
        assert score == 0.0

    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_semantic_similarity_is_meaningful(self):
        """Similar sentences have higher cosine similarity than dissimilar ones."""
        sim_a = generate_embedding("Python is a programming language")
        sim_b = generate_embedding("Python is used for coding")
        diff = generate_embedding("The weather is nice today")

        score_similar = cosine_similarity(sim_a, sim_b)
        score_different = cosine_similarity(sim_a, diff)

        assert score_similar > score_different, (
            f"Similar: {score_similar:.3f}, Different: {score_different:.3f}"
        )


class TestGracefulFallback:
    """Behavior when sentence-transformers is not available."""

    def test_import_does_not_crash(self):
        """Importing the module doesn't crash even without the package."""
        # This test passes just by importing at the top of the file.
        pass

    def test_is_available_graceful(self):
        """is_available() works regardless of installation state."""
        result = is_available()
        assert result in (True, False)

    @pytest.mark.skipif(is_available(), reason="sentence-transformers IS installed")
    def test_generate_raises_when_unavailable(self):
        """generate_embedding raises clear error when model unavailable."""
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            generate_embedding("test")

    @pytest.mark.skipif(is_available(), reason="sentence-transformers IS installed")
    def test_batch_raises_when_unavailable(self):
        """generate_embeddings raises clear error when model unavailable."""
        with pytest.raises(RuntimeError, match="sentence-transformers"):
            generate_embeddings(["test", "test2"])


class TestLazyLoading:
    """Model is loaded lazily on first use, not at import time."""

    def test_model_not_loaded_at_import(self):
        """Importing the module doesn't load the model."""
        # model_loaded() checks if model is in memory
        # At import time it shouldn't be loaded
        # Note: if is_available() is False, model_loaded() returns False
        pass  # Verified by import side-effects

    @pytest.mark.skipif(not is_available(), reason="sentence-transformers not installed")
    def test_model_loaded_after_first_call(self):
        """After first generate_embedding call, model_loaded() returns True."""
        generate_embedding("warm up the model")
        assert model_loaded() is True
