"""
Cross-encoder reranking for memory search results (opt-in).

Uses sentence-transformers cross-encoder model for precision reranking
of candidate search results. Model: ms-marco-TinyBERT-L-2-v2

Design:
  - Lazy loading: model loaded only when rerank_results() is called
  - Opt-in: disabled by default, requires explicit enablement
  - Falls back to original ordering when model unavailable
  - Adds ~200ms latency but significantly improves relevance ordering
"""

from typing import Optional

# Lazy globals
_reranker_model: Optional[object] = None
_reranker_name: str = "cross-encoder/ms-marco-TinyBERT-L-2-v2"


def is_reranker_available() -> bool:
    """Check if sentence-transformers can load the cross-encoder."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def reranker_loaded() -> bool:
    """Check if the cross-encoder model is in memory."""
    global _reranker_model
    return _reranker_model is not None


def _get_reranker():
    """Lazily load the cross-encoder model."""
    global _reranker_model
    if _reranker_model is not None:
        return _reranker_model

    try:
        from sentence_transformers import CrossEncoder
        _reranker_model = CrossEncoder(_reranker_name)
        return _reranker_model
    except ImportError:
        return None
    except Exception:
        return None


def rerank_results(
    query: str,
    candidates: list,
    max_to_rerank: int = 20,
) -> list:
    """Rerank search results using cross-encoder relevance scoring.

    Args:
        query: The original search query.
        candidates: List of MemoryEntry objects to rerank.
        max_to_rerank: Maximum candidates to pass to the model (reduces latency).

    Returns:
        Reranked list of candidates sorted by relevance, or original
        list unchanged if reranker is unavailable.
    """
    if not candidates or not is_reranker_available():
        return candidates

    model = _get_reranker()
    if model is None:
        return candidates

    # Limit candidates for performance
    pool = candidates[:max_to_rerank]

    try:
        # Build (query, content) pairs for the cross-encoder
        pairs = [(query, c.title + "\n" + c.content) for c in pool]

        scores = model.predict(pairs, show_progress_bar=False)

        # Attach scores
        for i, candidate in enumerate(pool):
            score = float(scores[i]) if hasattr(scores, '__iter__') else float(scores)
            candidate._rerank_score = score  # type: ignore[attr-defined]

        # Sort by score descending
        pool.sort(key=lambda c: c._rerank_score, reverse=True)  # type: ignore[attr-defined]

        # Append any remaining candidates beyond max_to_rerank
        rest = candidates[max_to_rerank:]
        return pool + rest

    except Exception:
        return candidates
