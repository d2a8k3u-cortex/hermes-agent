"""
Type-aware retrieval — query-driven scoring with text match, recency, and
access frequency. No static importance scores — relevance is computed
dynamically at query time.

Scoring formula:
    score = text_score × w_text + recency_score × w_rec + access_score × w_access

All weights are configurable but default to text-heavy (0.5) since text relevance
is the strongest signal for retrieval quality.

This module also provides formatting for prompt injection — rendering retrieved
entries as compact context blocks suitable for the agent's user message.
"""

import time
from typing import Optional
from agent.cognitive_memory.types import MemoryType, MemoryEntry
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.embeddings import is_available as _emb_available

# ── Scoring weights ─────────────────────────────────────────────────────────
# These are tunable. Text dominates because keyword/embedding match is the
# strongest relevance signal. Recency keeps things fresh. Access frequency
# surfaces frequently-used knowledge.

DEFAULT_WEIGHTS = {
    "text": 0.5,
    "recency": 0.3,
    "access": 0.2,
}

# Hybrid: 0.4 FTS, 0.6 vector — vector is richer but needs both
HYBRID_FTS_WEIGHT = 0.4
HYBRID_VECTOR_WEIGHT = 0.6

# ── Retrieval ───────────────────────────────────────────────────────────────

def get_relevant_entries(
    store: CognitiveMemoryStore,
    query: str,
    max_entries: int = 10,
    type_filter: Optional[MemoryType] = None,
    weights: Optional[dict] = None,
) -> list[MemoryEntry]:
    """Retrieve memory entries relevant to the query.

    Uses hybrid search (FTS + embeddings) when embeddings are available,
    falling back to FTS-only mode. Results are sorted by blended score.

    Each returned entry has a ``_retrieval_score`` attribute (float) set
    for downstream consumers (e.g., to log or display relevance).

    Args:
        store: The cognitive memory store to query.
        query: Search query string. Empty/whitespace returns [].
        max_entries: Maximum results to return (default 10).
        type_filter: Optional MemoryType to restrict results.
        weights: Optional score blending weights dict (keys: text, recency, access).
    """
    if _emb_available():
        return hybrid_search_entries(
            store, query, max_entries=max_entries, type_filter=type_filter, weights=weights
        )
    # Fallback to FTS-only
    query = query.strip()
    if not query:
        return []

    w = weights or DEFAULT_WEIGHTS

    candidates = store.search_fts(query, type_filter=type_filter, limit=100)
    if not candidates:
        return []

    return _score_and_sort(candidates, store, w)[:max_entries]


def hybrid_search_entries(
    store: CognitiveMemoryStore,
    query: str,
    max_entries: int = 10,
    type_filter: Optional[MemoryType] = None,
    weights: Optional[dict] = None,
) -> list[MemoryEntry]:
    """Hybrid search: blend FTS keyword matches with embedding vector similarity.

    Fetches candidates from both FTS and cosine similarity search, merges
    with deduplication, and blends scores: 0.4 FTS rank + 0.6 cosine similarity.
    When embeddings are unavailable, falls back to FTS-only.

    Args:
        store: The cognitive memory store to query.
        query: Search query string.
        max_entries: Maximum results.
        type_filter: Optional MemoryType filter.
        weights: Optional (text, recency, access) weights for FTS scoring.

    Returns:
        List of MemoryEntry with _retrieval_score set.
    """
    query = query.strip()
    if not query:
        return []

    w = weights or DEFAULT_WEIGHTS

    # Fetch from both channels
    fts_results = store.search_fts(query, type_filter=type_filter, limit=50)
    vec_results = store.cosine_search(query, limit=50, type_filter=type_filter)

    # If no vector results, pure FTS
    if not vec_results:
        if not fts_results:
            return []
        return _score_and_sort(fts_results, store, w)[:max_entries]

    # Build lookup by ID
    by_id: dict[str, tuple[MemoryEntry, float, float]] = {}
    # (entry, fts_score, vec_score)

    # Score FTS results by rank
    for i, entry in enumerate(fts_results):
        fts_score = 1.0 / (1.0 + i * 0.1)
        by_id[entry.id] = (entry, fts_score, 0.0)

    # Score vector results
    for entry in vec_results:
        vec_score = getattr(entry, "_retrieval_score", 0.5)
        if entry.id in by_id:
            existing_entry, fts_score, _ = by_id[entry.id]
            by_id[entry.id] = (existing_entry, fts_score, vec_score)
        else:
            by_id[entry.id] = (entry, 0.0, vec_score)

    # Blend: 0.4 * fts_score + 0.6 * vec_score
    scored = []
    for entry, fts, vec in by_id.values():
        blended = fts * HYBRID_FTS_WEIGHT + vec * HYBRID_VECTOR_WEIGHT

        # Add recency + access boosts
        now = time.time()
        age_days = (now - entry.created_at) / 86400.0
        recency_score = 1.0 / (1.0 + age_days * 0.1)
        access_score = min(entry.access_count / 10.0, 1.0)

        final = blended + recency_score * 0.15 + access_score * 0.05

        entry.register_access()
        store._update_metadata_only(entry)

        entry._retrieval_score = final  # type: ignore[attr-defined]
        scored.append(entry)

    scored.sort(key=lambda e: e._retrieval_score, reverse=True)  # type: ignore[attr-defined]
    return scored[:max_entries]


def _score_and_sort(
    candidates: list[MemoryEntry],
    store: CognitiveMemoryStore,
    weights: dict,
) -> list[MemoryEntry]:
    """Score FTS candidates with recency/access blend and sort."""
    now = time.time()
    max_access = max((e.access_count for e in candidates), default=1)

    scored = []
    for i, entry in enumerate(candidates):
        text_score = 1.0 / (1.0 + i * 0.15)
        age_days = (now - entry.created_at) / 86400.0
        recency_score = 1.0 / (1.0 + age_days * 0.1)
        access_score = min(entry.access_count / max(max_access, 1), 1.0)

        score = (
            text_score * weights["text"]
            + recency_score * weights["recency"]
            + access_score * weights["access"]
        )

        entry.register_access()
        store._update_metadata_only(entry)
        entry._retrieval_score = score  # type: ignore[attr-defined]
        scored.append(entry)

    scored.sort(key=lambda e: e._retrieval_score, reverse=True)  # type: ignore[attr-defined]
    return scored


# ── Prompt formatting ───────────────────────────────────────────────────────

def get_context_for_prompt(entries: list[MemoryEntry]) -> str:
    """Format retrieved entries as a compact context block for prompt injection.

    Follows the format convention established by claude-code-memory:
    - Each entry shows type marker, title, and truncated content.
    - Confidence is shown when < 1.0 (pattern entries from conversation extraction).
    - Empty list returns empty string — no injection needed.

    Args:
        entries: Retrieved MemoryEntry objects (with optional _retrieval_score).

    Returns:
        Formatted string suitable for injecting into a user message, or "".
    """
    if not entries:
        return ""

    lines = ["═══ BUILTIN MEMORY (relevant entries) ═══", ""]

    for entry in entries:
        type_marker = _type_marker(entry.type)
        # Truncate content to ~180 chars for prompt efficiency
        content = entry.content
        if len(content) > 180:
            content = content[:177] + "..."

        line = f"{type_marker} {entry.title} — {content}"

        # Show confidence for non-1.0 entries (pattern memories from extraction)
        if entry.confidence < 1.0:
            line += f"  [confidence: {entry.confidence:.0%}]"

        lines.append(line)

    return "\n".join(lines)


def _type_marker(mem_type: MemoryType) -> str:
    """Return a compact type marker for display."""
    markers = {
        MemoryType.SEMANTIC: "[semantic]",
        MemoryType.PROCEDURAL: "[procedural]",
        MemoryType.PATTERN: "[pattern]",
    }
    return markers.get(mem_type, "[?]")
