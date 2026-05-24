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

# ── Scoring weights ─────────────────────────────────────────────────────────
# These are tunable. Text dominates because keyword/embedding match is the
# strongest relevance signal. Recency keeps things fresh. Access frequency
# surfaces frequently-used knowledge.

DEFAULT_WEIGHTS = {
    "text": 0.5,
    "recency": 0.3,
    "access": 0.2,
}

# ── Retrieval ───────────────────────────────────────────────────────────────

def get_relevant_entries(
    store: CognitiveMemoryStore,
    query: str,
    max_entries: int = 10,
    type_filter: Optional[MemoryType] = None,
    weights: Optional[dict] = None,
) -> list[MemoryEntry]:
    """Retrieve memory entries relevant to the query.

    Uses FTS5 for text matching, then blends with recency and access
    frequency scores. Results are sorted by blended score descending.

    Each returned entry has a ``_retrieval_score`` attribute (float) set
    for downstream consumers (e.g., to log or display relevance).

    Args:
        store: The cognitive memory store to query.
        query: Search query string. Empty/whitespace returns [].
        max_entries: Maximum results to return (default 10).
        type_filter: Optional MemoryType to restrict results.
        weights: Optional score blending weights dict (keys: text, recency, access).
    """
    query = query.strip()
    if not query:
        return []

    w = weights or DEFAULT_WEIGHTS

    # Get all FTS matches (large limit — we'll re-rank and trim)
    candidates = store.search_fts(query, type_filter=type_filter, limit=100)

    if not candidates:
        return []

    now = time.time()
    max_access = max((e.access_count for e in candidates), default=1)

    scored = []
    for i, entry in enumerate(candidates):
        # Text score: best FTS match (position 0) = 1.0, decays by rank
        text_score = 1.0 / (1.0 + i * 0.15)

        # Recency score: newer entries score higher
        age_days = (now - entry.created_at) / 86400.0
        recency_score = 1.0 / (1.0 + age_days * 0.1)

        # Access score: normalized by max access count in this result set
        access_score = min(entry.access_count / max(max_access, 1), 1.0)

        # Blend
        score = (
            text_score * w["text"]
            + recency_score * w["recency"]
            + access_score * w["access"]
        )

        # Register access and persist
        entry.register_access()
        store.update(entry)

        # Attach score for downstream use
        entry._retrieval_score = score  # type: ignore[attr-defined]
        scored.append(entry)

    # Sort by blended score descending
    scored.sort(key=lambda e: e._retrieval_score, reverse=True)  # type: ignore[attr-defined]

    return scored[:max_entries]


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
