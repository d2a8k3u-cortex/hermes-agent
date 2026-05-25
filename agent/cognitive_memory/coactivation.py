"""
Co-activation tracking — discovers emergent connections between memories.

When two memories are injected together in the same session, their
co-activation count increments. At count ≥ 3, a relates_to relation
is automatically created. This discovers connections that embedding
similarity misses — e.g., "pytest" and "coverage" injected together
repeatedly are clearly related even if their text/embeddings differ.

No decay: connections discovered through co-activation are real and
permanent. They only strengthen with usage.
"""

from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.relations import create_relation, RelationType

PROMOTION_THRESHOLD = 3


class CoActivationTracker:
    """Tracks which memory pairs have been injected together in a session.

    Stores counts in memory only (not persisted to DB). Counts are
    ephemeral per session — the promotion to a relation is the
    persistent artifact.
    """

    def __init__(self):
        self._counts: dict[tuple[str, str], int] = {}

    def increment(self, id_a: str, id_b: str) -> int:
        """Increment the co-activation count for a pair. Order-independent."""
        if id_a == id_b:
            return 0
        # Explicit cast to help type checker
        key: tuple[str, str] = (id_a, id_b) if id_a < id_b else (id_b, id_a)
        self._counts[key] = self._counts.get(key, 0) + 1
        return self._counts[key]

    def get_count(self, id_a: str, id_b: str) -> int:
        """Get the current co-activation count for a pair."""
        key: tuple[str, str] = (id_a, id_b) if id_a < id_b else (id_b, id_a)
        return self._counts.get(key, 0)

    def clear(self) -> None:
        """Reset all counts (called at session start)."""
        self._counts.clear()

    @property
    def total_pairs(self) -> int:
        return len(self._counts)


def check_and_promote(
    store: CognitiveMemoryStore,
    tracker: CoActivationTracker,
    id_a: str,
    id_b: str,
) -> bool:
    """Check if a pair has reached the promotion threshold and create a relation.

    Called after each increment. At count == PROMOTION_THRESHOLD, creates
    a relates_to relation with initial weight 0.45.

    Returns True if a relation was created.
    """
    count = tracker.get_count(id_a, id_b)
    if count < PROMOTION_THRESHOLD:
        return False

    # Only create on exact threshold hit (not every time after)
    if count != PROMOTION_THRESHOLD:
        return False

    rel = create_relation(
        store, id_a, id_b,
        RelationType.RELATES_TO,
        weight=0.45,
    )
    return rel is not None
