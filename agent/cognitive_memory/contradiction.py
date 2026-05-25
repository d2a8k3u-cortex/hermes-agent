"""
Contradiction detection and resolution.

When a new memory contradicts an existing one, the system flags the conflict
rather than silently overwriting. The user resolves which one is correct.

Core rules:
  - High-similarity entries with opposing content → contradict
  - Older entry gets marked with contradicted_by pointing to the new one
  - Confidence of contradicted entry drops to 0.3
  - Never auto-resolve — always surface to user
  - Resolution by user: confirmed → 1.0, rejected → 0.0
"""

from dataclasses import dataclass
from typing import Optional
from agent.cognitive_memory.types import MemoryType, MemoryEntry
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.embeddings import cosine_similarity

CONTRADICTION_SIMILARITY_THRESHOLD = 0.85  # Very high similarity needed
CONTRADICTED_CONFIDENCE = 0.3


@dataclass
class ContradictionResult:
    """Result of a contradiction check."""
    detected: bool = False
    existing_entry_id: Optional[str] = None
    conflicting_entry_id: Optional[str] = None
    similarity: float = 0.0


def detect_contradiction(
    store: CognitiveMemoryStore,
    new_entry: MemoryEntry,
) -> Optional[ContradictionResult]:
    """Check if a new entry contradicts any existing entry of the same type.

    Only flags when similarity is very high (>0.85) but content suggests
    the same topic with different information — e.g., "uses Make" vs "uses Ninja".

    Args:
        store: The cognitive memory store.
        new_entry: The newly inserted or updated entry.

    Returns:
        ContradictionResult if a contradiction was detected, None otherwise.
    """
    same_type = store.list_by_type(new_entry.type, limit=100)
    if not same_type:
        return None

    try:
        new_blob = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (new_entry.id,)
        ).fetchone()
        if not new_blob or new_blob["embedding"] is None:
            return None
        new_vec = store._blob_to_vector(new_blob["embedding"])
    except Exception:
        return None

    best_match: Optional[MemoryEntry] = None
    best_sim = 0.0

    for existing in same_type:
        if existing.id == new_entry.id:
            continue

        try:
            ex_blob = store._conn.execute(
                "SELECT embedding FROM memories WHERE id = ?", (existing.id,)
            ).fetchone()
            if not ex_blob or ex_blob["embedding"] is None:
                continue
            ex_vec = store._blob_to_vector(ex_blob["embedding"])
            sim = cosine_similarity(new_vec, ex_vec)

            if sim > CONTRADICTION_SIMILARITY_THRESHOLD and sim > best_sim:
                # Verify they're actually contradictory (not just similar)
                if _are_contradictory(new_entry, existing):
                    best_sim = sim
                    best_match = existing
        except Exception:
            continue

    if best_match:
        # Mark the older entry as contradicted
        best_match.mark_contradicted(new_entry.id)
        store.update(best_match)

        return ContradictionResult(
            detected=True,
            existing_entry_id=best_match.id,
            conflicting_entry_id=new_entry.id,
            similarity=best_sim,
        )

    return None


def _are_contradictory(a: MemoryEntry, b: MemoryEntry) -> bool:
    """Quick heuristic: similar topic but different key terms."""
    words_a = set(a.content.lower().split())
    words_b = set(b.content.lower().split())
    common = words_a & words_b
    diff_a = words_a - words_b
    diff_b = words_b - words_a

    # Contradiction if: significant overlap AND significant differences
    overlap_ratio = len(common) / max(len(words_a | words_b), 1)
    diff_ratio = (len(diff_a) + len(diff_b)) / max(len(words_a | words_b), 1)

    return overlap_ratio > 0.3 and diff_ratio > 0.2


def resolve_contradiction(
    store: CognitiveMemoryStore,
    old_id: str,
    new_id: str,
    keep_new: bool = True,
) -> None:
    """Resolve a detected contradiction — user has decided which entry is correct.

    Args:
        store: The cognitive memory store.
        old_id: ID of the older (contradicted) entry.
        new_id: ID of the newer entry that caused the contradiction.
        keep_new: If True, new entry stays (old remains contradicted).
                  If False, new entry is rejected (confidence → 0.0).
    """
    old_entry = store.get_by_id(old_id)
    new_entry = store.get_by_id(new_id)

    if keep_new:
        # Old entry stays contradicted, new entry confidence → 1.0
        if new_entry:
            new_entry.confidence = 1.0
            store.update(new_entry)
    else:
        # New entry was wrong, reject it
        if new_entry:
            new_entry.confidence = 0.0
            store.update(new_entry)
        # Restore old entry if it was marked
        if old_entry:
            old_entry.contradicted_by = None
            old_entry.confidence = 1.0
            store.update(old_entry)
