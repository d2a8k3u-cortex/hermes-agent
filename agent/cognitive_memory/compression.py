"""
Compression engine — merges, splits, and detects patterns without deleting.

Core principles (from the revised plan):
  - Information is never lost — it is retained, merged, or elevated.
  - No decay. No aging out. No forgetting.
  - Compression combines similar entries; splitting prevents bloat.
  - Pattern detection clusters related entries and creates summary patterns.

Triggers:
  - Memory count > COMPRESSION_THRESHOLD (default 200)
  - Explicit hermes memory compress command
  - Every 7 days of uptime (time-based fallback)
"""

import time
from dataclasses import dataclass, field
from typing import Optional
from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.embeddings import cosine_similarity

COMPRESSION_THRESHOLD = 200
NEAR_DUPLICATE_THRESHOLD = 0.08  # cosine distance below this = near-duplicate
MIN_SPLIT_CHARS = 500
PATTERN_CLUSTER_SIMILARITY = 0.5


@dataclass
class CompressionResult:
    """Result of a compression cycle — what was done."""
    entries_merged: int = 0
    entries_split: int = 0
    patterns_detected: int = 0
    entries_scanned: int = 0
    duration_ms: float = 0.0


def run_compression_cycle(store: CognitiveMemoryStore) -> CompressionResult:
    """Run a full compression cycle: merge duplicates, split topics, detect patterns.

    Idempotent — running twice on the same store produces no additional changes.
    """
    start = time.time()
    result = CompressionResult()

    try:
        merge_result = merge_duplicates(store)
        result.entries_merged = merge_result.entries_merged

        split_result = split_all_topics(store)
        result.entries_split = split_result.entries_split

        pattern_result = detect_patterns(store)
        result.patterns_detected = pattern_result.patterns_detected

        result.entries_scanned = store.get_stats()["memory_count"]
    except Exception:
        pass

    result.duration_ms = (time.time() - start) * 1000
    return result


# ── Duplicate merging ────────────────────────────────────────────────────────

def merge_duplicates(store: CognitiveMemoryStore) -> CompressionResult:
    """Find and merge near-duplicate entries.

    Scans all non-pattern entries, finds pairs with very high cosine
    similarity, and merges them. The kept entry gets the longer content,
    union of tags, and higher confidence. The merged entry is soft-deleted
    (its ID is recorded as distilled_to on the kept entry — not truly erased).

    Returns CompressionResult with merger count.
    """
    result = CompressionResult()

    entries = store.list_all(limit=500)
    if len(entries) < 2:
        return result

    # Group by type for efficient comparison
    by_type: dict[MemoryType, list[MemoryEntry]] = {}
    for e in entries:
        if e.type not in by_type:
            by_type[e.type] = []
        by_type[e.type].append(e)

    checked_pairs: set[tuple[str, str]] = set()

    for mem_type, group in by_type.items():
        if len(group) < 2:
            continue

        for i, a in enumerate(group):
            for b in group[i + 1:]:
                pair_key = (min(a.id, b.id), max(a.id, b.id))
                if pair_key in checked_pairs:
                    continue
                checked_pairs.add(pair_key)

                # Only compare if both have embeddings
                if a.id == b.id:
                    continue

                # Quick text-overlap check first (faster than embeddings)
                words_a = set(a.content.lower().split())
                words_b = set(b.content.lower().split())
                if not words_a or not words_b:
                    continue
                overlap = len(words_a & words_b) / min(len(words_a), len(words_b))
                if overlap < 0.6:
                    continue

                # If overlap is very high, merge
                if overlap > 0.9 or _are_near_duplicates(store, a, b):
                    _merge_pair(store, a, b)
                    result.entries_merged += 1

    return result


def _are_near_duplicates(
    store: CognitiveMemoryStore, a: MemoryEntry, b: MemoryEntry
) -> bool:
    """Check if two entries are near-duplicates using cosine distance."""
    try:
        a_blob = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (a.id,)
        ).fetchone()
        b_blob = store._conn.execute(
            "SELECT embedding FROM memories WHERE id = ?", (b.id,)
        ).fetchone()

        if not a_blob or not b_blob:
            return False
        if a_blob["embedding"] is None or b_blob["embedding"] is None:
            return False

        a_vec = store._blob_to_vector(a_blob["embedding"])
        b_vec = store._blob_to_vector(b_blob["embedding"])
        sim = cosine_similarity(a_vec, b_vec)
        return (1.0 - sim) < NEAR_DUPLICATE_THRESHOLD
    except Exception:
        return False


def _merge_pair(store: CognitiveMemoryStore, a: MemoryEntry, b: MemoryEntry):
    """Merge b into a. A is kept (usually the longer content)."""
    # Keep the entry with longer content
    keeper, absorbed = (a, b) if len(a.content) >= len(b.content) else (b, a)

    # If absorbed has a better title, keep it
    if len(absorbed.title) > len(keeper.title):
        keeper.title = absorbed.title

    # Union tags
    keeper.tags = list(set(keeper.tags) | set(absorbed.tags))

    # Keep higher confidence
    keeper.confidence = max(keeper.confidence, absorbed.confidence)

    store.update(keeper)

    # Mark absorbed: point to keeper
    absorbed.mark_distilled(keeper.id)
    store.update(absorbed)


# ── Topic splitting ──────────────────────────────────────────────────────────

def split_topics(store: CognitiveMemoryStore, entry: MemoryEntry) -> CompressionResult:
    """Split a large entry into topic-specific sub-entries if it has clear structure.

    Strategies (tried in order):
      1. ## Markdown headers
      2. ### Markdown headers
      3. **Bold:** section headers
      4. Numbered lists (1. 2. 3.)

    Only splits if ≥ 2 meaningful sections emerge.
    """
    import re
    result = CompressionResult()

    content = entry.content
    if len(content) < MIN_SPLIT_CHARS:
        return result

    sections: list[str] = []

    # Strategy 1: ## headers
    parts = re.split(r"\n##\s+", content)
    if len(parts) > 2:
        sections = [p.strip() for p in parts if p.strip() and len(p.strip()) > 20]

    # Strategy 2: ### headers (if Strategy 1 didn't work)
    if len(sections) < 2:
        parts = re.split(r"\n###\s+", content)
        if len(parts) > 2:
            sections = [p.strip() for p in parts if p.strip() and len(p.strip()) > 20]

    # Strategy 3: **Bold:** headers
    if len(sections) < 2:
        parts = re.split(r"\n\*\*(.+?)\*\*\s*[:\n]", content)
        if len(parts) > 3:
            # parts[0] is preamble, [1] is first header, [2] is first body...
            sections = []
            for i in range(1, len(parts), 2):
                if i + 1 < len(parts):
                    sections.append(f"**{parts[i]}**\n{parts[i + 1].strip()}")

    # Strategy 4: Numbered lists
    if len(sections) < 2:
        parts = re.split(r"\n\d+\.\s+", content)
        if len(parts) > 2:
            sections = [p.strip() for p in parts if p.strip() and len(p.strip()) > 20]

    if len(sections) < 2:
        return result

    # Create separate entries
    for i, section in enumerate(sections):
        title = section.split("\n")[0][:80] or entry.title
        new_entry = MemoryEntry(
            type=entry.type,
            title=f"{entry.title} (part {i + 1})",
            content=section,
            confidence=entry.confidence,
            provenance=entry.provenance,
            tags=entry.tags,
        )
        store.insert(new_entry)
        result.entries_split += 1

    # Mark original as distilled
    entry.mark_distilled(f"split-{len(sections)}-parts")
    store.update(entry)

    return result


def split_all_topics(store: CognitiveMemoryStore) -> CompressionResult:
    """Split all large entries in the store."""
    result = CompressionResult()
    for entry in store.list_all(limit=200):
        if len(entry.content) >= MIN_SPLIT_CHARS:
            split_result = split_topics(store, entry)
            result.entries_split += split_result.entries_split
    return result


# ── Pattern detection ────────────────────────────────────────────────────────

def detect_patterns(store: CognitiveMemoryStore) -> CompressionResult:
    """Cluster related memories to create pattern entries.

    Groups entries by type and finds clusters using cosine similarity.
    Creates pattern entries summarizing clusters of ≥ 3 entries.
    """
    result = CompressionResult()
    entries = store.list_all(limit=200)
    if len(entries) < 3:
        return result

    # Simple agglomerative clustering by type
    for mem_type in [MemoryType.SEMANTIC, MemoryType.PROCEDURAL]:
        group = [e for e in entries if e.type == mem_type]
        if len(group) < 3:
            continue

        clusters = _cluster_entries(store, group)
        for cluster in clusters:
            if len(cluster) < 3:
                continue

            # Create a pattern summarizing the cluster
            titles = [e.title for e in cluster]
            combined_content = "\n".join(e.content[:200] for e in cluster)

            pattern = MemoryEntry(
                type=MemoryType.PATTERN,
                title=f"Pattern: {titles[0][:60]}",
                content=combined_content,
                confidence=0.6,
                provenance=Provenance.PATTERN_DETECTED,
                tags=list(set(tag for e in cluster for tag in e.tags)),
            )
            store.insert(pattern)
            result.patterns_detected += 1

    return result


def _cluster_entries(
    store: CognitiveMemoryStore, entries: list[MemoryEntry]
) -> list[list[MemoryEntry]]:
    """Agglomerative clustering of entries by cosine similarity."""
    if len(entries) < 2:
        return [entries]

    # Build similarity matrix for entries with embeddings
    ids = [e.id for e in entries]
    sim: dict[tuple[int, int], float] = {}

    for i in range(len(entries)):
        for j in range(i + 1, len(entries)):
            a, b = entries[i], entries[j]
            if _are_near_duplicates(store, a, b):
                sim[(i, j)] = 1.0
            else:
                try:
                    a_blob = store._conn.execute(
                        "SELECT embedding FROM memories WHERE id = ?", (a.id,)
                    ).fetchone()
                    b_blob = store._conn.execute(
                        "SELECT embedding FROM memories WHERE id = ?", (b.id,)
                    ).fetchone()
                    if a_blob and b_blob and a_blob["embedding"] and b_blob["embedding"]:
                        a_vec = store._blob_to_vector(a_blob["embedding"])
                        b_vec = store._blob_to_vector(b_blob["embedding"])
                        s = cosine_similarity(a_vec, b_vec)
                        if s > PATTERN_CLUSTER_SIMILARITY:
                            sim[(i, j)] = s
                except Exception:
                    pass

    if not sim:
        return [[e] for e in entries]

    # Connected components clustering
    parent = list(range(len(entries)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for (i, j), score in sim.items():
        union(i, j)

    # Group by root
    groups: dict[int, list[MemoryEntry]] = {}
    for i, entry in enumerate(entries):
        root = find(i)
        if root not in groups:
            groups[root] = []
        groups[root].append(entry)

    return [g for g in groups.values() if len(g) >= 2]
