"""
Memory relation graph — typed, weighted connections between memories.

Six relation types:
  - relates_to: generic association
  - depends_on: prerequisite (B needs A)
  - contradicts: conflicting information
  - extends: elaboration/supplement
  - implements: concrete realization of an abstract memory
  - derived_from: parent → child derivation (consolidation, patterns)

Relations have evolving weights (0-1) that go up with usage. No decay —
connections don't fade, they only strengthen or are deleted by explicit action.

Storage: memory_relations table in the existing CognitiveMemoryStore.
"""

import time
import json
from dataclasses import dataclass
from enum import Enum
from typing import Optional
from agent.cognitive_memory.types import MemoryEntry
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.embeddings import cosine_similarity


class RelationType(Enum):
    RELATES_TO = "relates_to"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    EXTENDS = "extends"
    IMPLEMENTS = "implements"
    DERIVED_FROM = "derived_from"


@dataclass
class Relation:
    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 0.5
    created_at: float = 0.0
    updated_at: float = 0.0


def _ensure_relations_table(store: CognitiveMemoryStore) -> None:
    """Create the relations table if it doesn't exist. Idempotent."""
    store._conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_relations (
            source_id TEXT NOT NULL,
            target_id TEXT NOT NULL,
            relation_type TEXT NOT NULL,
            weight REAL NOT NULL DEFAULT 0.5,
            created_at REAL NOT NULL,
            updated_at REAL NOT NULL,
            PRIMARY KEY (source_id, target_id, relation_type),
            FOREIGN KEY (source_id) REFERENCES memories(id) ON DELETE CASCADE,
            FOREIGN KEY (target_id) REFERENCES memories(id) ON DELETE CASCADE
        )
    """)
    store._conn.commit()


def create_relation(
    store: CognitiveMemoryStore,
    source_id: str,
    target_id: str,
    relation_type: RelationType,
    weight: float = 0.5,
) -> Optional[Relation]:
    """Create a relation between two memories."""
    _ensure_relations_table(store)
    now = time.time()

    rel = Relation(
        source_id=source_id,
        target_id=target_id,
        relation_type=relation_type,
        weight=weight,
        created_at=now,
        updated_at=now,
    )

    store._conn.execute(
        """INSERT OR REPLACE INTO memory_relations
           (source_id, target_id, relation_type, weight, created_at, updated_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (source_id, target_id, relation_type.value, weight, now, now),
    )
    store._conn.commit()
    return rel


def get_relations(store: CognitiveMemoryStore, entry_id: str) -> list[Relation]:
    """Get all relations for a memory (both directions)."""
    _ensure_relations_table(store)
    rows = store._conn.execute(
        """SELECT * FROM memory_relations
           WHERE source_id = ? OR target_id = ?
           ORDER BY weight DESC""",
        (entry_id, entry_id),
    ).fetchall()

    return [
        Relation(
            source_id=r["source_id"],
            target_id=r["target_id"],
            relation_type=RelationType(r["relation_type"]),
            weight=r["weight"],
            created_at=r["created_at"],
            updated_at=r["updated_at"],
        )
        for r in rows
    ]


def get_graph(
    store: CognitiveMemoryStore,
    start_id: str,
    depth: int = 3,
    max_nodes: int = 50,
) -> dict[str, list[Relation]]:
    """BFS traversal of the relation graph from start_id.

    Returns dict mapping entry ID → list of Relation objects.
    """
    _ensure_relations_table(store)
    visited: set[str] = set()
    result: dict[str, list[Relation]] = {}
    frontier = {start_id}

    for _ in range(depth):
        if not frontier or len(visited) >= max_nodes:
            break
        next_frontier: set[str] = set()
        for node_id in list(frontier)[:max_nodes - len(visited)]:
            if node_id in visited:
                continue
            visited.add(node_id)
            rels = get_relations(store, node_id)
            result[node_id] = rels
            for rel in rels:
                neighbor = rel.target_id if rel.source_id == node_id else rel.source_id
                if neighbor not in visited:
                    next_frontier.add(neighbor)
        frontier = next_frontier

    return result


def delete_relation(
    store: CognitiveMemoryStore,
    source_id: str,
    target_id: str,
    relation_type: RelationType,
) -> bool:
    """Delete a specific relation."""
    _ensure_relations_table(store)
    cursor = store._conn.execute(
        """DELETE FROM memory_relations
           WHERE source_id = ? AND target_id = ? AND relation_type = ?""",
        (source_id, target_id, relation_type.value),
    )
    store._conn.commit()
    return cursor.rowcount > 0


def auto_relate_on_insert(store: CognitiveMemoryStore, entry: MemoryEntry) -> int:
    """Auto-create relations for a newly inserted entry.

    Finds related existing memories by embedding similarity and creates
    up to 5 auto-relations with appropriate types.

    Returns number of relations created.
    """
    _ensure_relations_table(store)

    similar = store.cosine_search(entry.title + " " + entry.content, limit=10)
    if not similar:
        return 0

    count = 0
    for other in similar[:5]:
        score = getattr(other, "_retrieval_score", 0.0)

        rel_type = RelationType.RELATES_TO
        if score > 0.95:
            rel_type = RelationType.CONTRADICTS
        elif score > 0.7:
            rel_type = RelationType.EXTENDS
        elif score > 0.5:
            rel_type = RelationType.DERIVED_FROM

        weight = min(score, 1.0)
        create_relation(store, entry.id, other.id, rel_type, weight=weight)
        count += 1

    return count


def sweep_underconnected(
    store: CognitiveMemoryStore,
    time_budget_ms: int = 500,
) -> int:
    """Periodic sweep: find memories with few relations and create connections.

    Time-budgeted to avoid blocking.
    Returns number of new relations created.
    """
    _ensure_relations_table(store)

    start = time.time()
    budget = time_budget_ms / 1000.0
    count = 0

    # Find entries with 0-2 relations
    rows = store._conn.execute(
        """SELECT m.id FROM memories m
           LEFT JOIN (
               SELECT source_id as id FROM memory_relations
               UNION SELECT target_id as id FROM memory_relations
           ) rels ON m.id = rels.id
           GROUP BY m.id
           HAVING COUNT(rels.id) <= 2
           LIMIT 20"""
    ).fetchall()

    for row in rows:
        if time.time() - start > budget:
            break

        entry = store.get_by_id(row["id"])
        if not entry:
            continue

        similar = store.cosine_search(entry.title + " " + entry.content, limit=3)
        for other in similar:
            if time.time() - start > budget:
                break
            if other.id != entry.id:
                create_relation(store, entry.id, other.id, RelationType.RELATES_TO, weight=0.3)
                count += 1

    return count
