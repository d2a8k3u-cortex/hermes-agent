"""
Health monitoring and diagnostics for the cognitive memory system.

Provides comprehensive metrics:
  - Counts by type, confidence distribution, embedding coverage
  - Staleness analysis, age distribution
  - Relation graph statistics
  - Health score (0-1) with recommendations
"""

import time
from agent.cognitive_memory.types import MemoryType
from agent.cognitive_memory.store import CognitiveMemoryStore


def get_health_report(store: CognitiveMemoryStore) -> dict:
    """Generate a comprehensive health report for the cognitive memory store.

    Returns:
        Dict with keys: memory_count, type_counts, embedding_coverage,
        confidence_distribution, age_distribution, relation_stats,
        health_score, recommendations.
    """
    stats = store.get_stats()

    # Embedding coverage
    total = stats["memory_count"]
    with_emb = 0
    if total > 0:
        with_emb = store._conn.execute(
            "SELECT COUNT(*) FROM memories WHERE embedding IS NOT NULL"
        ).fetchone()[0]

    embedding_coverage = (with_emb / total * 100) if total > 0 else 0.0

    # Confidence distribution
    confidence_dist = {"high": 0, "medium": 0, "low": 0}
    if total > 0:
        rows = store._conn.execute(
            "SELECT confidence FROM memories"
        ).fetchall()
        for row in rows:
            c = row["confidence"]
            if c >= 0.8:
                confidence_dist["high"] += 1
            elif c >= 0.3:
                confidence_dist["medium"] += 1
            else:
                confidence_dist["low"] += 1

    # Age distribution
    now = time.time()
    age_dist = {"day": 0, "week": 0, "month": 0, "older": 0}
    if total > 0:
        rows = store._conn.execute(
            "SELECT created_at FROM memories"
        ).fetchall()
        for row in rows:
            age_days = (now - row["created_at"]) / 86400.0
            if age_days < 1:
                age_dist["day"] += 1
            elif age_days < 7:
                age_dist["week"] += 1
            elif age_days < 30:
                age_dist["month"] += 1
            else:
                age_dist["older"] += 1

    # Relation stats
    relation_count = 0
    try:
        relation_count = store._conn.execute(
            "SELECT COUNT(*) FROM memory_relations"
        ).fetchone()[0]
    except Exception:
        pass

    # Health score
    score = 1.0
    recommendations = []

    if total == 0:
        score = 0.5
        recommendations.append("No memories stored yet. Consider running migration.")
    if embedding_coverage < 80 and total > 0:
        score -= 0.1
        recommendations.append(
            f"Low embedding coverage ({embedding_coverage:.0f}%). "
            "Install sentence-transformers for semantic search."
        )
    if confidence_dist["low"] > total * 0.2:
        score -= 0.1
        recommendations.append(
            f"{confidence_dist['low']} entries have low confidence. "
            "Consider reviewing contradicted entries."
        )
    if relation_count == 0 and total > 10:
        recommendations.append(
            "No relations found. Run compression to build relation graph."
        )

    return {
        "memory_count": total,
        "type_counts": stats["type_counts"],
        "embedding_coverage_pct": round(embedding_coverage, 1),
        "confidence_distribution": confidence_dist,
        "age_distribution": age_dist,
        "relation_count": relation_count,
        "health_score": round(max(score, 0.0), 2),
        "recommendations": recommendations,
    }
