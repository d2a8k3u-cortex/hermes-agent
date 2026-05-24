"""
Ambient memory injection — pre-turn context surfacing.

This module handles selecting which memories to inject into the agent's
conversation context before each API call. It replaces the previous
"dump all memory into system prompt" approach with per-turn relevance
injection.

Core components:
  - PerTurnDedupTracker: prevents re-injecting the same memory twice in a session
  - content_length_penalty: penalizes verbose entries in relevance scoring
  - InjectionConfig: tunable limits per type
  - inject_memory_context: the main entry point for pre-turn injection
"""

from dataclasses import dataclass, field
from typing import Optional
from agent.cognitive_memory.types import MemoryType, MemoryEntry
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.retrieval import (
    get_relevant_entries,
    get_context_for_prompt,
)


@dataclass(frozen=True)
class InjectionConfig:
    """Configuration for ambient memory injection.

    Tunable limits prevent overwhelming the model with too much context.
    """
    max_semantic_per_turn: int = 3
    max_procedural_per_turn: int = 2
    max_pattern_per_turn: int = 2
    max_total_per_turn: int = 6
    min_confidence: float = 0.3


DEFAULT_INJECTION_CONFIG = InjectionConfig()


class PerTurnDedupTracker:
    """Tracks which memory IDs have been injected this session.

    Prevents the same memory from being injected repeatedly,
    which would waste context and annoy the model.

    Capped at max_injected to prevent unbounded growth.
    """

    def __init__(self, max_injected: int = 200):
        self.injected_ids: set[str] = set()
        self._max_injected = max_injected

    def is_injected(self, memory_id: str) -> bool:
        """Check if a memory has already been injected this session."""
        return memory_id in self.injected_ids

    def mark_injected(self, memory_id: str) -> None:
        """Record that a memory was injected."""
        if len(self.injected_ids) >= self._max_injected:
            # FIFO eviction: remove oldest half
            to_remove = list(self.injected_ids)[:self._max_injected // 2]
            self.injected_ids.difference_update(to_remove)
        self.injected_ids.add(memory_id)

    def clear(self) -> None:
        """Reset tracker (called at session start)."""
        self.injected_ids.clear()

    def __len__(self) -> int:
        return len(self.injected_ids)


def content_length_penalty(content: str, threshold: int = 500, max_penalty: float = 0.15) -> float:
    """Compute a penalty for long content to prevent verbosity bias.

    Progressive penalty starting at `threshold` chars, asymptotically
    approaching `max_penalty`. Short content (< threshold) gets 0 penalty.

    Args:
        content: The memory content string.
        threshold: Characters before penalty starts.
        max_penalty: Maximum penalty (asymptote).

    Returns:
        Penalty value in [0.0, max_penalty].
    """
    length = len(content)
    if length <= threshold:
        return 0.0
    # Asymptotic: penalty = max_penalty * (1 - threshold / length)
    return max_penalty * (1.0 - threshold / length)


def inject_memory_context(
    store: CognitiveMemoryStore,
    query: str,
    dedup_tracker: PerTurnDedupTracker,
    config: InjectionConfig = DEFAULT_INJECTION_CONFIG,
) -> str:
    """Select and format memory entries for pre-turn context injection.

    Queries the cognitive store for entries relevant to the user's message,
    filters by confidence and dedup, enforces per-type caps, and formats
    the results for injection into the agent's conversation context.

    Args:
        store: The cognitive memory store to query.
        query: The user's message or query string.
        dedup_tracker: Session-level tracker to prevent re-injection.
        config: Injection limits and thresholds.

    Returns:
        Formatted context block string suitable for user-message injection,
        or empty string if nothing relevant was found.
    """
    query = query.strip()
    if not query:
        return ""

    all_results = get_relevant_entries(store, query, max_entries=config.max_total_per_turn * 3)

    if not all_results:
        return ""

    # Filter and cap
    by_type: dict[MemoryType, list[MemoryEntry]] = {
        MemoryType.SEMANTIC: [],
        MemoryType.PROCEDURAL: [],
        MemoryType.PATTERN: [],
    }

    type_caps: dict[MemoryType, int] = {
        MemoryType.SEMANTIC: config.max_semantic_per_turn,
        MemoryType.PROCEDURAL: config.max_procedural_per_turn,
        MemoryType.PATTERN: config.max_pattern_per_turn,
    }

    selected: list[MemoryEntry] = []
    seen: set[str] = set()

    for entry in all_results:
        if len(selected) >= config.max_total_per_turn:
            break

        # Skip if already injected this session
        if dedup_tracker.is_injected(entry.id):
            continue

        # Skip low-confidence entries
        if entry.confidence < config.min_confidence:
            continue

        # Skip if cap for this type is reached
        if len(by_type[entry.type]) >= type_caps[entry.type]:
            continue

        # Apply content-length penalty: deprioritize but don't exclude
        penalty = content_length_penalty(entry.content)
        adjusted_score = getattr(entry, "_retrieval_score", 0.5) - penalty

        if adjusted_score <= 0.0 and len(selected) > 0:
            continue  # Too verbose for too little relevance

        if entry.id not in seen:
            selected.append(entry)
            by_type[entry.type].append(entry)
            dedup_tracker.mark_injected(entry.id)
            seen.add(entry.id)

    return get_context_for_prompt(selected)
