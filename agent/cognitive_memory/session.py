"""
Session lifecycle hooks — ambient memory injection at session boundaries.

Session start: inject relevant context from git signals and recent activity.
Session end: run extraction pass across collected turns, store results.

This is Hermes' equivalent of claude-code-memory's hook system, but
integrated into the agent loop rather than via external process hooks.
"""

import time
from typing import Optional
from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.injection import (
    PerTurnDedupTracker,
    inject_memory_context,
    InjectionConfig,
)
from agent.cognitive_memory.extraction import extract_from_turn, ExtractionResult
from agent.cognitive_memory.retrieval import get_relevant_entries


class SessionContext:
    """Tracks per-session state for ambient memory injection.

    Keeps a dedup tracker so the same memory isn't injected twice,
    and records conversation turns for end-of-session extraction.
    """

    def __init__(self, session_id: str = ""):
        self.session_id = session_id or f"session_{int(time.time())}"
        self.injected_ids: set[str] = set()
        self._dedup_tracker = PerTurnDedupTracker()
        self._turns: list[tuple[str, str]] = []  # (user_msg, assistant_msg)
        self.started_at = time.time()

    def record_turn(self, user_message: str, assistant_response: str) -> None:
        """Record a conversation turn for later extraction."""
        self._turns.append((user_message, assistant_response))

    @property
    def turns(self) -> list[tuple[str, str]]:
        return list(self._turns)

    @property
    def dedup(self) -> PerTurnDedupTracker:
        return self._dedup_tracker

    @property
    def turn_count(self) -> int:
        return len(self._turns)


def session_start_injection(
    store: CognitiveMemoryStore,
    session: SessionContext,
    git_signals: Optional[dict] = None,
) -> str:
    """Inject memory context at session start.

    Searches for memories relevant to git signals (branch name, modified files,
    recent commits) and injects them as context. This helps the agent
    orient itself at the start of a new session.

    Args:
        store: The cognitive memory store.
        session: Session context (dedup tracker will be cleared).
        git_signals: Optional dict with keys: branch, files, commits.

    Returns:
        Formatted context string for injection, or "" if nothing found.
    """
    session.dedup.clear()

    signals = git_signals or {}
    queries: list[str] = []

    if "branch" in signals:
        branch = signals["branch"]
        # Extract meaningful terms from branch name
        terms = branch.replace("-", " ").replace("_", " ").replace("/", " ")
        queries.append(terms)

    if "files" in signals:
        files = signals["files"]
        if isinstance(files, list) and files:
            # Extract directory names and basenames
            parts: set[str] = set()
            for f in files[:5]:
                for part in f.replace("/", " ").replace(".", " ").split():
                    if len(part) > 2 and part not in ("src", "test", "tests", "main", "app"):
                        parts.add(part)
            if parts:
                queries.append(" ".join(list(parts)[:8]))

    # If no signals, inject top recent entries
    if not queries:
        recent = store.list_all(limit=5)
        from agent.cognitive_memory.retrieval import get_context_for_prompt
        if recent:
            return get_context_for_prompt(recent)
        return ""

    # Search with each query and merge results
    all_contexts: list[str] = []
    for q in queries:
        ctx = inject_memory_context(store, q, session.dedup)
        if ctx:
            all_contexts.append(ctx)

    if all_contexts:
        return "\n".join(all_contexts)

    return ""


def session_end_extraction(
    store: CognitiveMemoryStore,
    session: SessionContext,
) -> list[MemoryEntry]:
    """Run extraction pass across all collected turns at session end.

    For each recorded turn, runs passive extraction and stores any
    resulting candidates in the cognitive store with provenance
    CONVERSATION_EXTRACTED.

    Args:
        store: The cognitive memory store.
        session: Session context with recorded turns.

    Returns:
        List of newly stored MemoryEntry objects.
    """
    stored: list[MemoryEntry] = []

    for user_msg, assistant_msg in session.turns:
        results = extract_from_turn(user_msg, assistant_msg)
        for result in results:
            # Derive title from first sentence
            title = result.content.split(".")[0].strip()[:80] or result.content[:50]

            entry = MemoryEntry(
                type=result.memory_type,
                title=title,
                content=result.content,
                confidence=result.confidence,
                provenance=result.provenance,
                source_session_id=session.session_id,
            )
            store.insert(entry)
            stored.append(entry)

    return stored
