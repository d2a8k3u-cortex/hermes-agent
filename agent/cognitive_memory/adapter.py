"""
Dual-write adapter — hooks into existing MemoryStore to mirror writes
to the cognitive memory store.

This adapter wraps a CognitiveMemoryStore and provides mirror_* methods
that should be called from within MemoryStore.add(), .replace(), and .remove().

Design:
  - Non-invasive: adds cognitive writes alongside existing file writes.
    The existing MemoryStore API is unchanged — this is a write-through cache.
  - Classifies entries on mirror using the migration classifier for consistency.
  - All mirrored entries get provenance=USER_EXPLICIT (confidence 1.0) because
    they were explicitly stored via the memory tool.
  - Returns entry IDs so callers can track what was mirrored.
  - Gracefully disabled when no cognitive store is available.
"""

from typing import Optional

from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore
from agent.cognitive_memory.migration import classify_entry


class CognitiveMemoryAdapter:
    """Dual-write adapter bridging MemoryStore and CognitiveMemoryStore.

    Usage:
        adapter = CognitiveMemoryAdapter(cognitive_store)

        # In MemoryStore.add(), after writing to file:
        if self._cognitive_adapter:
            self._cognitive_adapter.mirror_add(target, content)

        # In MemoryStore.replace():
        if self._cognitive_adapter:
            self._cognitive_adapter.mirror_replace(target, old_text, new_content)

        # In MemoryStore.remove():
        if self._cognitive_adapter:
            self._cognitive_adapter.mirror_remove(target, old_text)
    """

    def __init__(self, cognitive_store: Optional[CognitiveMemoryStore] = None):
        self._store = cognitive_store
        self.enabled = cognitive_store is not None

    def mirror_add(self, target: str, content: str) -> Optional[str]:
        """Mirror a MemoryStore.add() to the cognitive store.

        Args:
            target: 'memory' or 'user' (from MemoryStore)
            content: The raw entry content that was added.

        Returns:
            The ULID of the created cognitive memory entry, or None if disabled.
        """
        if not self.enabled or not self._store:
            return None

        mem_type, _ = classify_entry(content)

        # Derive title from first sentence
        first_sentence = content.split(".")[0].strip()
        title = first_sentence[:80] if len(first_sentence) > 80 else first_sentence
        if not title:
            title = content[:50]

        entry = MemoryEntry(
            type=mem_type,
            title=title,
            content=content,
            provenance=Provenance.USER_EXPLICIT,
            confidence=1.0,
        )
        self._store.insert(entry)
        return entry.id

    def mirror_replace(
        self, target: str, old_text: str, new_content: str
    ) -> Optional[str]:
        """Mirror a MemoryStore.replace() to the cognitive store.

        Finds the cognitive entry matching old_text, updates it with new_content.
        If not found, no-op (the file-based store may have had the entry added
        before the cognitive store was initialized).

        Args:
            target: 'memory' or 'user'
            old_text: Substring to search for in stored entries.
            new_content: Replacement content.

        Returns:
            The ULID of the updated entry, or None if not found or disabled.
        """
        if not self.enabled or not self._store:
            return None

        # Search for matching entry in cognitive store
        old_text_stripped = old_text.strip()
        entries = self._store.list_all(limit=200)
        match = None
        for entry in entries:
            if old_text_stripped in entry.content:
                match = entry
                break

        if match is None:
            return None

        # Re-classify (type may have changed)
        mem_type, _ = classify_entry(new_content)
        match.type = mem_type
        match.title = new_content.split(".")[0].strip()[:80] or new_content[:50]
        match.content = new_content.strip()
        self._store.update(match)
        return match.id

    def mirror_remove(self, target: str, old_text: str) -> bool:
        """Mirror a MemoryStore.remove() to the cognitive store.

        Args:
            target: 'memory' or 'user'
            old_text: Substring to search for in stored entries.

        Returns:
            True if an entry was deleted, False if not found or disabled.
        """
        if not self.enabled or not self._store:
            return False

        old_text_stripped = old_text.strip()
        entries = self._store.list_all(limit=200)
        for entry in entries:
            if old_text_stripped in entry.content:
                self._store.delete(entry.id)
                return True

        return False
