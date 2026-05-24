"""
Cognitive memory types — MemoryType enum, MemoryEntry data structure,
Provenance tracking, and ULID generation.

Three memory types, no more:
  - semantic:   project facts, architecture, conventions, environment details
  - procedural: build/test/deploy commands, workflows, tool-specific patterns
  - pattern:    consolidated insights, user corrections, learned behaviors

Key design decisions:
  - No importance scores — relevance is dynamic, computed at query time
  - No decay — information is never lost, only merged or elevated
  - Confidence tracks certainty, not "importance"
  - Provenance tracks why a memory exists, for trust calibration
"""

import time
import os
import hashlib
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional


class MemoryType(Enum):
    """Exactly three memory types — stripped to what a machine actually needs."""

    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    PATTERN = "pattern"


class Provenance(Enum):
    """Why was this memory stored? Calibrates trust in the entry."""

    USER_EXPLICIT = "user_explicit"
    CONVERSATION_EXTRACTED = "conversation_extracted"
    PATTERN_DETECTED = "pattern_detected"
    SKILL_DISTILLED = "skill_distilled"


DEFAULT_CONFIDENCE = {
    Provenance.USER_EXPLICIT: 1.0,
    Provenance.CONVERSATION_EXTRACTED: 0.7,
    Provenance.PATTERN_DETECTED: 0.6,
    Provenance.SKILL_DISTILLED: 1.0,
}

_CONTRADICTED_CONFIDENCE = 0.3


def generate_ulid() -> str:
    """Generate a ULID — unique, lexicographically sortable, timestamp-based.

    Format: 26 characters, Crockford base32 (0-9, A-H JKMNP-TV-Z).
    First 10 chars encode a millisecond timestamp, last 16 are random.
    """
    timestamp = int(time.time() * 1000)
    randomness = os.urandom(10)

    # Crockford base32 encoding
    def _encode_timestamp(ts: int) -> str:
        encoding = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
        result = []
        for _ in range(10):
            result.append(encoding[ts & 0x1F])
            ts >>= 5
        return "".join(reversed(result))

    def _encode_randomness(data: bytes) -> str:
        encoding = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
        result = []
        # Process 10 bytes into 16 base32 characters (5 bits each)
        bits = int.from_bytes(data, "big")
        for _ in range(16):
            result.append(encoding[bits & 0x1F])
            bits >>= 5
        return "".join(reversed(result))

    return _encode_timestamp(timestamp) + _encode_randomness(randomness)


@dataclass
class MemoryEntry:
    """A single typed memory entry.

    Fields:
        id: ULID — unique, sortable by creation time
        type: MemoryType — semantic, procedural, or pattern
        title: short descriptive title
        content: the memory content itself
        tags: normalized deduplicated lowercase tag list
        confidence: 0.0-1.0 — how certain this is still true
        provenance: Provenance — why this was stored
        contradicted_by: optional ID of a newer conflicting entry
        distilled_to: optional skill name this was elevated into
        created_at: POSIX timestamp
        updated_at: POSIX timestamp
        access_count: times accessed for retrieval
        last_accessed: POSIX timestamp of last access
        source_session_id: optional session this was learned in
    """

    type: MemoryType
    title: str
    content: str

    # Auto-generated
    id: str = field(default_factory=generate_ulid)
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    provenance: Provenance = Provenance.USER_EXPLICIT
    contradicted_by: Optional[str] = None
    distilled_to: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    access_count: int = 0
    last_accessed: Optional[float] = None
    source_session_id: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize after construction."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(
                f"confidence must be between 0.0 and 1.0, got {self.confidence}"
            )
        # Normalize tags: lowercase, strip whitespace, deduplicate
        seen: set[str] = set()
        normalized: list[str] = []
        for tag in self.tags:
            t = tag.strip().lower()
            if t and t not in seen:
                seen.add(t)
                normalized.append(t)
        self.tags = normalized

    def mark_contradicted(self, newer_entry_id: str) -> None:
        """Mark this entry as contradicted by a newer entry.

        Confidence drops to signal uncertainty — the system should ask the user
        to resolve the conflict, not auto-delete either entry.
        """
        self.contradicted_by = newer_entry_id
        self.confidence = _CONTRADICTED_CONFIDENCE

    def mark_distilled(self, skill_name: str) -> None:
        """Mark this entry as elevated into a skill.

        The entry remains in the database — distillation is elevation, not deletion.
        Confidence is unaffected.
        """
        self.distilled_to = skill_name

    def register_access(self) -> None:
        """Record that this memory was accessed for retrieval.

        Increments access_count and updates last_accessed timestamp.
        Used for recency/access-frequency scoring at retrieval time.
        """
        self.access_count += 1
        self.last_accessed = time.time()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MemoryEntry):
            return NotImplemented
        return self.id == other.id

    def __hash__(self) -> int:
        return hash(self.id)

    def __repr__(self) -> str:
        return (
            f"MemoryEntry(type={self.type.value}, "
            f"title={self.title[:50]!r}, "
            f"confidence={self.confidence})"
        )
