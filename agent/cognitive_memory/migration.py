"""
Migration from file-based memory (MEMORY.md/USER.md) to typed cognitive memory.

Classification heuristics ported from claude-code-memory's classifier.ts:
  - Build/test/deploy commands → procedural
  - User corrections, preferences, rules → pattern
  - Project facts, environment details → semantic (default)

Entry delimiter: "\n§\n" (section sign between newlines), matching the existing
MemoryStore format in tools/memory_tool.py.

On migration, all entries get provenance=USER_EXPLICIT (confidence 1.0) because
they were explicitly stored by the agent/user via the memory tool.
"""

import os
import re
from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore

MEMORY_FILES = ["MEMORY.md", "USER.md"]
SECTION_DELIMITER = "\n§\n"

MIGRATION_MARKER_FILE = ".cognitive_memory_migrated"


def classify_entry(content: str) -> "tuple[MemoryType, float]":
    """Classify a single memory entry into a MemoryType using regex heuristics.

    Returns (type, confidence) where confidence is based on signal strength.
    Defaults to SEMANTIC if no strong signals are found.
    """
    content_lower = content.lower()

    # --- Procedural signals ---
    procedural_patterns = [
        r"\b(build|compile|deploy|release|install|publish)\b.*\b(with|using|via|command)\b",
        r"\b(run|execute|test|lint|format)\b.*\b(pytest|npm|pip|make|cargo|go|docker|kubectl)\b",
        r"\b(command|workflow|step|procedure|recipe)\b.*\b(to|for|how)\b",
        r"`[^`]+`.*`[^`]+`",  # Multiple inline code references
    ]
    procedural_score = 0
    for pattern in procedural_patterns:
        if re.search(pattern, content, re.IGNORECASE):
            procedural_score += 1

    # --- Pattern signals ---
    pattern_signals = [
        r"\b(always|never|must|should|don't|do not|remember)\b",
        r"\b(correction|corrected|prefer|prefers|rule|convention|policy)\b",
        r"\b(reply|respond|speak|write|answer)\b.*\b(english|czech|language|only)\b",
        r"\b(never|don't|do not)\b.*\b(without|unless|before)\b.*\b(asking|approval|permission)\b",
    ]
    pattern_score = 0
    for pattern in pattern_signals:
        if re.search(pattern, content, re.IGNORECASE):
            pattern_score += 1

    # --- Determine type ---
    if pattern_score > procedural_score and pattern_score >= 2:
        return MemoryType.PATTERN, min(0.6 + pattern_score * 0.1, 1.0)
    elif procedural_score >= 2:
        return MemoryType.PROCEDURAL, min(0.6 + procedural_score * 0.1, 1.0)
    elif pattern_score >= 1:
        return MemoryType.PATTERN, 0.6
    elif procedural_score >= 1:
        return MemoryType.PROCEDURAL, 0.6
    else:
        return MemoryType.SEMANTIC, 0.6


def _read_file_memory(memories_dir: str) -> "list[str]":
    """Read all entries from MEMORY.md and USER.md in the given directory.

    Returns a flat list of entry content strings.
    """
    entries: list[str] = []

    for filename in MEMORY_FILES:
        path = os.path.join(memories_dir, filename)
        if not os.path.isfile(path):
            continue

        with open(path, "r") as f:
            content = f.read().strip()

        if not content:
            continue

        # Split by section delimiter
        parts = content.split(SECTION_DELIMITER)
        for part in parts:
            part = part.strip()
            if part:
                entries.append(part)

    return entries


def migrate_file_memory(store: CognitiveMemoryStore, memories_dir: str) -> int:
    """Migrate file-based memory to the cognitive store.

    Reads MEMORY.md and USER.md from `memories_dir`, classifies each entry,
    and inserts it into the store with provenance=USER_EXPLICIT.

    Idempotent: if a migration marker exists, returns 0 immediately.

    Args:
        store: The CognitiveMemoryStore to populate.
        memories_dir: Path to ~/.hermes/memories/ containing MEMORY.md and USER.md.

    Returns:
        Number of entries migrated (0 if already done or no entries found).
    """
    # Check migration marker
    marker_path = os.path.join(memories_dir, MIGRATION_MARKER_FILE)
    if os.path.exists(marker_path):
        return 0

    entries = _read_file_memory(memories_dir)
    if not entries:
        return 0

    count = 0
    for content in entries:
        mem_type, _ = classify_entry(content)

        # Derive a title from the first sentence or first 50 chars
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
        store.insert(entry)
        count += 1

    # Write migration marker
    with open(marker_path, "w") as f:
        f.write(str(count))

    return count
