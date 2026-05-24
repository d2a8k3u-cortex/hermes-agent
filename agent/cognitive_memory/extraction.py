"""
Post-turn passive extraction — heuristic fact extraction from conversation.

After each assistant response, this module scans the user message and
assistant response for memory-worthy facts. Extraction is heuristic-based
(regex, not LLM calls) for speed and zero cost. A heavier LLM-based
extraction pass runs at session end.

Design:
  - Fast: regex-based, runs in <1ms per turn
  - Conservative: prioritizes precision over recall (errors on side of silence)
  - Provenance: all extracted facts get provenance=conversation_extracted (0.7 conf)
  - Non-duplicative: uses existing store to avoid storing what's already known
"""

import re
from dataclasses import dataclass
from enum import Enum
from agent.cognitive_memory.types import MemoryType, Provenance


class ExtractableSignal(Enum):
    """Type of signal detected in a conversation turn."""
    USER_CORRECTION = "user_correction"       # "Don't do X", "Always do Y"
    PROCEDURAL_COMMAND = "procedural_command"  # Commands in backticks or "run:" patterns
    PROJECT_FACT = "project_fact"             # "This project uses X", "We have Y configured"
    NEW_KNOWLEDGE = "new_knowledge"            # Novel fact not previously discussed


@dataclass
class ExtractionResult:
    """A single extracted memory candidate."""
    content: str
    signal: ExtractableSignal
    memory_type: MemoryType
    confidence: float = 0.7
    provenance: Provenance = Provenance.CONVERSATION_EXTRACTED


# ── Correction patterns ──────────────────────────────────────────────────────

_CORRECTION_PATTERNS = [
    re.compile(r"\b(always|never|must|should)\s+\w+", re.IGNORECASE),
    re.compile(r"\b(don'?t|do not)\s+\w+.*\b(without|unless|before)\b", re.IGNORECASE),
    re.compile(r"\b(remember|keep in mind|note that)\b", re.IGNORECASE),
    re.compile(r"\b(from now on|going forward|henceforth)\b", re.IGNORECASE),
]

# ── Command patterns ─────────────────────────────────────────────────────────

_COMMAND_PATTERNS = [
    re.compile(r"`([^`]{10,})`"),  # Backtick commands >= 10 chars
    re.compile(r"\b(run|execute|build|deploy|test|install|publish)\s*[:\-]\s*(.+)", re.IGNORECASE),
    re.compile(r"\b(npm|pip|pytest|docker|kubectl|git|make|cargo|go)\s+\w+", re.IGNORECASE),
]

# ── Fact patterns ────────────────────────────────────────────────────────────

_FACT_PATTERNS = [
    re.compile(r"\b(this\s+project|we|the\s+system)\s+(uses?|has|runs?|employs?|supports?|is)", re.IGNORECASE),
    re.compile(r"\b(is\s+(configured|set\s+up|running|deployed)\s)", re.IGNORECASE),
    re.compile(r"\b(version\s+\d+\.\d+|v\d+\.\d+)", re.IGNORECASE),
]


def extract_from_turn(user_message: str, assistant_response: str) -> list[ExtractionResult]:
    """Extract memory-worthy facts from a single conversation turn.

    Scans the user message for corrections and the assistant response
    for procedural commands and project facts. Uses regex heuristics
    for speed — false positives are acceptable (they get flagged with
    lower confidence) but false negatives are preferred over noise.

    Args:
        user_message: The user's message text.
        assistant_response: The assistant's response text.

    Returns:
        List of ExtractionResult candidates for potential memory storage.
        Typically 0-3 results per turn.
    """
    results: list[ExtractionResult] = []

    # Skip trivial turns
    user_short = len(user_message.strip()) < 20 and len(assistant_response.strip()) < 20
    if user_short and not _is_important_short(user_message, assistant_response):
        return []

    # ── User corrections → pattern memories ──────────────────────────────
    for pattern in _CORRECTION_PATTERNS:
        match = pattern.search(user_message)
        if match:
            content = user_message.strip()
            # Only extract if it's a substantive correction
            if len(content) >= 20:
                results.append(ExtractionResult(
                    content=content[:300],
                    signal=ExtractableSignal.USER_CORRECTION,
                    memory_type=MemoryType.PATTERN,
                    confidence=0.7,  # Heuristic extraction → moderate confidence
                ))
                break  # One correction per turn max

    # ── Command patterns → procedural memories ──────────────────────────
    for pattern in _COMMAND_PATTERNS:
        for m in pattern.finditer(assistant_response):
            cmd = m.group(0).strip()
            if len(cmd) >= 10:
                results.append(ExtractionResult(
                    content=cmd,
                    signal=ExtractableSignal.PROCEDURAL_COMMAND,
                    memory_type=MemoryType.PROCEDURAL,
                    confidence=0.7,
                ))
                break
        if any(r.signal == ExtractableSignal.PROCEDURAL_COMMAND for r in results):
            break

    # ── Project facts → semantic memories ────────────────────────────────
    for pattern in _FACT_PATTERNS:
        m = pattern.search(assistant_response)
        if m:
            sentence = _extract_sentence(assistant_response, m)
            if len(sentence) >= 20:
                results.append(ExtractionResult(
                    content=sentence[:300],
                    signal=ExtractableSignal.PROJECT_FACT,
                    memory_type=MemoryType.SEMANTIC,
                    confidence=0.7,
                ))
                break

    return results


def _extract_sentence(text: str, match: re.Match) -> str:
    """Extract the full sentence containing a regex match."""
    start = match.start()
    end = match.end()

    # Expand backward to sentence start
    sentence_start = start
    for i in range(start - 1, max(start - 200, 0), -1):
        if text[i] in ".!\n" and i + 1 < start:
            sentence_start = i + 1
            break

    # Expand forward to sentence end
    sentence_end = end
    for i in range(end, min(end + 200, len(text))):
        if text[i] in ".!\n":
            sentence_end = i
            break

    return text[sentence_start:sentence_end].strip()


def _is_important_short(user_msg: str, assistant_msg: str) -> bool:
    """Check if a short message is worth extracting from."""
    combined = (user_msg + " " + assistant_msg).lower()
    important_words = [
        "always", "never", "remember", "don't", "do not",
        "command", "build", "test", "deploy", "version",
        "configured", "running", "important",
    ]
    return any(word in combined for word in important_words)
