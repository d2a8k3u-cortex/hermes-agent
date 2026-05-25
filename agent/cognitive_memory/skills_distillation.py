"""
Skills distillation — the central pipeline that converts repeated memory
patterns into permanent skills.

This is the organizing principle of the entire cognitive memory system.
Everything feeds into this: patterns become skills, procedural clusters
become skill templates, and semantic clusters become project context.

Design:
  - Pattern memories with confidence ≥ 0.8 and access_count ≥ 5 become candidates
  - User approval is ALWAYS required before creating a skill
  - On approval, source memories are marked distilled_to (not deleted)
  - Generated skills include: goal, steps, commands, pitfalls, verification
"""

from dataclasses import dataclass, field
from agent.cognitive_memory.types import MemoryType, MemoryEntry, Provenance
from agent.cognitive_memory.store import CognitiveMemoryStore

SKILL_CANDIDATE_MIN_CONFIDENCE = 0.8
SKILL_CANDIDATE_MIN_ACCESS = 5
SKILL_CANDIDATE_MIN_CONTENT_LENGTH = 100


@dataclass
class SkillCandidate:
    """A memory entry that qualifies for skill distillation."""
    source_entry: MemoryEntry
    skill_name: str
    confidence: float
    reason: str = ""


def detect_skill_candidates(store: CognitiveMemoryStore) -> list[SkillCandidate]:
    """Scan pattern entries for skill distillation candidates.

    Criteria:
      - Type == PATTERN
      - Confidence >= 0.8
      - Access count >= 5
      - Content >= 100 chars (substantive)

    Returns sorted list of candidates, highest confidence first.
    """
    candidates: list[SkillCandidate] = []
    patterns = store.list_by_type(MemoryType.PATTERN, limit=100)

    for entry in patterns:
        if entry.confidence < SKILL_CANDIDATE_MIN_CONFIDENCE:
            continue
        if entry.access_count < SKILL_CANDIDATE_MIN_ACCESS:
            continue
        if len(entry.content) < SKILL_CANDIDATE_MIN_CONTENT_LENGTH:
            continue
        if entry.distilled_to is not None:
            continue  # Already distilled

        skill_name = _derive_skill_name(entry)
        candidates.append(SkillCandidate(
            source_entry=entry,
            skill_name=skill_name,
            confidence=entry.confidence,
            reason=f"Pattern '{entry.title}' meets thresholds: "
                    f"confidence={entry.confidence}, access={entry.access_count}",
        ))

    candidates.sort(key=lambda c: c.confidence, reverse=True)
    return candidates


def _derive_skill_name(entry: MemoryEntry) -> str:
    """Derive a clean skill name from entry title/content."""
    title = entry.title.lower()
    # Strip "Pattern:" prefix if present
    title = title.replace("pattern:", "").strip()
    # Convert to lowercase hyphenated
    name = title.lower().replace(" ", "-").replace("_", "-")
    # Remove non-alphanumeric/hyphen chars
    name = "".join(c for c in name if c.isalnum() or c == "-")
    # Collapse multiple hyphens
    while "--" in name:
        name = name.replace("--", "-")
    return name.strip("-")[:64]


def generate_skill_draft(candidate: SkillCandidate) -> dict:
    """Generate a draft SKILL.md from a pattern memory.

    Returns dict with keys: skill_name, content (markdown body).
    The caller uses skill_manage(action='create') to create the skill.
    """
    entry = candidate.source_entry
    content = entry.content

    # Extract steps from content (numbered or bullet points)
    steps = _extract_steps(content)

    # Build the SKILL.md
    skill_md = f"""---
name: {candidate.skill_name}
description: "Auto-generated from cognitive memory pattern: {entry.title}"
version: 1.0.0
author: Hermes Cognitive Memory
license: MIT
metadata:
  hermes:
    tags: [auto-generated, cognitive-memory]
    source_memory_id: {entry.id}
---

# {entry.title}

{content}

## When to Use

This skill was distilled from repeated patterns in conversation. Apply it when:
- The task matches: {entry.title}
- The situation described in the content above applies

"""

    if steps:
        skill_md += "## Steps\n\n"
        for i, step in enumerate(steps, 1):
            skill_md += f"{i}. {step}\n"
        skill_md += "\n"

    skill_md += """## Pitfalls

- This skill was auto-generated. Review and refine before relying on it.
- The source memory was learned from conversation extraction — verify accuracy.
"""

    return {
        "skill_name": candidate.skill_name,
        "content": skill_md,
        "source_entry_id": entry.id,
    }


def _extract_steps(content: str) -> list[str]:
    """Extract numbered or procedural steps from content text."""
    steps: list[str] = []
    lines = content.split("\n")
    for line in lines:
        line = line.strip()
        # Numbered: "1. Do something" or "Step 1: Do something"
        if line and (line[0].isdigit() and ". " in line[:4]):
            steps.append(line.split(". ", 1)[1] if ". " in line else line)
        # Bullet: "- Do something"
        elif line.startswith("- ") and len(line) > 3:
            steps.append(line[2:])
    return steps
