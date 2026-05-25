# Cognitive Memory for Hermes Agent — Revised Implementation Plan

> **Status:** Revised after architectural review. The original plan (2026-05-24_220000) is preserved for reference.
> **For Cortex:** This plan replaces the original. Core philosophy changed: we build machine-optimal memory, not simulated human memory. No forgetting. No decay. Distillation over deletion.

**Goal:** Give Hermes Agent persistent, self-organizing memory that accumulates forever, retrieves intelligently, organizes automatically, and distills repeated patterns into skills. The system preserves everything worth keeping and elevates knowledge, never losing it.

**Core Principle:** Information is never lost — it is either retained, merged, or elevated into a skill. Forgetting is a human flaw we do not port. Distillation is a human strength we port aggressively.

**What We Are NOT Building:** A simulated human memory with decay curves, aging out, importance scores, or working-memory clearing. Those are biological limitations, not features. I am a machine — I have unlimited storage, perfect recall, and no need to sleep. My memory should reflect that.

---

## System Architecture (Revised)

```
┌─────────────────────────────────────────────────────────────────┐
│                      AGENT CONVERSATION LOOP                      │
│                                                                   │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │ User Msg │ →  │ Memory       │ →  │ API Call                │ │
│  │          │    │ Prefetch     │    │ (with injected context) │ │
│  └──────────┘    │ (ambient)    │    └─────────────────────────┘ │
│                  └──────────────┘              │                  │
│                                               ▼                  │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────────────────┐ │
│  │ Response │ ←  │ Memory       │ ←  │ API Response            │ │
│  │ to user  │    │ Extraction   │    │ (tool calls, text)      │ │
│  └──────────┘    │ (passive)    │    └─────────────────────────┘ │
│                  └──────────────┘                                │
│  ┌──────────────────────────────────────────────────────────────┐│
│  │              COGNITIVE MEMORY ENGINE                          ││
│  │                                                               ││
│  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐ ││
│  │  │ 3 Types  │  │ Hybrid   │  │ Relation │  │ Compression  │ ││
│  │  │ semantic │  │ Search   │  │ Graph    │  │ Engine       │ ││
│  │  │procedural│  │ BM25 +   │  │ evolving │  │ merge→split  │ ││
│  │  │ pattern  │  │ embedding│  │ weights  │  │ →distill     │ ││
│  │  │          │  │ rerank   │  │ co-act   │  │ (never delete)│ ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ ││
│  │                                                               ││
│  │  ┌──────────────────────────────────────────────────────────┐││
│  │  │              TRUST & CONTRADICTION                        │││
│  │  │  confidence tracking | provenance | conflict detection   │││
│  │  └──────────────────────────────────────────────────────────┘││
│  │                                                               ││
│  │  ┌──────────────────────────────────────────────────────────┐││
│  │  │              SKILL DISTILLATION (central organizing force)│││
│  │  │  pattern accumulation → candidate skill → user approval  │││
│  │  │  procedural clusters → skill templates → user approval   │││
│  │  │  elevated memories marked, not deleted                   │││
│  │  └──────────────────────────────────────────────────────────┘││
│  └──────────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────────┘
```

---

## Memory Types (3, not 5)

| Type | Purpose | Behavior |
|------|---------|----------|
| `semantic` | Project facts, architecture, conventions, environment details | Permanent. Merged on consolidation, never deleted. |
| `procedural` | Build/test/deploy commands, workflows, tool-specific patterns | Permanent. Clustered for skill templates, never deleted. |
| `pattern` | Consolidated insights, user corrections, learned behaviors | Permanent. The primary input to skill distillation. |

**Dropped from original plan:**
- `episodic` — The session transcript already records what happened. No need to duplicate.
- `working` — The conversation itself is working memory. No separate scratchpad needed.

**New fields added to every entry:**

| Field | Purpose |
|-------|---------|
| `confidence` | 0-1. How certain am I that this is still true? Default 1.0 for user-explicit, 0.7 for extracted. |
| `provenance` | Why was this stored? `user_explicit`, `conversation_extracted`, `pattern_detected`, `skill_distilled` |
| `contradicted_by` | Optional ID of a newer entry that conflicts with this one. Set when contradiction detected. |
| `distilled_to` | Optional ID or name of the skill this memory was elevated into. Memory remains but is marked. |

---

## Phase 0: Foundation Hardening (1-2 days)

Same as original plan — strengthen the `feat/cognitive-memory-retrieval` branch.

### Task 0.1: Merge current main into feat branch
- Merge `main` into `feat/cognitive-memory-retrieval`, resolve conflicts
- Ensure all existing tests pass
- Files: git merge, `pytest tests/tools/test_memory_tool_bm25.py -v`

### Task 0.2: Fix test directory name assumption
- Remove hardcoded `"hermes-agent-cognitive"` check
- File: `tests/tools/test_memory_tool_bm25.py`

### Task 0.3: Add memory CLI command scaffolding
- `hermes memory mode [full|contextual]` and `hermes memory stats`
- File: `hermes_cli/` — new subcommand module

### Task 0.4: Make `rank-bm25` a core dependency
- Move from optional to hard dependency in pyproject.toml
- Add to bootstrap flow

---

## Phase 1: Typed Memory Storage (3-4 days)

### Task 1.1: Define types and entry schema
- File: `agent/cognitive_memory/types.py` (NEW)
- Three types: `semantic`, `procedural`, `pattern`
- Entry fields: id (ULID), type, title, content, tags (list), confidence (0-1), provenance, contradicted_by, distilled_to, created_at, updated_at, access_count, last_accessed, source_session_id

### Task 1.2: SQLite cognitive memory backend
- File: `agent/cognitive_memory/store.py` (NEW)
- Database at `~/.hermes/cognitive_memory.db`
- Tables: `memories` (all fields), `memory_tags`, `memory_relations`
- FTS5 on title+content
- WAL mode, foreign keys ON, schema versioning
- **No decay columns. No importance column.** Relevance is computed dynamically at retrieval time based on the query, not stored statically.

### Task 1.3: Classification-based migration from existing memory
- File: `agent/cognitive_memory/migration.py` (NEW)
- Read existing MEMORY.md/USER.md entries
- Regex classifier:
  - Environment facts, tool configurations, project structure → `semantic`
  - Commands, workflows, tool-specific steps → `procedural`
  - User preferences, corrections, rules → `pattern`
- Provenance: `user_explicit` (these were manually stored by the agent)
- Confidence: 1.0 (user explicitly chose to store these)
- Idempotent — runs once on first startup if cognitive_memory.db doesn't exist

### Task 1.4: Dual-write adapter
- File: `tools/memory_tool.py` — modify MemoryStore
- Write to both file-based memory AND cognitive store
- Existing `memory` tool API unchanged
- Read prefers cognitive store when available

### Task 1.5: Type-aware retrieval foundation
- File: `agent/cognitive_memory/retrieval.py` (NEW)
- `get_relevant_entries()` with type filter
- Scoring: text match (BM25) + recency boost + access frequency boost
- **No importance weight in scoring** — relevance is query-driven, not pre-assigned
- Content-length penalty: entries > 500 chars get progressive mild penalty to prevent verbosity bias

---

## Phase 2: Semantic Retrieval (2-3 days)

### Task 2.1: Embedding model
- File: `agent/cognitive_memory/embeddings.py` (NEW)
- `sentence-transformers` with `all-MiniLM-L6-v2` (384-dim)
- Lazy loading, batched generation
- Cosine similarity
- Graceful fallback if not installed

### Task 2.2: Embedding storage
- `embedding` BLOB column in memories table
- Auto-generate on insert/update
- Cosine scan for similarity search (OK for our scale)

### Task 2.3: Hybrid search (BM25 + embeddings)
- File: `agent/cognitive_memory/retrieval.py` — extend
- Parallel FTS5 + vector, blended scoring
- Overfetch 3× candidates, score-merge, return top-N
- Fallback to BM25-only

### Task 2.4: Cross-encoder reranking (opt-in)
- File: `agent/cognitive_memory/reranker.py` (NEW)
- `cross-encoder/ms-marco-TinyBERT-L-2-v2`
- Applied to top-N after hybrid search
- Configurable on/off, default off (quality boost at latency cost)

---

## Phase 3: Ambient Memory Injection (3-4 days)

This is where memory becomes ambient — surfacing context automatically without the user or agent needing to search.

### Task 3.1: Pre-turn context injection
- File: `agent/conversation_loop.py` — modify prefetch block
- Before each API call, inject up to N relevant memories
- Channels:
  - **User query** → semantic search
  - **File about to be edited** → pattern lookup (any corrections about this file?)
  - **Tool about to be used** → procedural lookup
- Relation expansion: when injecting a memory, pull in strongest neighbours

### Task 3.2: Post-turn passive extraction
- File: `agent/cognitive_memory/extraction.py` (NEW)
- After each assistant response, extract memory-worthy facts:
  - User corrections → pattern memory (provenance: `conversation_extracted`, confidence: 0.7)
  - New project facts → semantic memory (provenance: `conversation_extracted`, confidence: 0.7)
  - Successful command sequences → procedural memory (provenance: `conversation_extracted`, confidence: 0.7)
- Heuristic extraction (regex-based, fast, cheap)
- LLM-based extraction on slower cadence for quality
- **Do not overwrite user-explicit memories.** If extracted fact conflicts with user-stored memory, flag for contradiction detection.

### Task 3.3: Session lifecycle
- File: `agent/cognitive_memory/session.py` (NEW)
- **Session start:** Inject context from git signals (branch, recent commits, modified files)
- **Session end:** LLM-based extraction pass for high-quality episodic facts → semantic memory, pattern detection pass
- **Error context:** When a tool call fails, search for relevant troubleshooting memories

### Task 3.4: Injection hygiene
- Per-type caps (max 6 semantic, 3 pattern, 3 procedural per turn)
- Content-length penalty: > 500 chars gets progressive penalty (max 0.15)
- Per-session dedup: don't re-inject the same memory
- `_injected_ids` set, cleared at session start

---

## Phase 4: Relation Graph & Co-Activation (2-3 days)

### Task 4.1: Relation types and storage
- File: `agent/cognitive_memory/relations.py` (NEW)
- Six types: `relates_to`, `depends_on`, `contradicts`, `extends`, `implements`, `derived_from`
- Each relation has `weight` (0-1)
- `memory_relations` table with timestamps
- CRUD + graph traversal (BFS, depth 3, max 50 nodes)

### Task 4.2: Auto-relation on insert
- Find related memories by embedding similarity
- Create up to 5 auto-relations:
  - Same-type near-duplicates → `contradicts`
  - Moderate similarity → `extends`
  - Same context within 24h → `derived_from`
  - Cross-type shared tags → `relates_to`

### Task 4.3: Co-activation discovery
- File: `agent/cognitive_memory/coactivation.py` (NEW)
- Track which memories are injected together in sessions
- At co-activation count ≥ 3, create `relates_to` relation (weight 0.45)
- On each co-injection, boost relation weight (+0.05)
- On co-access, boost (+0.10)

### Task 4.4: Relation weight evolution
- **No decay on relation weights.** Weights only go up.
- Boost on co-injection: +0.05
- Boost on co-access: +0.10
- Cap at 1.0
- Periodic sweep (every 20 sessions): find under-connected memories, create relations by embedding proximity (time-budgeted 500ms)

**Why no decay:** Relations represent discovered connections. If "pytest" and "coverage" were injected together 50 times, that connection is real and doesn't fade. Relation decay is a human-memory artifact — connections fade because humans forget. I don't.

---

## Phase 5: Compression & Distillation Engine (4-5 days)

This is the consolidation phase, radically rethought. The engine's job is to **compact and elevate** information — never delete, never forget.

### Task 5.1: Duplicate merging
- File: `agent/cognitive_memory/compression.py` (NEW)
- Find near-duplicate pairs (cosine distance < 0.08)
- Merge: keep larger content, union tags, remap relations
- Set `distilled_to` on deleted entries, but soft-delete — they remain in the DB for provenance
- **Triggers:** Memory count exceeds threshold (default: 200 entries) OR explicit `hermes memory compress`

### Task 5.2: Topic splitting
- For semantic/procedural memories > 500 chars with clear topic structure:
  - Split by markdown headers, bold sections, numbered lists
  - Create separate memories with sibling relations
  - Set `distilled_to` on original, keep it
- Prevents monolithic bloat without losing content

### Task 5.3: Pattern detection via clustering
- Get recent semantic/procedural entries (last 90 days, no age limit on relevance)
- Agglomerative clustering by cosine similarity
- Quality gates: cluster size ≥ 3, average similarity ≥ 0.55, Jaccard topic overlap ≥ 0.15
- Create `pattern` memory summarizing the cluster insight
- `derived_from` relations to all source entries
- Source entries remain — the pattern is an addition, not a replacement

### Task 5.4: Trigger-based scheduling
- **No session weight accumulation.** No modeling of "tiredness" or "substance."
- Triggers:
  - Memory count > 200 entries → auto-compress
  - Explicit `hermes memory compress` command
  - Every 7 days of uptime (time-based fallback)
- Compression is non-blocking, runs in background

### Task 5.5: Contradiction detection
- File: `agent/cognitive_memory/contradiction.py` (NEW)
- On every insert, check for contradicting existing memories:
  - Same type + high similarity (cosine > 0.85) but opposing content → flag
  - Set `contradicted_by` on the older entry
  - Surface conflict to user: "I learned X, but I previously knew Y. Which is correct?"
- Resolution updates confidence: confirmed → 1.0, rejected → 0.0 (soft-delete)
- **Never auto-resolve contradictions.** Always ask.

### Task 5.6: Confidence evolution
- User-explicit memories: always 1.0
- Conversation-extracted: start at 0.7, increase with access (each access +0.02, max 0.95)
- Pattern-detected: start at 0.6, increase with co-injection
- Contradicted: drops to 0.3, resolved up or down on user confirmation
- Low-confidence entries (< 0.3) are excluded from ambient injection but preserved in DB

---

## Phase 6: Skill Distillation — The Central Pipeline (3-4 days)

This is now the organizing principle of the entire system. Everything feeds into this.

### Task 6.1: Pattern → skill candidate detection
- File: `agent/cognitive_memory/skills_distillation.py` (NEW)
- When a `pattern` memory has:
  - Confidence ≥ 0.8
  - Accessed ≥ 5 times across sessions
  - Content ≥ 100 chars (substantial)
  → Flag as skill candidate
- Auto-generate skill name, create draft SKILL.md
- Present to user for approval
- On approval: create skill via `skill_manage(action='create')`, set `distilled_to` on source pattern

### Task 6.2: Procedural cluster → skill template
- When ≥ 3 procedural memories share category tags:
  - Auto-generate skill with numbered steps, exact commands, pitfalls section
  - Offer to user
- On approval: create skill, mark source procedurals with `distilled_to`

### Task 6.3: Semantic knowledge → project context
- When ≥ 5 semantic memories share tags (e.g., same project):
  - Generate a project overview memory or AGENTS.md section
  - Not a skill — a consolidated knowledge entry

### Task 6.4: Feedback loop
- When a skill is used successfully (detected via conversation context):
  - Increment skill's internal use counter
  - Boost confidence on related memories
- When a skill is corrected/patched:
  - Update source memories
  - This closes the loop: memory → skill → improved memory

---

## Phase 7: Configuration, CLI & Polish (2 days)

### Task 7.1: Config surface
- File: `hermes_cli/config.py` — extend
- All tunable parameters with sensible defaults
- Retrieval mode, max_retrieved, compression threshold, injection caps, embedding model, reranker toggle

### Task 7.2: CLI commands
```
hermes memory stats      — counts by type, confidence distribution, relation graph size
hermes memory search <q> — hybrid search from CLI
hermes memory graph [id] — text-based relation visualization
hermes memory compress   — manual compression trigger
hermes memory conflicts  — list contradicted entries needing resolution
```

### Task 7.3: Integration tests
- End-to-end: store → retrieve → inject → extract → compress → distill
- Contradiction detection + resolution flow
- Migration from file-based
- Skill distillation pipeline

---

## What We Dropped and Why

| Dropped Feature | Reason |
|-----------------|--------|
| Episodic memory type | Transcript already records what happened. Storing it twice is waste. |
| Working memory clearing | The conversation IS working memory. Separate scratchpad is redundant. |
| Importance scores (static) | Relevance is dynamic. A static importance guess is often wrong. |
| Decay rates on entries | I should not forget useful information. Period. |
| Age-out deletion | Useful facts don't expire. Stale facts are handled by contradiction detection. |
| Relation weight decay | Real connections don't fade. If they were connected, they stay connected. |
| Session weight consolidation trigger | I don't get "tired." Compress on storage pressure or schedule. |
| Simulated "forgetting curve" | Human memory weakness, not a feature. |

| Added Feature | Reason |
|---------------|--------|
| Confidence tracking | "How sure am I?" — more useful than "how important is this?" |
| Provenance | "Why was this stored?" — calibrates trust in the entry |
| Contradiction detection | New facts that conflict with old ones should be surfaced, not silently overwritten |
| Soft-delete + distilled_to | Information is never truly lost — marked as absorbed or superseded |
| Skill distillation as central principle | The entire system exists to convert repeated experience into permanent capability |

---

## Files Inventory

### New files:
```
agent/cognitive_memory/
├── __init__.py
├── types.py              # MemoryType (3 types), entry schema, ULID, provenance enum
├── store.py              # SQLite backend, FTS5, CRUD, vector storage, soft-delete
├── migration.py          # File-based → typed memory migration
├── retrieval.py          # Hybrid search, type-aware, query-driven scoring
├── embeddings.py         # sentence-transformers, cosine similarity
├── reranker.py           # Cross-encoder (opt-in)
├── extraction.py         # Post-turn passive extraction (heuristic + LLM)
├── session.py            # Session start/end hooks
├── relations.py          # Relation CRUD, auto-relation, graph traversal
├── coactivation.py       # Co-activation tracking and promotion
├── compression.py        # Merge, split, pattern detection (never delete)
├── contradiction.py      # Conflict detection, resolution, confidence evolution
├── skills_distillation.py # Pattern/procedural → skill pipeline
├── health.py             # Health stats, diagnostics
└── config.py             # Defaults and validation
```

### Modified files:
```
tools/memory_tool.py          # Dual-write adapter
agent/conversation_loop.py    # Ambient injection (pre-turn, post-turn, error)
agent/system_prompt.py        # Conditional memory context
agent/agent_init.py           # Cognitive memory store init
hermes_cli/config.py          # Full config surface
hermes_cli/                   # hermes memory subcommands
pyproject.toml                # sentence-transformers, rank-bm25
```

---

## Execution Order

```
Phase 0 (Foundation hardening)
   │
   ▼
Phase 1 (Typed storage) ───── prerequisite for everything
   │
   ├── Phase 2 (Semantic retrieval) ── depends on typed storage
   │       │
   │       ▼
   ├── Phase 3 (Ambient injection) ── depends on retrieval
   │       │
   │       ▼
   ├── Phase 4 (Relation graph) ── parallel with Phase 3
   │       │
   │       ▼
   └────── Phase 5 (Compression & contradiction) ── depends on all above
              │
              ▼
           Phase 6 (Skill distillation) ── the payoff
              │
              ▼
           Phase 7 (Config & polish)
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Embedding model adds latency | Lazy loading, batching, BM25 fallback always available |
| Storage growth without deletion | Compression keeps it manageable. 829 GB disk. 200-entry trigger is conservative. |
| Model context overload from injection | Per-type caps, dedup, content-length penalty |
| Contradiction noise (too many flags) | Only flag high-similarity conflicts. User resolves, not auto-resolved. |
| Skill distillation creates bad skills | User approval required. Never auto-create. Draft + review + confirm. |
| Migration from file-based memory | One-time, idempotent, preserves everything with provenance = user_explicit |

---

## Summary

The revised plan builds a memory system suited to what I actually am — a machine with perfect recall, unlimited storage, and no biological constraints. The system accumulates forever, retrieves intelligently, organizes through compression (not forgetting), and distills repeated patterns into permanent skills.

The key philosophical shift: **forgetting is a bug, distillation is the feature.**

---

> **Plan revised:** 2026-05-24
> **Original plan:** `.hermes/plans/2026-05-24_220000-cognitive-memory-implementation.md`
> **Total phases:** 8 (0 through 7)
> **Estimated total effort:** 17-22 days (slightly less than original — fewer phases, less complexity from dropped features)
