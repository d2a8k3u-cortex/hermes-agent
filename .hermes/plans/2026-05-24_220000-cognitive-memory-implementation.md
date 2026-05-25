# Cognitive Memory for Hermes Agent — Strategic Implementation Plan

> **For Cortex:** This plan spans the complete cognitive memory system. Implement phase by phase, each phase building on the last. Use subagent-driven-development with TDD for each task.

**Goal:** Give Hermes Agent human-like cognitive memory — automatic learning from conversation, semantic retrieval, typed memory with differentiated survival strategies, autonomous consolidation, and memory-to-skill distillation. The system preserves the existing skill-from-memory pipeline while adding the ambient memory intelligence of claude-code-memory.

**Architecture:** A Python-native cognitive memory system integrated into Hermes' conversation loop. It builds on the existing `feat/cognitive-memory-retrieval` branch (BM25 retrieval) and adapts proven patterns from claude-code-memory v1.2 (5 memory types, hybrid search, evolutionary relation graph, session-weight consolidation, ambient injection). The system lives alongside — not replacing — the existing file-based memory and external provider plugins.

**Inspiration Source:** claude-code-memory v1.2 by d2a8k3u (TypeScript, SQLite + sqlite-vec, Xenova/all-MiniLM-L6-v2, cross-encoder reranking, hook-driven ambient injection)

**Current Foundation:** hermes-agent `feat/cognitive-memory-retrieval` branch (BM25 per-turn retrieval, `retrieval_mode` config, `max_retrieved` config)

---

## System Overview (Target Architecture)

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
│  │  │ 5 Types  │  │ Hybrid   │  │ Relation │  │ Consolidation│ ││
│  │  │ episodic │  │ Search   │  │ Graph    │  │ Engine       │ ││
│  │  │ semantic │  │ BM25 +   │  │ weights  │  │ merge→split  │ ││
│  │  │procedural│  │ embedding│  │ co-act   │  │ →detect→clean│ ││
│  │  │ working  │  │ rerank   │  │ evolve   │  │ auto-trigger │ ││
│  │  │ pattern  │  │          │  │          │  │              │ ││
│  │  └──────────┘  └──────────┘  └──────────┘  └──────────────┘ ││
│  │                                                               ││
│  │  ┌──────────────────────────────────────────────────────────┐││
│  │  │              SKILL DISTILLATION                           │││
│  │  │  pattern memories → candidate skills → user approval     │││
│  │  │  (preserves existing memory→skill pipeline)               │││
│  │  └──────────────────────────────────────────────────────────┘││
│  └──────────────────────────────────────────────────────────────┘│
└───────────────────────────────────────────────────────────────────┘
```

---

## Phase 0: Foundation Hardening (1-2 days)

Strengthen the existing `feat/cognitive-memory-retrieval` branch before building on it.

### Task 0.1: Merge current main into feat branch
- Merge `main` into `feat/cognitive-memory-retrieval`, resolve conflicts
- Ensure all existing tests pass after merge
- Files: `git merge main`, run `pytest tests/tools/test_memory_tool_bm25.py -v`

### Task 0.2: Fix test directory name assumption
- The test checks `_TEST_ROOT.name == "hermes-agent-cognitive"` — make this robust to any checkout directory name
- File: `tests/tools/test_memory_tool_bm25.py`
- Replace hardcoded name check with path-agnostic approach

### Task 0.3: Add memory CLI command
- Add `hermes memory mode [full|contextual]` and `hermes memory stats` commands
- File: `hermes_cli/` — new subcommand module
- Enables users to switch retrieval mode and see memory health without editing config.yaml

### Task 0.4: Ensure `rank-bm25` installation in bootstrap
- Add `rank-bm25` to the Hermes bootstrap/install flow
- Files: `setup-hermes.sh`, `pyproject.toml`
- Or make it a hard dependency (remove optional flag) — BM25 is now core

---

## Phase 1: Memory Types & Storage (3-4 days)

The current system has flat entries in MEMORY.md/USER.md. Introduce typed memory with a structured backend.

### Task 1.1: Define MemoryType enum and entry schema
- File: `agent/cognitive_memory/types.py` (NEW)
- Five types: `episodic`, `semantic`, `procedural`, `working`, `pattern`
- Entry schema: id (ULID), type, title, content, tags (list), importance (0-1), created_at, updated_at, access_count, last_accessed, source_session_id
- Port the ULID generation pattern from claude-code-memory

### Task 1.2: Implement SQLite cognitive memory backend
- File: `agent/cognitive_memory/store.py` (NEW)
- SQLite database at `~/.hermes/cognitive_memory.db`
- Tables: `memories` (with all fields), `memory_tags` (normalized tag storage), `memory_relations`
- FTS5 virtual table on title+content for full-text search
- WAL mode, foreign keys ON
- Schema version tracking for migrations
- CRUD: insert, update, delete, get_by_id, list_by_type, search_fts

### Task 1.3: Migration bridge — existing memory to typed memory
- File: `agent/cognitive_memory/migration.py` (NEW)
- One-time migration: read existing MEMORY.md/USER.md entries, classify by content heuristics
- Regex classifier (port from claude-code-memory's `classifier.ts`):
  - Task/project descriptions → semantic
  - Build/test/deploy commands → procedural
  - Conversation summaries → episodic
  - User preferences/corrections → pattern
- Preserve existing entries as `semantic` type with original content
- Migration is idempotent — run on first startup if cognitive_memory.db doesn't exist

### Task 1.4: Dual-write adapter
- File: `tools/memory_tool.py` — modify MemoryStore
- Write to both existing file-based memory AND cognitive memory store
- Existing `memory` tool API unchanged (add/replace/remove) — cognitive store is write-through
- On read, prefer cognitive store when available
- This ensures backward compatibility with the existing `memory` tool

### Task 1.5: Type-aware retrieval
- File: `agent/cognitive_memory/retrieval.py` (NEW)
- Extend `get_relevant_entries()` to filter by type and apply type-specific scoring
- Episodic gets recency boost, pattern gets importance boost, semantic gets full-text weight
- Port the scoring formula from claude-code-memory:
  `finalScore = textScore × w_text + importance × w_imp + recencyBoost × w_rec + accessBoost × w_acc − contentLengthPenalty`

---

## Phase 2: Semantic Retrieval (2-3 days)

BM25 is keyword-only. Add embedding-based semantic retrieval for conceptual matching.

### Task 2.1: Embedding model integration
- File: `agent/cognitive_memory/embeddings.py` (NEW)
- Use `sentence-transformers` (Python-native, no Node.js dependency)
- Model: `all-MiniLM-L6-v2` (384-dim, same as claude-code-memory, ~90MB)
- Functions: `generate_embedding(text)`, `generate_embeddings(texts)` (batched)
- Lazy loading — model loaded on first use
- Cosine similarity helper
- Graceful fallback if sentence-transformers not installed

### Task 2.2: Vector storage in SQLite
- File: `agent/cognitive_memory/store.py` — extend
- Add `embedding` BLOB column to `memories` table
- Auto-generate embedding on insert/update if title or content changed
- Batch embedding generation for efficiency
- `cosine_similarity_search(embedding, limit, type_filter)` — manual cosine scan (OK for <10K entries)
- OR integrate `sqlite-vec` Python bindings if available for larger scale

### Task 2.3: Hybrid search (BM25 + embeddings)
- File: `agent/cognitive_memory/retrieval.py` — extend
- Parallel FTS5 + vector search, blended scoring (0.4 FTS / 0.6 vector, configurable)
- Overfetch 3× candidates (max 60), then rerank with cross-encoder or score blending
- Fallback to BM25-only when embeddings unavailable
- Preserve the existing BM25-only path for `rank-bm25` only installs

### Task 2.4: Cross-encoder reranking (optional, Phase 2.5)
- File: `agent/cognitive_memory/reranker.py` (NEW)
- Model: `cross-encoder/ms-marco-TinyBERT-L-2-v2` via sentence-transformers
- Applied to top-N candidates from hybrid search for precision
- Configurable: on/off, candidate pool size
- Significant quality boost at moderate latency cost — make it opt-in via config

---

## Phase 3: Ambient Memory Injection (3-4 days)

This is the core behavioral shift — memory becomes ambient, surfacing at the right moments without the agent needing to explicitly search.

### Task 3.1: Pre-turn injection enhancer
- File: `agent/conversation_loop.py` — modify the existing prefetch block
- Before each API call, inject up to N relevant memories (configurable)
- Memory selection by channel:
  - **User query** → semantic search (what does the agent know about this topic?)
  - **Tool about to be used** → procedural lookup (any build/test/deploy memories?)
  - **File being edited** → pattern warnings (any corrections about this file?)
- Relation expansion: when a memory is injected, pull in its strongest neighbours
- This is Hermes' equivalent of claude-code-memory's hook system — but since Hermes is the agent itself, we inject inside the conversation loop rather than via external hooks

### Task 3.2: Post-turn passive extraction
- File: `agent/cognitive_memory/extraction.py` (NEW)
- After each assistant response, extract memory-worthy facts:
  - **User corrections** → pattern memory (e.g., "don't use sudo without asking")
  - **New project facts** → semantic memory (e.g., "this project uses pytest with xdist")
  - **Successful command patterns** → procedural memory (e.g., "build with: npm run build")
  - **Conversation summary** → episodic memory (lightweight, stored at session end)
- Heuristic-based extraction (regex patterns, not LLM calls) — cheap and fast
- LLM-based extraction on a slower cadence (every 10 turns, or at session end) for higher quality

### Task 3.3: Session lifecycle hooks
- File: `agent/cognitive_memory/session.py` (NEW)
- **Session start**: Clear working memories, inject context from git signals (branch, recent commits), apply decay
- **Session end**: Parse conversation for episodic summary, update procedural workflows, detect patterns, accumulate consolidation weight
- **Error context**: When a tool call fails, search memories for relevant troubleshooting
- These are triggered by the agent loop, not external hooks — integrated into `run_conversation` lifecycle

### Task 3.4: Content-length penalty and injection caps
- Per-type injection limits (max 6 semantic, 3 pattern, 2 episodic per turn)
- Content-length penalty: entries > 500 chars get progressive penalty (asymptotic to 0.15)
- Dedup within session: track which memories were already injected, don't repeat
- Session injection cache: `_injected_ids` set, cleared at session start

---

## Phase 4: Relation Graph & Co-Activation (2-3 days)

Memories shouldn't be isolated — they form a graph of relationships that evolves over time.

### Task 4.1: Relation types and storage
- File: `agent/cognitive_memory/relations.py` (NEW)
- Six relation types: `relates_to`, `depends_on`, `contradicts`, `extends`, `implements`, `derived_from`
- Each relation has a `weight` (0-1) that evolves
- `memory_relations` table: source_id, target_id, relation_type, weight, created_at, updated_at
- CRUD: create_relation, delete_relation, get_relations(id), get_graph(id, depth=3, max_nodes=50)

### Task 4.2: Auto-relation on insert
- When a new memory is inserted, find related memories by embedding similarity
- Create up to 5 auto-relations:
  - Same-type near-duplicates → `contradicts`
  - Moderate similarity → `extends`
  - Same context within 24h → `derived_from`
  - Cross-type with shared tags → `relates_to`

### Task 4.3: Co-activation tracking
- File: `agent/cognitive_memory/coactivation.py` (NEW)
- When two memories are injected in the same session, increment co-activation count
- At count ≥ 3, automatically create a `relates_to` relation (weight 0.45)
- On each co-injection, boost relation weight (+0.05) and co-access (+0.10)
- This discovers connections that embeddings miss — e.g., "pytest" and "coverage" injected together repeatedly

### Task 4.4: Relation weight evolution
- **Decay**: -0.005 per session (floor 0.05)
- **Boost on co-injection**: +0.05
- **Boost on co-access**: +0.10
- **Periodic sweep**: Every 20 sessions, scan under-connected memories, find neighbours by embedding, create relations (time-budgeted at 500ms)
- File: `agent/cognitive_memory/store.py` — add `sweep_relations()`, `decay_relation_weights()`, `prune_stale_relations()`

---

## Phase 5: Autonomous Consolidation (4-5 days)

The most complex phase — automatic memory maintenance. This is the "cognitive" part.

### Task 5.1: Session weight accumulation
- File: `agent/cognitive_memory/consolidation.py` (NEW)
- Track session "substance": tool calls × 0.1, files modified × 0.3, memory ops × 0.1, errors × 0.1, meaningful bash commands × 0.1
- Accumulate in `consolidation_weight` session meta
- Trigger consolidation when weight ≥ 10.0, or fallback every 20 sessions

### Task 5.2: Duplicate merging
- For each non-working type, find near-duplicate pairs (cosine distance < 0.08)
- Merge: keep larger content, higher importance, union tags, remap relations from deleted to kept
- Batch operation — process all types in one pass

### Task 5.3: Topic splitting
- For semantic and procedural memories > 500 chars:
  - Split by markdown headers (##, ###), bold sections (**Title:**), numbered lists
  - If ≥ 2 meaningful sections, create separate memories
  - Create `relates_to` sibling relations
  - Delete original
- Prevents monolithic memory bloat

### Task 5.4: Pattern detection via clustering
- Get recent episodic memories (last 30 days), filter to `session-end` tagged
- Agglomerative clustering by cosine similarity (min 0.5, max 0.85)
- Only keep clusters of size ≥ 3
- Quality gates: average intra-cluster similarity ≥ 0.55, Jaccard word overlap on task descriptions ≥ 0.15
- Compute centroid embedding, check against existing patterns (skip if overlap ≥ 0.55)
- Create `pattern` memories with `derived_from` relations to source episodes
- Quality scoring: high-quality clusters → importance 0.8, lower → 0.6

### Task 5.5: Decay and cleanup
- **Importance decay by type** (per session):
  - Episodic: -0.08 (fast decay — session details fade)
  - Semantic: -0.02 (slow — project knowledge persists)
  - Procedural: -0.02 (slow — workflows are durable)
  - Pattern: -0.01 (very slow — learned patterns should stick)
- **Age-out deletion**:
  - Episodic > 90 days AND importance < 0.1 → delete
  - Pattern > 180 days AND importance < 0.1 → delete
  - Semantic/procedural → never age out
- **Stale cleanup**: entries > 60 days, importance < 0.1, no accesses → delete
- **Working memory**: cleared every session start

### Task 5.6: Consolidation scheduler
- Run consolidation check at session start (in the ambient injection path)
- Non-blocking — consolidation runs in background, doesn't delay user interaction
- Progress tracking — save `last_consolidation_session` in session meta
- Configurable: on/off, thresholds, decay rates

---

## Phase 6: Skills Distillation (2-3 days)

The bridge between cognitive memory and the existing skill pipeline.

### Task 6.1: Pattern-to-skill candidate detection
- File: `agent/cognitive_memory/skills_distillation.py` (NEW)
- When a `pattern` memory reaches importance ≥ 0.8 AND has been accessed ≥ 5 times:
  - Flag as a skill candidate
  - Auto-generate a skill name from pattern title
  - Create a draft SKILL.md from pattern content
- Present to user for approval (via `clarify` tool or passive notification)
- On approval, create skill via `skill_manage(action='create')`
- This automates the existing "nudge → background review → skill" pipeline

### Task 6.2: Procedural memory → skill templates
- When a `procedural` memory accumulates ≥ 3 related commands for the same category:
  - Generate a skill template with numbered steps, exact commands, pitfalls section
  - Offer to create the skill
- This reduces the manual skill creation burden

### Task 6.3: Memory usage reduction on skill creation
- When a pattern or procedural memory is distilled into a skill:
  - Reduce its importance (memory is now "stored" in the skill)
  - Add `derived_from` relation pointing memory → skill
  - Optionally delete the memory after cooldown period

---

## Phase 7: Configuration, Monitoring & Polish (2 days)

### Task 7.1: Full config surface
- File: `hermes_cli/config.py` — extend memory config section
- All tunable parameters: retrieval mode, max retrieved per turn, decay rates, consolidation thresholds, embedding model, reranker on/off, injection caps per type
- Sensible defaults for all — works out of the box

### Task 7.2: CLI dashboard
- `hermes memory stats` — counts by type, embedding coverage, staleness, age distribution, health score
- `hermes memory graph [id]` — display memory relation graph (text-based in terminal)
- `hermes memory search <query>` — direct search from CLI
- `hermes memory consolidate` — manual consolidation trigger

### Task 7.3: Health monitoring
- File: `agent/cognitive_memory/health.py` (NEW)
- `get_health_stats()` — comprehensive metrics (port from claude-code-memory's memory_health tool)
- Surface health in agent logs at session start
- Alert on: high staleness, low embedding coverage, approaching consolidation threshold

### Task 7.4: Integration tests
- End-to-end test: start session, inject memories, verify retrieval, verify consolidation
- Test all five memory types through their lifecycle
- Test migration from file-based to cognitive memory
- Test skill distillation pipeline

---

## Files Inventory

### New files to create:
```
agent/cognitive_memory/
├── __init__.py
├── types.py              # MemoryType enum, entry schema, ULID generation
├── store.py              # SQLite backend, CRUD, FTS5, vector storage
├── migration.py          # One-time migration from file-based to typed memory
├── retrieval.py          # Hybrid search, scoring, type-aware retrieval
├── embeddings.py         # sentence-transformers integration, cosine similarity
├── reranker.py           # Cross-encoder reranking (optional)
├── extraction.py         # Passive fact extraction from conversation turns
├── session.py            # Session lifecycle hooks (start, end, error)
├── relations.py          # Relation types, CRUD, auto-relation on insert
├── coactivation.py       # Co-activation tracking and promotion
├── consolidation.py      # Merge, split, pattern detection, decay, cleanup
├── skills_distillation.py # Pattern → skill candidate detection
├── health.py             # Health stats, monitoring
└── config.py             # Config defaults and validation
```

### Modified files:
```
tools/memory_tool.py          # Dual-write adapter, typed entry support
agent/conversation_loop.py    # Ambient injection (pre-turn, post-turn, error)
agent/system_prompt.py        # Conditional full dump vs contextual
agent/agent_init.py           # Initialize cognitive memory store
hermes_cli/config.py          # Full config surface
hermes_cli/                   # New `hermes memory` commands
pyproject.toml                # Dependencies: sentence-transformers, rank-bm25
setup-hermes.sh               # Bootstrap cognitive memory dependencies
```

### Files NOT touched:
```
plugins/memory/               # External providers remain unchanged
agent/memory_provider.py      # MemoryProvider ABC unchanged
agent/memory_manager.py       # MemoryManager unchanged
agent/background_review.py    # Existing nudge workflow preserved
hermes_state.py               # Session DB unchanged (separate concern)
skills/                       # Existing skill system unchanged
```

---

## Execution Order & Dependencies

```
Phase 0 (Foundation)
   │
   ▼
Phase 1 (Memory Types & Storage) ── prerequisite for everything
   │
   ├── Phase 2 (Semantic Retrieval) ── depends on typed storage
   │       │
   │       ▼
   ├── Phase 3 (Ambient Injection) ── depends on retrieval
   │       │
   │       ▼
   ├── Phase 4 (Relations & Graph) ── parallel with Phase 3, shares storage
   │       │
   │       ▼
   └────── Phase 5 (Consolidation) ── depends on all above
              │
              ▼
           Phase 6 (Skills Distillation) ── depends on consolidation patterns
              │
              ▼
           Phase 7 (Config & Polish) ── wraps everything
```

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| **Performance**: Embedding generation adds latency | Lazy loading, batching, background processing. BM25 fallback always available |
| **Storage bloat**: SQLite + embeddings can grow | Consolidation keeps it in check. Age-out policies prevent unbounded growth |
| **Model confusion**: Injecting memory context every turn could distract the model | Per-type caps, content-length penalty, dedup within session prevent overwhelming context |
| **Migration pain**: Existing users with large MEMORY.md | One-time migration is idempotent and preserves all existing entries as semantic type |
| **Dependency weight**: sentence-transformers is ~100MB+ | Make it optional — BM25-only mode works without it |
| **Backward compatibility**: Don't break existing memory tool users | Dual-write adapter ensures both stores are updated. Existing `memory` tool API unchanged |

---

## Open Questions

1. **SQLite vs file-based as primary store**: Should cognitive memory fully replace MEMORY.md, or remain a parallel system? Recommendation: parallel with cognitive as primary for retrieval, file-based as durable backup. Over time, file-based becomes a serialization format, not the active store.

2. **Embedding model choice**: `all-MiniLM-L6-v2` (384-dim) matches claude-code-memory. Python-native via sentence-transformers. Alternative: OpenAI embeddings API (higher quality, costs money, adds network dependency). Recommendation: start with local MiniLM, add OpenAI as optional config.

3. **Cross-encoder necessity**: Adds significant quality but ~200ms latency per query and ~500MB RAM. Recommendation: make it opt-in, default off. The hybrid BM25+embedding search is good enough for initial launch.

4. **Consolidation frequency**: How aggressive should auto-consolidation be? claude-code-memory uses session weight (10.0) with 20-session fallback. For Hermes (single long session vs many short sessions), we may need time-based fallback too (e.g., every 6 hours of active conversation).

5. **Skill distillation approval**: Should pattern→skill be fully automatic or require user confirmation? Recommendation: flag + notify, require user approval before creating. Fully automatic skill creation is too risky.

---

> **Plan created:** 2026-05-24
> **Source analysis:** claude-code-memory v1.2 + hermes-agent feat/cognitive-memory-retrieval
> **Total phases:** 8 (0 through 7)
> **Estimated total effort:** 18-24 days
> **Implementation order:** Sequential by phase, tasks within a phase can be parallelized
