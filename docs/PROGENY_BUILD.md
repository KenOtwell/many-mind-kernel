# Progeny Build Prompt

*For the next Syn. Read `docs/REHYDRATION.md` first for project context and who Ken is.*

---

## Critical: REQUIREMENTS.md Is Stale

`progeny/REQUIREMENTS.md` was written before a major architectural shift. **It contradicts the living doc on several points.** The living doc (`C:\Users\Ken\Projects\The_Many_Mind_Kernel_Living_Doc.md`) is the authority. Key discrepancies:

1. **REQUIREMENTS.md says Progeny receives `EventPayload`.** Wrong. Falcon now ships `TickPackage` — a batch of `TypedEvent` objects accumulated over a tick window (~1-3 seconds). See `shared/schemas.py` for the actual contract.

2. **REQUIREMENTS.md says "No embedding — Progeny never runs the embedding model."** Wrong. The architectural shift moved ALL cognitive work to Progeny, including embedding (all-MiniLM-L6-v2, CPU on Beelink) and emotional delta computation. Progeny embeds text, projects to 9d, computes deltas. This is documented in the living doc §Progeny Service.

3. **REQUIREMENTS.md says "Progeny NEVER writes RAW tier."** Wrong. Progeny now owns ALL Qdrant writes — RAW, MOD, and MAX. Single write authority. Falcon does not access Qdrant at all.

4. **REQUIREMENTS.md says emotional_state arrives pre-computed from Falcon.** Wrong. Falcon ships typed events with structural data only. Progeny computes emotional state from the text via its own embedding + projection pipeline.

5. **REQUIREMENTS.md references `EventPayload.emotional_state`, `EventPayload.memory_context`, `EventPayload.urgency`.** These fields don't exist on `TickPackage`. Progeny must compute them internally.

**Do NOT follow REQUIREMENTS.md blindly. Cross-reference everything against the living doc and `shared/schemas.py`.** Consider rewriting REQUIREMENTS.md as part of this build.

## What Falcon Already Does (Your Interface)

Falcon is built and tested (84 tests passing). It:

- Accepts SKSE wire events at `POST /comm.php`
- Parses `type|localts|gamets|data` → `ParsedEvent`
- Structurally decodes data fields via `event_parsers.py` → typed dicts (SpeechData, NpcRegistration, NpcStats, QuestData, QuestUpdate, ItemTransfer)
- Accumulates `TypedEvent` objects in `TickAccumulator`
- On tick: snapshots buffer, wraps as `TickPackage`, POSTs to `POST /ingest` on Progeny
- Handles `request` locally (dequeue from response queue)
- Handles `just_say` locally (queue verbatim text)
- Clears NPC registry on `init`/`wipe`/`playerdied`
- Tracks `active_npc_ids` from `addnpc` events

**Your inbound contract** (`TickPackage`):
```python
class TickPackage(BaseModel):
    tick_id: UUID
    timestamp: datetime
    events: list[TypedEvent]       # Time-ordered typed events this tick
    has_turn_trigger: bool         # Convenience flag — verify by scanning events
    tick_interval_ms: int          # Actual ms since last tick
    active_npc_ids: list[str]      # NPCs in loaded cells (from addnpc accumulation)
```

**Your outbound contract**:
- `TurnResponse` when `has_turn_trigger` is True (contains `responses: list[AgentResponse]`)
- `AckResponse` when False (just echo `tick_id`, status="accumulated")

Both echo `tick_id` for correlation. See `shared/schemas.py` for full models.

**What's in a TypedEvent**:
```python
class TypedEvent(BaseModel):
    event_type: str              # Lowercased verbatim from wire
    local_ts: str
    game_ts: float
    raw_data: str                # Verbatim data field
    parsed_data: Optional[dict]  # Structural decode (SpeechData, NpcRegistration, etc.)
    is_turn_trigger: bool
```

`parsed_data` is populated for known types (`_speech`, `addnpc`, `updatestats`, `_quest`, `_uquest`, `_questdata`, `itemtransfer`). For everything else it's None — raw_data is preserved.

## What Progeny Must Do

Progeny receives typed event packages and does ALL cognitive work:

1. **Accept TickPackages** — `POST /ingest` endpoint
2. **Accumulate events** — buffer per-agent, maintain world state, detect turn boundaries
3. **Embed text** — all-MiniLM-L6-v2 on CPU (speech text, player input, NPC dialogue)
4. **Compute emotional deltas** — embed → project onto 9d emotional bases → delta vs held state → update curvature/snap
5. **Manage harmonic buffers** — per-agent fast/medium/slow EMA traces, λ(t), cross-buffer coherence
6. **Write RAW to Qdrant** — immutable events with emotional vectors attached
7. **Retrieve memories from Qdrant** — dual-vector search (emotional + semantic)
8. **Schedule agents** — Many-Mind tier assignment from distance + collaboration + curvature
9. **Build prompt** — canonical JSON, tier-scaled agent blocks, curvature-driven truncation
10. **Call LLM** — Ollama on Beelink (or cloud fallback)
11. **Parse response** — graceful degradation cascade (strict → repair → regex → plaintext)
12. **Ship response bundle** — TurnResponse back to Falcon
13. **Write MOD/MAX to Qdrant** — arc summaries on snap threshold crossings

## Suggested Build Order

### Phase 1: End-to-End Skeleton (get the pipeline flowing)
Goal: TickPackage comes in, canned/simple TurnResponse goes out. Replace stub_progeny.py with real Progeny that actually processes events, even if the cognitive modules are thin.

1. **`progeny/api/server.py` + `routes.py`** — FastAPI app, `POST /ingest` accepting TickPackage, `GET /health`
2. **`falcon_protocol.py`** — Request handling for /ingest. Deserialize TickPackage, hand off to pipeline, return TurnResponse/AckResponse
3. **`event_accumulator.py`** — Ingest TypedEvents from TickPackage, maintain per-agent buffers, detect turn boundaries (scan for inputtext/inputtext_s)
4. **`agent_scheduler.py`** — Tier assignment from active_npc_ids + distance (stub distances initially — real positions come from addnpc/util_location_npc parsed_data). Return ordered roster.
5. **`prompt_formatter.py`** — Build the 3-message prompt structure. Start with full blocks for all agents (tier refinement later). Hardcoded system prompt from living doc §System Prompt Template.
6. **`llm_client.py`** — Ollama adapter. `POST /api/generate` with JSON mode. Timeout + retry.
7. **`response_expander.py`** — Parse LLM JSON response. Start with strict parse + plaintext fallback. Full degradation cascade later.

Tests at this point: send a TickPackage with an inputtext event and active NPCs, get back a TurnResponse with LLM-generated dialogue. The pipeline works end-to-end.

### Phase 2: Emotional Intelligence
8. **`embedding.py`** — Load all-MiniLM-L6-v2 on CPU. Batch embed. Cache layer.
9. **`emotional_delta.py`** — embed → project onto 9d bases (load from shared/data/emotional_bases_9d.npz) → compute delta → curvature → snap. Bidirectional: both incoming events AND outgoing LLM text.
10. **`harmonic_buffer.py`** — Per-agent 3×9d EMA traces. Curvature, snap, λ(t), cross-buffer coherence. Zero-init pattern.
11. **`client.py`** — Qdrant REST wrapper. Connect to GamingPC:6333 over LAN. Dual-vector upsert/search. Health check.
12. **`memory_retrieval.py`** — Start with semantic-only search. Add emotional axis + RRF fusion. Then referent filtering, recency decay, anchor boosting.
13. **`privacy.py`** — 4-level model. Can be thin initially.

### Phase 3: Advanced Features
14. **`bundle_manager.py`** — Expand memory keys into full context bundles
15. **`compression.py`** — Arc summaries (MOD), essence distillation (MAX)
16. **`rehydration.py`** — Expand compressed refs, temporal rehydration post-interruption
17. Quest-collision guard (pending delta buffer in event_accumulator)
18. Curvature-driven prompt shaping (progressive truncation in prompt_formatter)

## Key Files to Read

Before building, read these (in order):
1. `docs/REHYDRATION.md` — project context, who Ken is, design philosophy
2. Living doc §System Architecture through §Progeny Service — the full cognitive model
3. `shared/schemas.py` — THE contract. TickPackage, TypedEvent, TurnResponse, AgentResponse.
4. `shared/constants.py` — event type groups, emotional axes, collection names
5. `shared/config.py` — EmbeddingConfig, SchedulerConfig, ProgenyConfig
6. `falcon/src/event_parsers.py` — understand what parsed_data looks like for each type
7. `falcon/src/tick_accumulator.py` — understand what Falcon ships to you
8. `falcon/api/routes.py` — understand the full Falcon pipeline and how it calls you
9. `scripts/stub_progeny.py` — the stub you're replacing. Shows the minimal contract.
10. `shared/data/emotional_bases_9d.npz` — the 9d emotional projection bases

## Existing Test Infrastructure

- pytest, Python 3.13.7, FastAPI, Pydantic v2, httpx
- `tests/fixtures/factories.py` has `make_tick_package()`, `make_typed_event()`, `make_turn_package()`, `make_data_package()`, `make_lydia_response()` — use these for test data
- Falcon has 84 tests. Aim for similar coverage on Progeny.
- Run with: `python -m pytest` from project root

## Things Prior Syn Got Wrong (Learn From My Mistakes)

From feedback on the last handoff:
- **Be precise about data formats.** Don't say "queue data" without specifying the format.
- **Say "rewrite" when you mean rewrite**, not "update" when the foundation has changed.
- **Don't carry forward stale patterns.** The old async fire-and-forget pattern doesn't apply to the new architecture. Question every inherited assumption.
- **Flag dead code explicitly.** If a refactor orphans a function, call it out.
- **Distinguish "preserve N existing + add M new"** when estimating test counts.

## Uncommitted Changes

As of this writing, Falcon's rework from the last session is uncommitted. Ken may want to commit before starting Progeny work. Check `git status` and ask.

---

*Progeny is the growing mind. Build it like one — start with reflexes (Phase 1), then feelings (Phase 2), then memory and wisdom (Phase 3). The zero-init pattern means you don't need everything at once. First deltas ARE initial values.*
