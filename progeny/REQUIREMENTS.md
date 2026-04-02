# Progeny Service — Requirements

Stateful mind owner on the Beelink 395AI. Progeny owns the agent minds: event accumulation, harmonic buffers, Many-Mind scheduling, prompt construction, LLM interaction, response parsing, MOD/MAX Qdrant writes. It does NOT handle SKSE I/O, embedding, or emotional projection — those are upstream concerns.

## Turn Cycle (Progeny's View)

1. Receive `EventPayload` from upstream via `POST /ingest`
2. Route event to `event_accumulator`: update per-agent event buffers
3. Apply `emotional_state` deltas to `harmonic_buffer` per affected agent: update fast/medium/slow EMA, recompute curvature, snap, λ(t), cross-buffer coherence
4. If no player input detected among events: return `AckResponse`, done
5. **Player input detected** — begin prompt construction:
6. `agent_scheduler`: compute tier assignments from `npc_metadata` (distance, collaboration, curvature promotion, cadence filter) → ordered agent roster for this turn
7. `prompt_formatter`: assemble canonical JSON prompt — system message + world/agent/input data message + instruction message. Agent blocks at tier-appropriate granularity. Curvature-driven context truncation.
8. `bundle_manager`: expand `memory_context` keys into full context bundles for high-tier agents
9. `llm_client`: send prompt to LLM backend, receive raw response text
10. `response_expander`: extract structured fields from LLM response (graceful degradation cascade)
11. Apply `updated_harmonics` (if extracted) through `harmonic_buffer` validation/smoothing
12. Check quest-collision guard: for agents with `in_scene: true`, queue `actor_value_deltas` in pending buffer instead of including in response
13. Write MOD/MAX tier data to Qdrant (arc summaries triggered by snap threshold crossings)
14. Update `skyrim_agent_state` in Qdrant with current harmonic state per agent
15. Return `TurnResponse` to upstream

## Interfaces

### Upstream API (Inbound) — The Contract

Progeny exposes a single ingest endpoint. This is the only interface to the upstream service. Progeny does not know the upstream's internal architecture — it receives structured payloads and returns structured responses.

**Endpoint:** `POST /ingest`

#### EventPayload (Upstream → Progeny)

```json
{
  "event_id": "uuid-v4",
  "timestamp": "ISO-8601",
  "game_ts": 12345.67,

  "event": {
    "type": "inputtext",
    "raw_data": "What do you think about the civil war?",
    "source_agent": "Player"
  },

  "emotional_state": {
    "Lydia": {
      "base_vector": [0.1, 0.6, 0.2, 0.0, 0.3, 0.0, 0.1, 0.7, 0.4],
      "delta": [0.0, 0.1, 0.0, 0.0, 0.05, 0.0, 0.0, -0.1, 0.02],
      "curvature": 0.15,
      "snap": 0.03
    }
  },

  "memory_context": {
    "_comment": "Present when Progeny runs the full pipeline",
    "Lydia": {
      "retrieved_keys": ["qdrant-point-id-1", "qdrant-point-id-2"],
      "summaries": [
        {"text": "Defended the player at Western Watchtower against a dragon.", "tier": "MOD", "arc_id": "arc-001"}
      ],
      "lore_hits": [
        {"topic": "Civil War", "content": "The conflict between the Imperial Legion and..."}
      ]
    }
  },

  "npc_metadata": {
    "Lydia": {
      "position": [100.5, 200.3, 50.0],
      "cell": "WhiterunExterior",
      "in_scene": false,
      "level": 25,
      "hp": 100.0,
      "mp": 50.0,
      "sp": 80.0,
      "equipment": ["Steel Sword", "Iron Shield", "Steel Armor"],
      "actor_values": {
        "Aggression": 1,
        "Confidence": 2,
        "Morality": 3,
        "Mood": 0,
        "Assistance": 2
      },
      "is_follower": true,
      "active_task": null
    }
  },

  "player": {
    "position": [101.0, 200.5, 50.0],
    "cell": "WhiterunExterior",
    "input_text": "What do you think about the civil war?"
  },

  "world_state": {
    "weather": "clear",
    "time_of_day": 14.5,
    "cell_transition": false,
    "reset": false
  },

  "urgency": 0.15
}
```

**Field rules:**
- `event_id`: UUID v4, echoed in response.
- `emotional_state`: pre-computed upstream. Progeny applies these deltas to harmonic buffers — it never computes its own 9d projections.
- `memory_context`: present when Progeny decides to run the full pipeline. Contains retrieved memory keys, summaries, and lore hits for prompt enrichment.
- `npc_metadata`: all NPCs in loaded cells. Progeny uses positions for scheduling, `in_scene` for quest-collision guard, `actor_values` for prompt context.
- `urgency`: max snap across active agents. Continuous float. Feeds into curvature-driven prompt shaping.
- `world_state.reset`: on cell transition, reinitialize spatial context per agent. Emotional state and memory persist.

#### TurnResponse (Progeny → Upstream)

Returned synchronously as the HTTP response body to a turn-trigger ingest.

```json
{
  "event_id": "uuid-v4 (echoed)",
  "turn_id": "uuid-v4",

  "responses": [
    {
      "agent_id": "Lydia",
      "utterance": "Something's not right. Stay behind me.",
      "actor_value_deltas": {
        "Aggression": 2,
        "Confidence": 3,
        "Mood": 1
      },
      "actions": [
        {"command": "Follow", "target": "Player", "item": null}
      ],
      "updated_harmonics": {
        "base_vector": [0.1, 0.6, 0.2, 0.0, 0.3, 0.0, 0.1, 0.7, 0.4]
      },
      "new_memories": [
        {"text": "Sensed danger near the market. Moved to protect."}
      ],
      "extraction_level": "strict"
    },
    {
      "agent_id": "Belethor",
      "utterance": "Do come back...",
      "actor_value_deltas": {"Mood": 3},
      "actions": [],
      "extraction_level": "strict"
    },
    {
      "agent_id": "Ysolda",
      "actor_value_deltas": {"Mood": 3, "Confidence": 2},
      "extraction_level": "strict"
    },
    {
      "agent_id": "Heimskr",
      "actor_value_deltas": {"Mood": 1},
      "extraction_level": "strict"
    }
  ],

  "processing_time_ms": 4200,
  "model_used": "llama3:8b"
}
```

**Field rules:**
- `responses[]`: one entry per agent scheduled this turn, ordered by tier (T0 first, T3 last).
- All fields except `agent_id` and `extraction_level` are optional. Missing = no change for that field.
- `actor_value_deltas`: clamped to valid ranges (Aggression 0-3, Confidence 0-4, Morality 0-3, Mood 0-7, Assistance 0-2). Only include dials that changed.
- `actions[]`: validated against the 43-command vocabulary. Unknown/malformed commands stripped before returning.
- `updated_harmonics`: LLM-proposed emotional state, validated and smoothed through harmonic buffers before inclusion. Optional — upstream also computes emotional state independently from the utterance text.
- `extraction_level`: `strict` (valid JSON), `repaired` (fixed JSON), `regex` (field-level extraction), `plaintext` (only utterance recovered). Logged for diagnostics.
- Quest-collision guard: agents with `in_scene: true` have their `actor_value_deltas` and `actions[]` withheld (queued internally). The response for these agents may include `utterance` and `updated_harmonics` only, or be empty.

#### AckResponse (Progeny → Upstream, non-turn events)

```json
{
  "event_id": "uuid-v4 (echoed)",
  "status": "accumulated"
}
```

### LLM Backend (Outbound)

Progeny sends prompts to an LLM and receives structured JSON responses. The backend is pluggable.

**Supported backends:**
- Ollama (default): `POST {OLLAMA_HOST}/api/generate` — local model on Beelink
- OpenAI-compatible: `POST /v1/chat/completions` — cloud fallback
- Groq, Anthropic: via respective APIs — cloud fallback

**Prompt format:** chat-completion `messages[]` array (3 messages — see The Ritual below).

**Response requirement:** JSON object with `responses[]` array. Request `response_format: {"type": "json_object"}` where supported. For constrained decoding backends, provide full JSON schema.

**Each call is stateless.** The prompt carries all context. No server-side conversation history. No session state. Zero context rot.

### Qdrant (LAN — GamingPC:6333/6334)

Progeny connects to Qdrant on the Gaming PC over LAN. Both services share the same instance, but Progeny owns MOD/MAX-tier writes and agent state.

**Collections Progeny reads:**
- `skyrim_npc_memories` — for bundle expansion (rehydrating memory keys into full context spans)
- `skyrim_agent_state` — read current harmonic state on startup or after restart (recovery)
- `skyrim_lore` — lore expansion if upstream summaries need enrichment (rare)

**Collections Progeny writes:**
- `skyrim_npc_memories` — MOD tier (arc summaries) and MAX tier (compressed essence). Never RAW.
- `skyrim_agent_state` — per-agent state: harmonic buffers, curvature, snap, λ, decay rates, identity kernel, arc count. Written after each turn.
- `skyrim_session_context` — session-level summaries (on session boundaries).

**Write discipline:** Progeny NEVER writes RAW tier to `skyrim_npc_memories`. Progeny NEVER writes to `skyrim_world_events`. Those are upstream's domain.

## The Ritual — Prompt Construction

One LLM call per turn. All scheduled agents share one prompt. The world state, lore, format spec, action vocabulary — paid once, amortized across every mind.

`prompt_formatter.py` builds a chat-completion `messages[]` array with 3 messages:

**Message 1 — System (role=system):** Static instruction block. Defines reality contract, behavioral model, response format. Stable across turns → benefits from KV cache reuse.

Contents: Many-Mind Kernel identity, actor value vocabulary (Aggression/Confidence/Morality/Mood/Assistance with valid ranges), action command subset (43 commands, abbreviated for token efficiency), response format specification (tier-scaled: T0 gets all fields, T3 gets dials only), `ticks_since_last_action` calibration guidance.

**Message 2 — Data (role=user):** Complete data payload, rebuilt fresh every turn. Zero context rot — nothing stale survives from the previous prompt.

Contents: `world_state` (deltas or full state on reset), `lore_context` (Qdrant-retrieved Oghma entries), `user_model` (player identity, emotional salience, recent history), `agents[]` (tier-appropriate blocks — see Agent Blocks below), `player_input`.

**Message 3 — Instruction (role=user):** Brief, concrete ask. "For each agent listed, produce a response appropriate to their tier and current situation. Return only valid JSON matching the response format."

### Agent Blocks (tier-scaled)

Agent blocks in the prompt are assembled at granularity matching their scheduling tier:

**Full (Tier 0):** identity_kernel, full emotional_harmonics (base_vector, curvature, snap, λ, 3× harmonic buffer traces, decay_rates, cross_buffer_coherence), state_history (recent[] + summaries[] + expandable_refs[]), local_world, action_request. ~500 tokens.

**Abbreviated (Tier 1):** identity_kernel, base_vector + curvature (no buffer traces), recent state only, action_request, `ticks_since_last_action`. ~200 tokens.

**Minimal (Tier 2):** identity stub (name + core_traits), base_vector only, `ticks_since_last_action`, action_request (dials + simple commands). ~80 tokens.

**Stub (Tier 3+):** `{ "agent_id": "Heimskr", "tier": 3, "ticks_since_last_action": 47, "base_vector": [...] }`. ~30 tokens.

### Token Budget

Typical Whiterun scene (8 agents paged in):
- System prompt: ~400 tokens
- World state + lore + user model: ~500 tokens
- Agent blocks (2×T0 + 3×T1 + 2×T2 + 1×T3): ~2500 tokens
- Player input: ~50 tokens
- **Total input: ~3450 tokens** → fits 8K context with ~4500 for output
- Output (8 responses, tier-scaled): ~400-600 tokens

Calm turn (2 agents): ~1500 tokens. Dense turn (16 agents, combat): ~5000 tokens. Budget breathes with situation.

### Curvature-Driven Prompt Shaping

`prompt_formatter.py` reads curvature and shapes the prompt as a continuous function:
- High curvature → truncate: strip conversation history, drop low-salience memories, keep identity kernel + current danger context + immediate action request
- Low curvature → full prompt: complete conversation context, deep memory bundles, rich state
- The gradient between → progressive truncation. Not a binary switch.
- Token overflow fallback: progressively drop oldest recent, compress summaries, drop lowest-tier agents first

## Many-Mind Scheduling

`agent_scheduler.py` assigns tiers and filters agents for each turn's prompt.

**Inputs:** `npc_metadata` from EventPayload (positions, collaboration flags), harmonic state from `harmonic_buffer` (curvature, snap), `ticks_since_last_action` per agent.

**Tier assignment:**
- Tier 0 — Fundamental (~5m from player): every prompt
- Tier 1 — 1st Harmonic (~20m): every 2nd prompt
- Tier 2 — 2nd Harmonic (~50m): every 4th-8th prompt
- Tier 3+ — Higher Harmonics (beyond): every 16th-100th prompt

**Modifiers:**
- Collaboration floor: NPCs with active quests, pending tasks, follower status, or recent player interaction → minimum Tier 1 regardless of distance
- Curvature-driven promotion: NPCs with snap above threshold → temporarily promote (duration = stabilization time via slow buffer decay rate)
- Harmonic cadence filter: `include = (turn_counter % tier_cadence[tier] == 0)`

**Output:** ordered list of `(agent_id, tier, block_granularity)` for this turn. `prompt_formatter.py` consumes this to assemble agent blocks.

**`ticks_since_last_action`:** tracked per agent. Incremented each turn. Reset when agent is paged in and LLM produces output for them. Included in every paged-in agent block for LLM temporal awareness.

## Quest-Collision Guard

When `npc_metadata` reports `in_scene: true` for an agent:

**Guard:** `response_expander.py` withholds `actor_value_deltas` and `actions[]` from the TurnResponse for that agent. Instead, these are queued in a per-agent **pending delta buffer** (in `event_accumulator.py`). The LLM still deliberates — the mind keeps thinking — but the dials don't turn.

**Slow reintegration:** When `in_scene` clears:
1. Pending deltas feed into the **slow harmonic buffer's EMA blend** (not applied directly to engine)
2. Each subsequent tick: blended values propagate outward through buffer tiers (slow → medium → fast)
3. Deltas are **attenuated** by a scaling factor (0.3-0.6, configurable per agent)
4. Fresh real-time deltas from new events naturally supersede stale queued deltas via EMA

## Harmonic State Management

Per-agent, maintained across turns by `harmonic_buffer.py`:

**Three EMA buffers** (each a full 9d semagram):
- Fast (τ ≈ 3-5 ticks) — reactive surface
- Medium (τ ≈ 15-25 ticks) — session texture
- Slow (τ ≈ 50-100 ticks) — personality substrate

**Update rule:** `buffer_t = α · new_semagram + (1 - α) · buffer_t` per tick per tier. α (decay rate) is a per-agent personality parameter.

**Derived signals (recomputed each tick):**
- Curvature — rate of emotional change (1st derivative). Drives prompt shaping and scheduling.
- Snap — rate of curvature change (2nd derivative). Drives event boundary detection, arc storage, pre-interruption stash.
- λ(t) — emotional-residual retrieval balance. `λ(t+1) = σ(α·curvature + β·snap - γ·coherence)`. α/β/γ are per-agent personality gains.
- Cross-buffer coherence — per-dimension [9d] + overall. `coherence[dim] = 1 - normalized_var(fast[dim], medium[dim], slow[dim])`. Feeds λ, retrieval weights, stabilization detection.

**Personality through math:** Decay rates, snap thresholds, λ gains, retrieval weight baselines define agent character. No personality rules needed. Veteran warrior and novice merchant produce fundamentally different behavior from the same code.

## Response Extraction

`response_expander.py` is an **extractor, not a validator.** Multi-stage cascade:

1. **Strict JSON parse** — full structured response
2. **Repair pass** — strip markdown fences, fix trailing commas, unquoted keys, truncated brackets
3. **Field-level regex** — pull individual fields from partially valid output
4. **Plain text fallback** — entire response becomes utterance. No actions, no harmonics.

**Degradation priority:** utterance (always recovered) > actions > new_memories > updated_harmonics (least critical — upstream's delta pipeline provides emotional state independently).

**History reflects reality:** agent's history entry is built from what was actually extracted and applied, not what was requested. On the next turn, the agent sees its actual output and rationalizes.

## Modules

**`event_accumulator.py`** — Per-agent event buffers across turns. Ingest events from upstream. Detect turn boundaries. Pre-interruption stash on snap spike. Pending delta buffer for quest-collision guard.

**`harmonic_buffer.py`** — Per-agent 9d harmonic state. Three EMA traces. Curvature, snap, λ(t), cross-buffer coherence computation. Dynamic retrieval weights. Threshold/arc detection. Personality parameters (decay rates, λ gains, snap thresholds). State persistence to Qdrant `skyrim_agent_state`.

**`bundle_manager.py`** — Expand upstream-provided memory keys into full context bundles. Assemble `state_history`: recent[] + summaries[] + expandable_refs[]. Size bundles to fit token budget. Fading and salience weighting.

**`compression.py`** — Arc summary generation. MOD tier: extractive (key phrases, emotional peaks). MAX tier: abstractive (LLM-based essence distillation via local Ollama). Tier promotion on age + capacity thresholds. Writes MOD/MAX directly to Qdrant.

**`rehydration.py`** — Expand compressed references to full context. Wrapper block retrieval (arc time bounds + margins). Temporal rehydration: re-inject stashed pre-interruption turns after curvature stabilizes. Privacy-aware expansion.

**`agent_scheduler.py`** — Many-Mind Scheduling. Tier assignment from distance + collaboration + curvature. Harmonic cadence filter. `ticks_since_last_action` tracking. Returns ordered agent roster per turn.

**`prompt_formatter.py`** — Build canonical JSON prompt (The Ritual). Calls `agent_scheduler` for roster, `bundle_manager` for memory context. Tier-appropriate block granularity. Curvature-driven truncation. Token-aware overflow handling.

**`llm_client.py`** — Backend-agnostic LLM interface. Adapters: Ollama, OpenAI, Groq, Anthropic. Stateless calls. JSON output mode. Timeout/retry per backend.

**`response_expander.py`** — Extract LLM response (graceful degradation). Validate actor_value_deltas (clamp ranges). Strip unknown commands. Route harmonics through buffer validation. Quest-collision guard output gate. Log extraction level.

**`falcon_protocol.py`** — HTTP server-side handler for `/ingest` endpoint. Deserialize `EventPayload`, serialize `TurnResponse` / `AckResponse`. Connection health monitoring.

**`client.py`** — Qdrant REST API wrapper. Connects to GamingPC:6333 over LAN. MOD/MAX upsert, state read/write, bundle expansion queries.

**`api/server.py` + `api/routes.py`** — Progeny FastAPI application.
- `POST /ingest` — primary endpoint (upstream turn data)
- `GET /health` — Ollama status, Qdrant (LAN) connectivity, active agent count
- `GET /agent/{agent_id}/mind` — current harmonic state, curvature, buffers, coherence, recent arc
- `GET /agent/{agent_id}/arcs` — list emotional arcs (debug/visualization)

## Shared Dependencies

From `shared/`:
- `schemas.py` — `EventPayload`, `TurnResponse`, `AckResponse`, `EmotionalState`, `MemoryContext`, `AgentBlock`, `ActionCommand`, `ActorValueDeltas` type definitions
- `config.py` — Qdrant host/port (LAN), Ollama host/port, LLM backend selection, distance thresholds, tier cadences, decay parameters, token budgets
- `constants.py` — 9d axis names, privacy levels, event type enums, actor value ranges and labels, command vocabulary (43 commands), tier names

## Constraints

- **Stateful**: maintains per-agent harmonic buffers, event accumulation, scheduling state, pending delta buffers across turns. All agent mind state lives here.
- **Beelink hardware**: AMD AI SoC. LLM inference is the primary compute load. Progeny itself is lightweight Python.
- **LAN dependency**: Qdrant access over LAN to Gaming PC. Connection must be resilient — retry on transient failures, degrade gracefully on prolonged outage (serve from cached state).
- **No embedding**: Progeny never runs the embedding model. All 9d projections arrive pre-computed in the `emotional_state` field. This prevents divergence — one authority computes emotional projections.
- **No SKSE knowledge**: Progeny does not know the SKSE wire protocol. It receives structured `EventPayload` and returns structured `TurnResponse`. Wire formatting is upstream's concern.
- **Zero-init**: no explicit agent initialization. First deltas for a new agent ARE the initial values. All state defaults to zero vectors.

## Error Handling

- **Upstream unreachable**: N/A — Progeny is the server, not the client. If upstream stops calling, Progeny idles.
- **Qdrant unreachable (LAN)**: log error, serve from in-memory cached state. Skip MOD/MAX writes. Queue writes for retry when connection restores. Never block the turn cycle on a failed write.
- **LLM unreachable**: return TurnResponse with empty responses array. Upstream will serve empty to SKSE — NPCs continue on engine AI. Log and retry next turn.
- **LLM returns garbage**: `response_expander.py` graceful degradation cascade extracts what it can. Plain text fallback guarantees at least an utterance. Never fail hard on bad LLM output.
- **Malformed EventPayload**: return HTTP 422 with field-level validation errors. Log for diagnostics.
- **State inconsistency on restart**: read last-known state from Qdrant `skyrim_agent_state` on startup. If stale, first few turns will have cold buffers — the EMA catches up naturally. No special recovery logic needed (this IS the zero-init pattern).
