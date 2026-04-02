# Falcon Service — Requirements

Stateless current-tick relay on the Gaming PC. Full HerikaServer replacement.
Falcon owns the current tick: SKSE I/O, embedding, emotional delta computation, memory retrieval, RAW Qdrant writes. It does NOT manage agent state across turns.

## Turn Cycle (Falcon's View)

1. SKSE plugin POSTs a game event (wire format: `type|localts|gamets|data`)
2. Parse event via wire protocol
3. Structurally decode data field via event_parsers.py
4. Push TypedEvent into TickAccumulator
5. On tick cadence (~2s): snapshot buffer, ship TickPackage to Progeny via WebSocket
6. Progeny autonomously decides whether to respond (TurnResponse) or accumulate (AckResponse)
7. On receiving TurnResponse: format responses as CHIM wire protocol, queue for SKSE
8. On SKSE `request` poll: dequeue and return formatted response, or return empty

Falcon is pure data transport — no turn-coupling flags, no gating on event types.
Progeny decides when to respond based on accumulated state.

## Interfaces

### SKSE Plugin (Inbound)

**Endpoint:** `POST /comm.php` (path configurable via `AIAgent.ini`)

**Inbound wire format:** `type|localts|gamets|data` (pipe-delimited string body)

**Event types:**

| Category | Types | Notes |
|---|---|---|
| Player input | `inputtext`, `inputtext_s` | Player typed/spoken input. Progeny decides when to respond. |
| Game state | `info`, `infonpc`, `infoloc`, `location` | NPC state, environment, position. Forward to Progeny for accumulation. |
| Narrative | `chat`, `death`, `diary`, `quest`, `_quest`, `book` | Story events. Forward to Progeny. |
| Control | `request` | SKSE polling for response. Falcon-local — dequeue or return empty. NOT forwarded. |
| Returns | `funcret` | Result of a previously issued action command. Forward to Progeny. |
| Session | `goodnight` | Session end. Falcon handles cleanup, notifies Progeny to flush state. |
| Speech | `_speech` | Speech recognition result (JSON payload). Forward to Progeny. |
| Override | `force_current_task` | Override NPC task. Forward to Progeny. |
| Error | `chatnf` | Chat target NPC not found. Falcon handles, optionally log. |

**Outbound wire format (to SKSE):**
- Dialogue: `NPCName|DialogueType|Text\r\n`
- Actions: `NPCName|command|ActionName@Params\r\n`
- Multiple lines per response for multi-NPC scenes

### Progeny API (Outbound) — The Contract

Falcon communicates with Progeny via a single HTTP endpoint. This is the only interface between the two services. Falcon does not know Progeny's internal architecture.

**Endpoint:** `POST {PROGENY_HOST}/ingest`

**Async Pattern:**
- Data events (non-turn): synchronous POST, Progeny returns immediate ack
- Turn triggers: async POST in background task. Falcon returns immediately to SKSE. When Progeny responds (3-6s), Falcon processes the response (bidirectional delta pipeline) and queues it. Next SKSE `request` poll dequeues.
- If Progeny is unreachable: Falcon returns empty to SKSE. NPCs continue on engine AI. No crash.

#### EventPayload (Falcon → Progeny)

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
- `event_id`: UUID v4, unique per event. Echoed back in response.
- `emotional_state`: keyed by agent_id. Only agents affected by this event are included.
- `memory_context`: present when Progeny runs the full pipeline. Contains retrieval results.
- `npc_metadata`: all NPCs in loaded cells with known metadata. Progeny uses this for scheduling.
- `urgency`: max snap across active agents this tick. Continuous float, not a mode flag.
- `world_state.reset`: true on cell transition — signals Progeny to reinitialize spatial context.

#### TurnResponse (Progeny → Falcon)

Returned as the HTTP response body to a turn-trigger `EventPayload`.

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
    }
  ],

  "processing_time_ms": 4200,
  "model_used": "llama3:8b"
}
```

**Field rules:**
- `responses[]`: one entry per agent that Progeny scheduled this turn. Order matches prompt order.
- All fields except `agent_id` and `extraction_level` are optional. Missing = no change.
- `actor_value_deltas`: integer values, clamped to valid ranges by Progeny. Falcon applies to engine via SKSE.
- `actions[]`: validated command names from the 43-command vocabulary. Unknown commands already stripped by Progeny.
- `updated_harmonics`: LLM-proposed emotional state. Falcon runs the utterance through its own delta pipeline independently — this is a refinement signal, not the authority.
- `extraction_level`: how cleanly the LLM response parsed. `strict` = valid JSON. `repaired` = fixable JSON. `regex` = field-level extraction. `plaintext` = only utterance recovered.
- `utterance`: the raw dialogue text. Falcon formats it into CHIM wire protocol AND runs it through bidirectional emotional delta computation.

#### AckResponse (Progeny → Falcon, non-turn events)

```json
{
  "event_id": "uuid-v4 (echoed)",
  "status": "accumulated"
}
```

### Qdrant (Local — localhost:6333/6334)

Falcon connects to the local Qdrant instance. Both services share the same Qdrant, but Falcon owns RAW-tier writes and all retrieval.

**Collections Falcon reads:**
- `skyrim_npc_memories` — dual-vector (semantic 384d + emotional 9d). Multi-axis retrieval for prompt enrichment.
- `skyrim_world_events` — semantic 384d. World context retrieval.
- `skyrim_lore` — semantic 384d. Oghma Infinium lore for context injection.
- `skyrim_agent_state` — zero-vector 384d, payload-filtered. Read current emotional state per agent (last state written by Progeny).

**Collections Falcon writes:**
- `skyrim_npc_memories` — RAW tier only (`compression_tier: "RAW"`). Immutable event log with emotional vector at encoding time.
- `skyrim_world_events` — game events and world state deltas.

**Write discipline:** Falcon NEVER writes MOD or MAX tier. Falcon NEVER writes to `skyrim_agent_state`. Those are Progeny's domain.

**Retrieval (on turn triggers only):**
- Dual-vector search: emotional resonance (9d) + semantic similarity (384d) via Qdrant `prefetch` + `FusionQuery(RRF)`
- Role referent filtering: payload filter by agents present in scene
- Recency weighting: exponential time-decay on game-time delta
- Sensory anchor boosting: `-log(P(feature))` for rare contextual matches
- Wrapper block expansion: anchor → arc bounds → margin scan → include neighborhood raw points
- Results distilled to keys and summaries for the `EventPayload.memory_context`

## Modules

**`wire_protocol.py`** — Parse inbound SKSE wire format → structured objects. Format outbound agent responses → CHIM wire format. Handle all event types. Input sanitization.

**`emotional_delta.py`** — Bidirectional 9d emotional computation. Processes both inbound game events and outbound LLM utterances through: embed → project to 9d → compute delta against held state → compute curvature → compute snap. Reads current state from Qdrant `skyrim_agent_state`. Does not maintain state across calls.

**`memory_retrieval.py`** — Multi-axis retrieval engine. Dual-vector search, referent filtering, recency decay, anchor boosting, wrapper block expansion. Returns keys + summaries for the EventPayload. Called only on turn triggers.

**`embedding.py`** — sentence-transformers (all-MiniLM-L6-v2) on CPU. Semantic embedding (384d). Emotional vector is passed through (it's the agent's projected semagram, not a model output). Batch embedding with caching.

**`privacy.py`** — 4-level access control (PRIVATE / SEMI_PRIVATE / COLLECTIVE / ANONYMOUS). Filters retrieval results based on querying agent's access level. Level assigned from content characteristics.

**`progeny_protocol.py`** — HTTP client for Progeny. Sends `EventPayload`, receives `TurnResponse` or `AckResponse`. Connection management, retry, timeout. Async support for turn-trigger calls.

**`client.py`** — Qdrant REST API wrapper. Health checks, collection CRUD, dual-vector upsert/search, batch operations.

**`static_import.py`** — One-time import: NPC bios → `skyrim_npc_memories` (data_type=bio), Oghma lore → `skyrim_lore`, function calling definitions. Idempotent via content hashing.

**`api/server.py` + `api/routes.py`** — FastAPI application.
- `POST /comm.php` — SKSE compatibility endpoint (configurable path)
- `POST /memory/search` — debug/admin manual search
- `POST /memory/store` — debug/admin manual store
- `GET /health` — Qdrant connectivity, Progeny status, collection stats
- `GET /agent/{agent_id}/state` — current emotional state from Qdrant

## Shared Dependencies

From `shared/`:
- `schemas.py` — `EventPayload`, `TurnResponse`, `AckResponse`, `EmotionalState`, `MemoryContext`, `ActionCommand`, `ActorValueDeltas` type definitions
- `config.py` — Qdrant host/port, Progeny host/port, embedding model path, distance thresholds, decay parameters
- `constants.py` — 9d axis names, privacy levels, event type enums, actor value ranges, command vocabulary (43 commands)
- `data/emotional_bases_9d.npz` — orthogonal basis vectors for 9d projection

## Constraints

- **Stateless**: no per-agent buffers, no turn history, no arc tracking. Only ephemeral state is the response queue (pending Progeny results awaiting SKSE poll).
- **CPU-only embedding**: GPU is fully committed to VR + Virtual Desktop Streamer. sentence-transformers runs on CPU (~200MB RAM).
- **Wire-compatible**: must accept `AIAgent.ini` configuration (`SERVER`, `PORT`, `PATH`, `POLINT`). SKSE plugin sees no difference from HerikaServer.
- **Single authority on emotional state**: Falcon computes all 9d projections and deltas. Progeny receives deltas but never computes its own projections. This prevents divergence.
- **No PHP, no Apache**: Falcon IS the backend. Zero external dependencies on HerikaServer.
- **Response queue**: the one piece of ephemeral state. In-memory dict keyed by agent_id. Lost on restart (acceptable — next turn regenerates). Not persisted to Qdrant.

## Error Handling

- **Progeny unreachable**: log warning, return empty to SKSE. NPCs continue on engine AI. Retry on next event.
- **Qdrant unreachable**: log error, skip retrieval and RAW write. Forward event to Progeny without memory_context. Degrade gracefully.
- **Malformed SKSE input**: log and drop. Never crash on bad wire data.
- **Malformed Progeny response**: treat as empty response. NPCs continue on engine AI.
- **Embedding failure**: log, forward event without emotional state. Progeny receives partial payload.
