# MMK Operations Guide — Progeny (Beelink)

*Last updated: March 31, 2026. For the next agent context picking up this work.*

## Current State

**400 tests passing.** Full pipeline proven end-to-end: SKSE → Falcon → WebSocket → Progeny → Mistral LLM → Qdrant → response back to Falcon. NPCs (Bryst, Hunter) responded to player dialogue in a live Skyrim VR session on March 30, 2026.

**Uncommitted live-debug fixes on Beelink** (3 files modified, not yet pushed):
- `falcon/api/routes.py` — Accept GET+POST on SKSE endpoint (DLL uses GET)
- `progeny/api/routes.py` — Non-blocking WebSocket handler (asyncio.create_task instead of await)
- `progeny/src/response_expander.py` — Strip `//` line comments from LLM JSON, accept `brief_utterance` field

**Untracked**: `launch-progeny.sh` (startup script, should be committed)

## Network Topology

- **StealthVI (Gaming PC, a.k.a. Falcon)** — Falcon service (:8000), Qdrant (:6333/:6334), Skyrim VR. Router DNS name `Falcon.kenos.one` (address DHCP-reserved on the router)
- **Progeny (Beelink 395AI)** — Progeny service (:8001), llama-server/Mistral (:8080). Router DNS name `Progeny.kenos.one` (address DHCP-reserved on the router)
- **Quest-3** — VR headset via Virtual Desktop
- Falcon connects to Progeny via WebSocket: `ws://Progeny.kenos.one:8001/ws`
- Progeny connects to Qdrant via REST: `http://Falcon.kenos.one:6333` (set with `QDRANT_HOST`)

## Startup Procedure

### On Beelink (Progeny)

```bash
cd ~/Neo
./launch-progeny.sh
```

This starts llama-server (Mistral Nemo 12B, port 8080) and Progeny (port 8001). Waits for LLM health before starting Progeny. Ctrl+C kills both.

Or manually:
```bash
# Terminal 1: LLM
~/llama.cpp/build/bin/llama-server -m ~/models/gguf/mistral-nemo-12b-instruct-q8.gguf --host 0.0.0.0 --port 8080 -ngl 99 -c 8192 --no-mmap

# Terminal 2: Progeny
cd ~/Neo
QDRANT_HOST=Falcon.kenos.one .venv/bin/python -m uvicorn progeny.api.server:app --host 0.0.0.0 --port 8001
```

### On StealthVI (Falcon)

```powershell
# Terminal 1: Qdrant
C:\Tools\qdrant\qdrant.exe

# Terminal 2: Falcon
cd C:\Users\Ken\Projects\many-mind-kernel
$env:PROGENY_HOST="Progeny.kenos.one"
python -m uvicorn falcon.api.server:app --host 0.0.0.0 --port 8000

# Then launch Skyrim from MO2 (Panda's Sovngarde profile)
```

### Startup Order
1. Qdrant (StealthVI) — must be up before Progeny starts
2. llama-server (Beelink) — must be up before Progeny starts
3. Progeny (Beelink) — connects to Qdrant + LLM on startup
4. Falcon (StealthVI) — connects WebSocket to Progeny, auto-reconnects if Progeny isn't ready
5. Skyrim (StealthVI via MO2) — SKSE events flow to Falcon on launch

### Health Checks
```bash
# From Beelink:
curl http://127.0.0.1:8001/health          # Progeny + LLM status + turn counter
curl http://127.0.0.1:8080/health           # llama-server
curl http://Falcon.kenos.one:6333/collections  # Qdrant

# WebSocket connected?
ss -tn | grep 8001 | grep ESTAB
```

## Known Bugs (Active — March 30 debugging session)

### 1. NPC Registry Lost on `init` Events
**Symptom**: Progeny logs "no agents to schedule" even after NPCs registered.
**Cause**: SKSE fires `addnpc` events, then fires `init` (game load signal). Falcon's `clear_npcs()` wipes the NPC registry on `init`, destroying the `addnpc` registrations that arrived in the same batch or earlier.
**Impact**: `active_npc_ids` in TickPackages is empty → scheduler has no agents → empty responses.
**Workaround**: Reload the save after Progeny restart. Or have Falcon replay `addnpc` events after `init`.
**Fix needed**: Falcon should process `init` before `addnpc` in the same tick, or re-register NPCs from `addnpc` events that arrive after `init` in the same batch.

### 2. WebSocket Drops Periodically (~90 seconds)
**Symptom**: Progeny logs "Falcon WebSocket disconnected" followed by reconnect.
**Cause**: Unknown. Possibly ping/pong timeout, or Falcon restart. Auto-reconnect works but state is lost.
**Impact**: After reconnect, Progeny's in-memory state (accumulator, NPC registry) is stale. Combined with bug #1, NPCs disappear.
**Workaround**: Falcon re-sends `addnpc` events after reconnect.

### 3. LLM JSON Comments Break Response Parsing
**Symptom**: Bryst's responses stored as raw JSON blobs in Qdrant instead of extracted utterances.
**Cause**: Mistral adds `// Anger`, `// Fear` etc. as line comments in JSON output. These break `json.loads()`.
**Status**: PARTIALLY FIXED — `_repair_llm_output()` now strips `//` comments. Some responses still slip through (Hunter showed raw JSON in last session). May need more aggressive cleanup.
**File**: `progeny/src/response_expander.py`

### 4. `brief_utterance` Field Not Extracted
**Symptom**: Lower-tier NPCs return empty utterances.
**Cause**: LLM uses `brief_utterance` for Tier 2+ agents, but expander only looked for `utterance`.
**Status**: FIXED (uncommitted) — expander now checks both fields.
**File**: `progeny/src/response_expander.py`

### 5. WebSocket Handler Was Blocking During LLM Generation
**Symptom**: Falcon's ticks queued up during the 12-15s LLM generation window.
**Cause**: `await ingest(package)` in WebSocket handler blocked the read loop.
**Status**: FIXED (uncommitted) — now uses `asyncio.create_task()` for fire-and-forget processing.
**File**: `progeny/api/routes.py`

### 6. Falcon GET vs POST
**Symptom**: SKSE DLL sends GET requests, Falcon only accepted POST.
**Cause**: Route decorators were `@router.post()` only.
**Status**: FIXED (uncommitted) — changed to `@router.api_route(..., methods=["GET", "POST"])`.
**File**: `falcon/api/routes.py`

## Key Files (Progeny side)

- `progeny/api/server.py` — FastAPI app, lifespan (load models, connect Qdrant, warm KV cache)
- `progeny/api/routes.py` — `/ingest` HTTP endpoint + `/ws` WebSocket endpoint + full pipeline
- `progeny/src/response_expander.py` — LLM JSON → AgentResponse extraction with repair pass
- `progeny/src/event_accumulator.py` — Per-agent event buffers, turn boundary detection
- `progeny/src/agent_scheduler.py` — Many-Mind tier scheduling + dispatch groups
- `progeny/src/prompt_formatter.py` — Canonical JSON prompt builder
- `progeny/src/llm_client.py` — LLM backend abstraction (Ollama/llama-server/cloud)
- `progeny/src/harmonic_buffer.py` — 9d EMA buffers, curvature, snap, λ
- `progeny/src/emotional_delta.py` — Inbound/outbound emotional processing
- `shared/config.py` — All configuration (env var overrides)
- `shared/schemas.py` — TickPackage, TurnResponse, AgentResponse, etc.
- `shared/qdrant_wrapper.py` — Enrichment gate: text → embed → project → store → key

## What's Working

- WebSocket Falcon↔Progeny communication (with auto-reconnect)
- SKSE event parsing and tick accumulation
- Many-Mind scheduling (Tier 0 solo, lower tiers batched)
- Parallel dispatch groups (asyncio.gather)
- LLM generation via Mistral Nemo 12B on llama-server
- Response expansion with JSON repair (comment stripping, trailing commas)
- Qdrant enrichment writes (dual-vector: semantic 384d + emotional 9d)
- Keys-over-wire: utterance_key in AgentResponse, Falcon reads by key
- Memory retrieval (dual-vector RRF fusion)
- Emotional delta pipeline (inbound + outbound/bidirectional)
- 400 tests passing

## What's Not Working Yet

- Automatic tick flow (bugs #1 and #2 prevent reliable NPC registration)
- TTS (architecture documented, not implemented)
- STT (SKSE handles via CHIM plugin, but no STT service running)
- Pipelined prompt construction (context_manager/llm_executor split)
- Engine preset values as dynamic modulators (wire protocol gap)
- Goal planner / affordance system
- Progeny-side session handling (Dragon Break, diary, rollback)

## Qdrant Collections (MMK)

- `skyrim_npc_memories` — dual vectors (semantic 384d + emotional 9d)
- `skyrim_agent_state` — emotional 9d (personality substrate)
- `skyrim_world_events` — dual vectors
- `skyrim_session_context` — semantic 384d
- `skyrim_lore` — semantic 384d (Oghma Infinium, 238 entries seeded)
- `skyrim_npc_profiles` — semantic 384d (~1300 NPC bios seeded)
