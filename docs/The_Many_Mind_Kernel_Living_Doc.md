# Skyrim VR AI Companion — Full HerikaServer Replacement

*Authoritative architecture document for the Many-Mind Kernel.*

## Notes for Implementing Agent

This plan was refined through deep conversation covering architecture, emotional cognition theory, and wire protocol analysis. Architecture evolved into a **Falcon/Progeny two-service split**.

* **TWO-SERVICE ARCHITECTURE (Falcon/Progeny)** — Falcon (Gaming PC) handles SKSE I/O and writes inbound dialogue to Qdrant via a shared enrichment wrapper that auto-embeds on ingestion. Progeny (Beelink) owns agent minds: event accumulation, LLM interaction, memory retrieval, and writes LLM responses through the same Qdrant wrapper. Progeny returns keys to Falcon (not full text) — Falcon does a key lookup for wire formatting. Shared emotional projection logic lives in `shared/emotional.py`.
* **FULL HerikaServer REPLACEMENT** — NOT a shim. Falcon replaces ALL of the PHP/Apache server. Only in-game CHIM mod files remain: AIAgent.dll (SKSE plugin), AIAgent.esp, Papyrus .pex scripts, AIAgent.ini (~10 files total).
* **SKSE plugin source is closed** — C++ DLL not published. Wire protocol fully known from the open-source PHP server that receives its output (`comm.php`, `processor/comm.php`, `main.php`). Papyrus .pex scripts decompilable with Champollion if needed. Also check `abeiro/aiagent-aiff` on GitHub for earlier open-source Papyrus.
* **AIAgent.ini** is the only config change: point `SERVER`/`PORT`/`PATH` at our FastAPI endpoint. Format: `SERVER=ip PORT=port PATH=/comm.php POLINT=1`
* **DO NOT** import from thoughtstream-ai, garden, or semantic_memory_management. Clean standalone package. Existing `qdrant_memory_store.py` and `thoughtstream_memory.py` are reference only — study for patterns, don't depend.
* Qdrant runs natively on Windows at `C:\Tools\qdrant\qdrant.exe` (NOT Docker). Ports 6333 (REST) / 6334 (gRPC). 12 existing collections documented below — wrapper MUST read/write without migration.
* **GPU fully committed to VR + Virtual Desktop Streamer (Quest 3).** ALL Python/embedding on Gaming PC must be CPU-only.
* **Emotional harmonics basis vectors ARE the Qdrant vector keys** — enables mood-congruent memory retrieval via vector similarity. This is the core cognitive innovation. See Emotional Architecture section.
* **Study HerikaServer source for protocol details**: `abeiro/HerikaServer` branch `aiagent`. Key files: `main.php`, `processor/comm.php`, `lib/data_functions.php` (273KB context-building functions). Borrow what works, then we're off their update chain. Leave acknowledgement of the upgrade.
* **Qdrant for everything** — no SQLite. Qdrant handles both vector similarity search and simple key-value storage. For pure key-value collections, store zero vectors and query by payload filter. Negligible overhead at this scale.
* Ken's PC: 96GB DRAM, 32GB VRAM. Beelink 395AI: AMD AI SoC running Ollama + Progeny service.

## Cognitive Model

The Many-Mind Kernel's cognition is built on a grammar of state changes. Event boundaries drive memory chunking — the primitive operation of abstraction. The full theory:

### Three Levels of Emotional Dynamics

1. **State** (position) — The current emotional semagram. Where the agent *is* in 9d emotional space right now.
2. **Curvature** (1st derivative) — The priority gradient over time. How fast and in what direction the emotional state is drifting. Curvature characterizes the *between-event* texture — the slow pressure building before something snaps. It IS the urgency signal for prompt shaping and context gating.
3. **Snap** (2nd derivative of emotional state, 1st derivative of curvature) — The rate of change of curvature. Snap detects *event boundaries*: the moment the gradient itself shifts. A sudden spike in snap = something just happened that changed the trajectory, not just the position. Snap is the storage trigger, the arc-boundary detector, and the pre-interruption stash signal.

**The key insight**: Curvature alone couldn't distinguish "steady drift toward danger" from "sudden ambush." Both might reach the same emotional position — but the ambush has high snap (curvature changed abruptly) while the steady drift has near-zero snap. Snap solves the event-boundary detection problem that curvature alone leaves ambiguous.

### Four-Step Cognitive Cycle

1. **Detect event boundaries** — Monitor snap (2nd derivative). When snap exceeds threshold, an event boundary has occurred. Maintain an implicit LIFO stack of open events — each snap spike opens or closes an event frame. Nested events (conversation interrupted by combat interrupted by dragon) stack naturally.
2. **Store definitive properties** — Encode the state at each event boundary as embedding keys searchable by similarity. The emotional semagram at boundary-crossing IS the retrieval key. This enables finding analogous situations: similar emotional signatures → similar contexts → proto-plans for action.
3. **Connect the dots on tension release** — When emotional tension resolves (snap spike marking arc completion), search backward for where the tension started. The span from arc-start to resolution = one compound phase transition. Store this as a single, reusable event trace. Composable: complex plans are chains of remembered phase transitions.
4. **Register dissonances for resolution** — When a resonated plan (retrieved by analogy) is activated, it must register all dissonances between the remembered pattern and the current situation. Each dissonance is either resolved (adapted) or triggers XOR rejection (plan abandoned). This is the error-correction step — not backpropagation, but forward-facing pattern completion with explicit mismatch handling.

### Composability

Event memory traces compose into complex plans. Individual states decompose into relationship components where different component types may resonate more strongly together — indicating tighter binding or better role match. The semagram geometry handles this: two memories with aligned emotional signatures but different semantic content are *analogies*. Two memories with aligned semantic content but different emotional signatures are *contrasts*. The dual-vector architecture makes both searchable.

### The Broken Loop (Affective vs. Domain Recall)

Two fundamentally different retrieval regimes operate in the same memory substrate:

**Affective recall (the "broken loop")** — Something triggers a feeling, below conscious threshold. You notice the feeling and wonder what caused it. Whatever is most strongly associated with that emotional signature emerges as a candidate. You try it on (XOR test against current context). If it doesn't fit, the next association surfaces. The feeling is the *glue* holding the episode together — the subjective experience of memory-as-meaningfully-connected. We presume memories recalled together are causally related, but that's just the emotional index creating narrative coherence, not proving causation. This is the loop: feel → notice → associate → test → narrate. It's post-hoc rationalization as a cognitive primitive.

**Domain recall** — Different animal. You start with the residual (structure, role, affordance) and remember how to use something in your hand — apart from how you feel about it. These memories are better indexed, require less search, because individuated abstractions are more unique in residual space. You don't "feel your way" into how a sword works. The residual encodes: role geometry, domain structure, relationship topology, the shape of the situation independent of emotion.

The distinction maps directly to the 9d semagram:
* Emotional axes (dims 1-8) → affective retrieval key. "How did this feel?"
* Residual magnitude (dim 9) + semantic vector → domain retrieval key. "What kind of thing was this?"

Both regimes use the same memory substrate, same Qdrant collections, same arc summaries. The difference is which index leads the search. See λ(t) in Multi-Axis Retrieval.

## Problem Statement

Build a tight, packageable Python/FastAPI dual-service architecture (Falcon + Progeny) that fully replaces HerikaServer as the backend for Skyrim VR's CHIM mod. Falcon (Gaming PC) is a tick-based black-box decoder: accepts SKSE plugin events via HTTP, structurally parses wire format into typed data, accumulates events between ticks, writes inbound dialogue to Qdrant via a shared enrichment wrapper (auto-embeds on ingestion), and signals Progeny that new content is ready. Progeny (Beelink 395AI) owns ALL cognitive work: memory retrieval, harmonic buffer management, Many-Mind scheduling, prompt building, LLM interaction, and response parsing. Progeny writes LLM responses through the same Qdrant wrapper (text in → key out), and returns keys to Falcon over HTTP. Falcon reads response text from Qdrant by key for wire formatting. Both services share the Qdrant wrapper API and the emotional projection math in `shared/emotional.py`. Emotional harmonics drive memory storage, retrieval, and agent behavior via forward-hold credit assignment — no backpropagation.

## System Architecture

```
GAMING PC — FALCON                         BEELINK 395AI — PROGENY
+-------------------------------------+   +-------------------------------------+
| Skyrim VR -> SKSE -> AIAgent.dll     |   | Progeny Service (FastAPI)           |
|    |                                 |   |   - Harmonic buffer memory         |
| Falcon Service (FastAPI :8000)       |   |   - Emotional delta computation    |
|   - Wire protocol (SKSE <-> JSON)    |LAN|   - Memory retrieval (dual-vector) |
|   - Structural parsing (typed data)  |<->|   - Many-Mind scheduling           |
|   - Tick-based event accumulator     |   |   - Prompt building                |
|   - Response queue (dequeue to SKSE) |   |   - Response parsing               |
|   |                                  |   |   |                                |
|   | Qdrant wrapper (write + embed)   |   | Ollama (local LLM)                 |
|   | Key lookup (read response text)  |   |   |                                |
| Qdrant (localhost:6333/6334)         |   | Qdrant wrapper (write + embed)     |
+-------------------------------------+   | Qdrant (GamingPC:6333 over LAN)    |
| Virtual Desktop Streamer (Quest 3)   |   +-------------------------------------+
+-------------------------------------+

INBOUND:  CE → Falcon → Qdrant wrapper (write + auto-embed) → signal Progeny
OUTBOUND: Progeny → Qdrant wrapper (write + auto-embed) → keys via HTTP → Falcon
          Falcon reads response text from Qdrant by key → formats to SKSE wire
Progeny reads from Qdrant for retrieval and rehydration.

Shared: emotional projection in shared/emotional.py
        Qdrant wrapper: text in → key + vectors out (single enrichment gate)
```

**No PHP. No Apache. No HerikaServer.** Falcon IS the SKSE-facing backend. Progeny IS the mind.

**CHIM is a white-box encoder** — it defines every event type, classifies each one, and triggers explicit handlers. **Falcon is a black-box decoder** — a tick-based metronome that wakes, scrapes everything new from the wire, packages it with structure intact, ships it to Progeny, and goes back to sleep. Falcon doesn't interpret meaning. Progeny does.

Tick-based data flow:
1. SKSE plugin POSTs game events to Falcon continuously (`type|localts|gamets|data`)
2. Falcon: parse wire format, decode data fields into typed structures (JSON, `@`-delimited, `/`-delimited → clean Python objects). No semantic interpretation — just structural decoding.
3. Falcon: accumulate typed events in a time-ordered buffer between ticks
4. Falcon: on tick, write inbound dialogue/text to Qdrant via shared enrichment wrapper (auto-embeds semantic 384d + emotional 9d on ingestion → returns key)
5. Falcon: signal Progeny — "dialogue_ready" + Qdrant key + source
6. Progeny: read content from Qdrant by key (vectors already computed by wrapper)
7. Progeny: compute emotional deltas against held harmonic state, update curvature/snap/λ
8. Progeny: update harmonic buffers, determine who to wake via Many-Mind scheduling
9. Progeny: retrieve relevant memories from Qdrant (dual-vector + referent + recency + anchors)
10. Progeny: build canonical JSON prompt from accumulated state + retrieved memories
11. Progeny → Ollama: send prompt, receive structured JSON response
12. Progeny: parse response, write LLM response text to Qdrant via same wrapper (auto-embeds → key)
13. **Progeny: run LLM-generated response text through the same emotional delta pipeline** — the agent's own words shift its emotional state the same way incoming events do.
14. Progeny: write MOD/MAX (arc summaries, compressed chunks) to Qdrant
15. Progeny → Falcon: return response keys + actions + actor_value_deltas (keys only, not full text)
16. Falcon: read response text from Qdrant by key, format to SKSE wire, enqueue for polling
17. Falcon: serve to SKSE on next `request` poll as CHIM wire format (`NPCName|DialogueType|Text\r\n`)

## Falcon / Progeny Two-Service Architecture

The Many-Mind Kernel splits into two cooperating FastAPI services. The naming reflects AI lineage — Falcon is the hunting arm, Progeny is the growing mind.

### Falcon (Gaming PC) — Tick-Based Black-Box Decoder

Falcon is the substrate layer on structured time. A lightweight metronome that decodes the SKSE wire format into typed data. It does NOT interpret, compute semantics, or embed text. It runs on the Gaming PC alongside Skyrim — minimal CPU footprint on spare cores of the 9950X3D while the 5090 renders.

* SKSE wire protocol parsing (pipe-delimited → typed Python objects)
* Structural data decoding — parse `@`-delimited, `/`-delimited, and JSON data fields into clean typed structures per event type. Mechanical, deterministic, no semantic interpretation.
* Tick-based event accumulation — events arrive continuously, accumulate in a time-ordered buffer. On each tick (~1-3 seconds), snapshot.
* **One write path** — inbound dialogue/text → Qdrant via shared enrichment wrapper (auto-embeds on ingestion). Falcon writes raw content, wrapper handles all embedding.
* **One read path** — key-based point lookup to read response text for wire formatting. Not a search — just a cache fetch by Qdrant key.
* Signal Progeny — notify that new content is ready + pass Qdrant keys
* Response queue — receive response keys from Progeny, read text from Qdrant by key, enqueue locally, serve to SKSE on `request` polls
* CHIM wire formatting — format responses as `NPCName|DialogueType|Text\r\n` for SKSE
* Handle `request` locally — dequeue and return responses without involving Progeny
* Stateless — no per-agent buffers, no emotional state, no memory, no embeddings. Does not compute embeddings (wrapper does).

### Progeny (Beelink 395AI) — Stateful Mind Owner

Progeny owns ALL cognitive work. It reads content from Qdrant (pre-embedded by the shared wrapper), computes emotional deltas, manages harmonic state, retrieves memories, builds prompts, runs LLM inference, and writes LLM responses through the same Qdrant wrapper. The urgency of processing is emergent — Progeny doesn't get wait states, it just processes incoming signals and responds as fast as it can.

* Emotional delta computation — **bidirectional**: processes both inbound game events AND outbound LLM response text through the same project → delta pipeline. Progeny is the single authority on emotional state. Embeddings arrive pre-computed from Qdrant (wrapper handles embedding on ingestion).
* LLM response writes — through the same Qdrant wrapper as Falcon. Text in → key out. Returns keys to Falcon, not full text.
* MOD/MAX writes — arc summaries and compressed essences written directly to Qdrant.
* Memory retrieval from Qdrant (dual-vector search, referent filtering, recency decay, anchor boosting). Progeny never needs to re-embed for retrieval — vectors are already stored from ingestion.
* Event accumulation (buffer per-agent across turns, detect turn boundaries)
* Harmonic buffer memory per agent (fast/medium/slow 9d buffers, curvature, snap, λ, cross-buffer coherence)
* Many-Mind scheduling — determine who to wake, allocate prompt slices by tier
* Memory bundle construction from retrieved keys and summaries
* Fading and salience computation
* LLM prompt building (canonical JSON from accumulated state + bundles)
* LLM interaction via local Ollama (structured JSON response)
* Response parsing → dialogue, actions, harmonics updates
* Privacy filtering on retrieved memories
* Stateful — maintains agent mind state between turns

### The Wire Principle: Qdrant as Buffer, Keys Over the Wire

Qdrant is the shared buffer between Falcon and Progeny. Both services write through the same enrichment wrapper (text in → key + auto-embedded vectors out). Communication between services carries **keys and signals**, not raw text payloads.

* **Falcon → Qdrant wrapper**: inbound dialogue/text → auto-embedded → stored with dual vectors → key returned
* **Falcon → Progeny**: signal ("dialogue_ready" + Qdrant key + source). Lightweight notification, not a full payload.
* **Progeny → Qdrant wrapper**: LLM response text → auto-embedded → stored → key returned
* **Progeny → Falcon**: response keys + actions[] + actor_value_deltas per agent. Keys only (~36 bytes each), not full dialogue text.
* **Falcon ← Qdrant**: key-based point lookup to read response text for wire formatting. Not a search — just a fetch.
* Progeny reads from Qdrant for retrieval, rehydration, and harmonic state recovery.
* LAN traffic is signals and keys (minimal), not structured text payloads.

### Zero-Init Pattern

No explicit agent initialization step. All agent state defaults to zero. The first deltas arriving for a new agent ARE the initial values.

* New agent activation = first game events generate first emotional deltas = initial state
* Cell transitions = spatial state resets to zero, first events in new cell provide new values
* Clean, uniform codepath — no special-case init logic, no "first encounter" branching

### Structured JSON Output (Option B)

The LLM returns structured JSON with explicit fields, replacing CHIM's pure-chat-with-parsing approach:

* CHIM used chat completion + MiniMe-T5 classifier (400MB) to post-process actions from free-text
* We require structured JSON: `updated_harmonics`, `new_memories[]`, `utterance`, `actions[]`
* SKSE plugin never knows the difference — Falcon formats structured response into CHIM wire format
* Eliminates classifier dependency, reduces latency, improves action reliability

### Everything Is Deltas

World state, emotions, memory, identity — all stored and communicated as changes.

* World state: only changes since last tick (full reset on cell transition)
* Emotional state: base vector + accumulated deltas (curvature = rate of delta change)
* Memory: retrieval by emotional delta query → arc expansion → bundle assembly
* Agent identity: persistent kernel + accumulated experience deltas
* LTM recall operator: delta query → expansion → bundle (not snapshot retrieval)

### Fast-Twitch / Slow-Twitch Decoupling (Critical Architectural Insight)

Skyrim's game engine already handles all fast-twitch behavior: combat mechanics, pathfinding, blocking, weapon swings, spell casting, Havok physics, AI packages. NPCs have built-in reflexes. **The MMK does not replace or compete with any of this.**

The MMK is the **slow-twitch mind** — contemplative dialogue, tactical planning, emotional processing, strategic decisions. Not "swing from left to right" but "he has a knife — switch to something with reach." Not "dodge the fireball" but "there are too many of them — we should fall back."

**What this means for latency:** A 3-6 second LLM response time is not a problem — it's *realistic*. It mimics a human's actual OODA loop (Observe, Orient, Decide, Act). The SKSE plugin keeps the NPC fighting natively while Progeny processes in the background. When the response arrives, the NPC adjusts its behavioral posture. Taking 3 seconds mid-combat to assess the tactical situation is *more* believable than instant omniscient response.

**The handoff is already designed.** When snap spikes (ambush, new threat), the pre-interruption stash fires instantly, the game AI takes over combat, and Progeny casually processes a high-curvature truncated prompt. A few seconds later, the response drops: a contextual combat bark + behavioral tuning. The native combat AI was never interrupted. The mind just caught up.

**The Tuning Knobs Model — "We don't tell the NPC to fight. We just make him mad."**

The Creation Engine exposes behavioral actor values — numeric dials that shape NPC behavior at the engine level. These are the MMK's primary output channel for influencing combat and social behavior, not explicit motor commands:

* **Aggression** (0=Unaggressive, 1=Aggressive, 2=Very Aggressive, 3=Frenzied) — attack initiation, engagement distance, target prioritization
* **Confidence** (0=Cowardly, 1=Cautious, 2=Average, 3=Brave, 4=Foolhardy) — flee threshold, defensive posture, blocking frequency
* **Morality** (0=Any crime, 1=Violence against enemies, 2=Property crime, 3=No crime) — willingness to steal, trespass, attack innocents
* **Mood** (0-7: Neutral/Anger/Fear/Happy/Sad/Surprised/Puzzled/Disgusted) — ambient expression, idle behavior, dialogue tone
* **Assistance** (0=Nobody, 1=Allies, 2=Friends and allies) — who they defend in combat

The LLM doesn't issue "Attack" or "Flee" — it adjusts these dials, and the engine's fast-twitch layer handles the behavioral implications. The mind sets the *disposition*; the reflexes execute. This is pure emergence over control: the LLM proposes an emotional/strategic posture, and the game engine's existing AI packages translate that into moment-to-moment behavior.

**Closed loop with harmonics:** The actor values are ALSO readable. Falcon ingests current Aggression/Confidence/Mood as part of the NPC metadata pipeline (alongside HP/MP/SP, equipment, level). These feed directly into the emotional delta computation — the engine's behavioral state becomes an input to the harmonic buffers, creating a bidirectional loop: harmonic state → LLM deliberation → actor value deltas → engine behavior → new events → new harmonic state.

**Implications:**
* `actor_value_deltas` are the primary behavioral output — not motor commands, not even explicit tactical commands. Turn the dials, let the engine act.
* `actions[]` remain for explicit commands from the HerikaServer vocabulary (43 commands — see Command Enum). These handle things the engine dials can't: equip item, travel to location, pick up object, cast specific spell, trade.
* Local 8B model on Beelink is fully viable — 4-6s response for slow-twitch planning is acceptable and even immersive
* Cloud LLM fallback becomes a quality upgrade (smarter dial-tuning), not a latency necessity
* Curvature-driven prompt truncation is about **cognitive focus** (strip irrelevant context, keep tactical situation), not speed. See Prompt Shaping.
* Opens Nexusmods distribution: users with modest hardware (RTX 3060 + local 8B) get viable companion AI, because we're not asking the LLM to keep up with the game engine's tick rate

**Deliberation → Habituation → Instinct (the acid test):**

The architecture's core claim: decisions monotonically improve with deliberation time, and repeated deliberation habituates into an instinct layer, enabling the decision regime to grow while improving reaction time.

1. **Deliberation** — More context in the prompt = better dial-tuning. The game AI buys time; Progeny spends it thinking strategically, not rushing. A 6-second response with full tactical context (enemy equipment, relative power level, party composition) produces smarter Aggression/Confidence settings than a 1-second response with truncated context. Truncation serves focus (strip irrelevant social context during combat), not speed.
2. **Habituation** — Each deliberated dial-setting is stored as a RAW point + arc summary with full emotional and semantic vectors. On the next similar encounter, buffer-sequenced retrieval finds the previous pattern via residual similarity (domain recall, low λ). The LLM receives the prior pattern in context and refines it rather than inventing from scratch. Qdrant retrieval IS habituation — the memory substrate IS the instinct layer.
3. **Growing regime** — More stored arcs = richer tactical repertoire. The agent's decision space expands with experience. Novel situations still require slow deliberation; familiar ones pull up proven dial-settings.
4. **Instinct formation** — After many encounters of the same type: high residual coherence across buffers → low λ → domain-first recall → instant pattern retrieval → confident, fast dial response. The slow-twitch deliberation from early encounters has become fast-retrieval pattern matching. The agent acts from experience. A veteran's Confidence stays at 3 (Brave) while a novice's oscillates wildly — same code, different accumulated patterns.

**Emergent expertise without personality rules:** A veteran warrior (200 combat arcs in Qdrant) and a novice merchant (3 combat arcs) in the same ambush produce fundamentally different behavior from the same code and the same model:
* Veteran: high residual coherence → low λ → domain-first recall → retrieval pulls up relevant dial patterns → Aggression=2, Confidence=3 (aggressive, brave) → engine fights confidently
* Novice: low residual coherence → high λ → emotion-first recall → sparse, panicky retrieval → Aggression=0, Confidence=0 (unaggressive, cowardly) → engine cowers, flees, hides
* Same progression over time: the novice who survives 50 fights gradually develops the veteran's dial signature. Expertise emerges from accumulated experience, not from a flag or a stat.

**Quest-Collision Guard & Slow Reintegration — "Coming out of a trance."**

When NPCs are in scripted quest sequences, the Creation Engine's quest AI owns their behavior — dialogue, movement, positioning, combat triggers. If the MMK applies `actor_value_deltas` during a scripted scene, it collides with the quest director. The NPC is a puppet on two strings.

*Detection:* The SKSE plugin checks `Actor.IsInScene()` (Papyrus) each tick and includes an `in_scene` flag in the NPC metadata sent to Falcon. This is a cheap, authoritative check — the engine knows when it's running a scripted sequence.

*Guard:* When `in_scene: true`, `response_expander.py` does NOT apply `actor_value_deltas` to the engine. Instead, it queues them in a **pending delta buffer** (per-agent, in `event_accumulator.py`). The LLM still deliberates — the mind keeps thinking — but the dials don't turn. `actions[]` are also held (except read-only intelligence commands like Inspect/SearchMemory that don't alter behavior).

*Slow reintegration:* When `in_scene` clears, the pending deltas don't slam in at full value. The NPC was in a narrative-induced trance — the quest AI was driving. The mind's accumulated state seeps back like a slow memory returning:

1. On scene exit, `response_expander.py` feeds the pending delta buffer into the **slow harmonic buffer's EMA blend** rather than applying directly to engine actor values.
2. Each subsequent tick, the blended values propagate outward: slow buffer → medium buffer → fast buffer → engine dials. The existing decay rates (α_slow, α_med, α_fast) govern the reintegration speed.
3. The deltas are also **attenuated** — not applied at full accumulated value. A scaling factor (e.g., 0.3-0.6, tunable per-agent) reduces the magnitude. The mind was absent; the returning impressions are faded, not crisp.
4. If new events arrive during reintegration (they will — the quest just ended, things are happening), fresh deltas from real-time deliberation naturally supersede the stale queued ones via the EMA. The returning trance-memory blends with present reality.

*Why this is emergence:* No mode switch, no "quest mode" flag in the LLM prompt, no special-case logic in the mind. The guard is a simple gate at the output layer. The reintegration uses the same harmonic buffer EMA that governs all emotional state transitions. The slow buffer IS the trance-recovery constant — agents with slower decay rates (grudge-holders, deep thinkers) take longer to "come back." Agents with fast decay (reactive, impulsive) snap to the present almost immediately. Same mechanism, different personality expression.

*Asymmetric timing (same pattern as Pre-Interruption Stash):* The guard is instant — one `in_scene: true` flag and the output gate closes. The release is slow — governed by buffer decay rates, attenuated, blended. You go under instantly; you come back gradually. The architecture keeps finding this pattern because it's how minds actually work.

## Current State

**GitHub**: https://github.com/KenOtwell/many-mind-kernel (public, MIT). Falcon + Progeny code, shared schemas, **400 tests passing**. Both Falcon (StealthVI) and Progeny (Beelink) pull from this repo.

**Qdrant Instance**: `C:\Tools\qdrant\qdrant.exe` — ports 6333 (REST) / 6334 (gRPC), bound to `0.0.0.0` (LAN-accessible). Config: `C:\Tools\qdrant\config.yaml`. Progeny (Beelink at `192.168.0.220`) confirmed connecting to `192.168.0.13:6333`.

**Shared Enrichment Layer** (committed, in `shared/`):
* `emotional.py` — 384d → 9d semagram projection. Basis loading, single/batch projection, residual computation. Used by both Falcon and Progeny.
* `embedding.py` — all-MiniLM-L6-v2 sentence embeddings on CPU. Singleton model loading, batch/single embed. Used by both Falcon and Progeny.
* `qdrant_wrapper.py` — **The enrichment gate.** `ingest()`: text in → embed (384d) → project (9d) → store dual-vector point → return key. `read_text()`: key-based point lookup. Both services call the same API.
* `schemas.py` — `AgentResponse.utterance_key`: Progeny can return a Qdrant key instead of inline text. Falcon resolves via `read_text()`. Falls back to inline `utterance` (backward compat).

**Falcon Enrichment Wiring** (committed):
* `server.py` — Startup loads embedding model + emotional bases + initializes AsyncQdrantClient (localhost).
* `routes.py` — `_resolve_utterance_keys()`: when Progeny returns `utterance_key`, Falcon reads text from Qdrant by key before wire formatting. Session events now forwarded to tick accumulator for Progeny visibility.
* Response queue bounded at `maxlen=64`.

**Progeny Qdrant Modules** (committed, in `progeny/src/`):
* `qdrant_client.py` — Async module-level API: `init()`, `get_client()`, `ensure_collections()`, `write_memory()`, `write_agent_state()`, `read_agent_state()`, `search_memories()` (RRF fusion), plus generic helpers. NOTE: `write_memory()` currently takes pre-computed vectors; RAW writes will flow through the shared wrapper once Progeny's response pipeline is updated.
* `memory_writer.py` — MemoryWriter: async RAW/MOD/MAX tier writes, world events, agent state snapshots, session stash, lore.
* `memory_retrieval.py` — MemoryRetriever: async λ(t)-weighted dual-vector search, recency decay, referent boosting, arc expansion.
* `compression.py` — ArcCompressor (snap-threshold → MOD) + EssenceDistiller (MOD groups → MAX).
* `rehydration.py` — Rehydrator: async MAX→MOD→RAW expansion chain, post-interruption stash recovery.
* `embedding.py`, `emotional_projection.py` — Re-export shims pointing to `shared/embedding.py` and `shared/emotional.py`.

**Remaining Progeny work to close the keys-over-wire loop:**
* `response_expander.py` — Write LLM utterances via `shared.qdrant_wrapper.ingest()` instead of inline.
* Set `AgentResponse.utterance_key` instead of `AgentResponse.utterance` when returning to Falcon.

**MMK Qdrant Collections** (5 new, 17 total):
* `skyrim_npc_memories` — dual named vectors (semantic:384d + emotional:9d), indexes: agent_id, tier, game_ts, privacy_level
* `skyrim_world_events` — dual vectors, indexes: event_type, location, game_ts
* `skyrim_session_context` — single semantic:384d, index: agent_id
* `skyrim_agent_state` — single emotional:9d, index: agent_id
* `skyrim_lore` — single semantic:384d, index: topic

**ChIM Source Code** (local reference copies, NOT dependencies):
* `docs/AIAgent/` — Complete AIAgent mod: SKSE DLL, Papyrus source (.psc), PrismaUI overlay (HTML/CSS/JS), ESP plugin
* `docs/Dwemer Distro/` — Complete HerikaServer: PHP backend (`var/www/html/HerikaServer/`), extractable from `DwemerAI4Skyrim3.tar`
* Key server files verified: `main.php` (wire protocol parser), `comm.php` (legacy entry point), `processor/` (response formatting)
* **Integration decision**: Use ChIM’s SKSE DLL + Papyrus scripts as-is for game-side I/O. Replace ONLY the PHP backend with Falcon/Progeny Python services. The cognitive architecture is the innovation; game integration is plumbing. ChIM thinks it’s talking to its PHP server — it’s actually talking to Falcon.

**Network Map** (confirmed March 2026):
* StealthVI/Falcon: `192.168.0.13` (wired) — Qdrant, Falcon service, Skyrim VR
* Progeny/Beelink: `192.168.0.220` (wired) / `192.168.0.231` (WiFi 6G) — Progeny service, Ollama
* Quest-3: `192.168.0.41` (WiFi 6G) — VR headset via Virtual Desktop

**Existing Collections** (12 pre-MMK):
* `garden_archive` — 1,148,827 pts, dim=384 (speaker, witnesses, content_hash, emotional_valence, concepts, source_file, document_type)
* `trinity_archive` — 43,409 pts, dim=384 (conversation_thread, author, message, emotional_salience, timestamp, was_filtered)
* `complete_codebase` — 300 pts, dim=384 (file_path, chunk_type, name, content, calls, patterns, tags)
* `project_docs` — 39 pts, dim=384 (content, file_path, project, chunk_type, layer, tags)
* `fractal_consciousness` — 13 pts, dim=768 (content + content_l1/l2/l3, current_tier, owner_id, privacy_level, resonance_score, emotion, wobble_amplitude, compressed_10/50/100, semantic_topics)
* `garden_emotions` / `garden_focus` — 10 each, dim=1536 (experiencer, emotion, focus_description, intensity, valence, arousal, focus_tags)
* `test_pubsub_integration` — 5 pts, dim=768
* `test_reciprocity` — 3 pts, dim=384
* `fractal_memories` — 1 pt, dim=1536 (text, author, emotion, timestamp)
* `access_contexts` / `certainty_timelines` — 0 pts, dim=1536 (schema stubs)

**Vector Dimensions in Play**: 384 (all-MiniLM-L6-v2), 768 (sentence-transformers), 1536 (OpenAI-compatible)

**CHIM/HerikaServer** (what we are REPLACING — GitHub: abeiro/HerikaServer):
* This entire PHP/Apache stack is replaced by our FastAPI service
* Original context building in `main.php`: system prompt + NPC personality + command prompt + world context + historic context + memory injection — we replicate in Python
* Original SQLite tables: `eventlog`, `log`, `diarylog`/`diarylogv2` — replaced by Qdrant collections
* Original memory: `logMemory()`/`offerMemory()` via TXT2VEC/ChromaDB — replaced by dual-vector Qdrant retrieval
* LLM connectors: was OpenAI/KoboldCPP/Openrouter/Groq — we route exclusively to Beelink Ollama
* Oghma Infinium: 1900+ lore topics — imported once into Qdrant as static reference data
* TTS/STT: Architecture documented — see Audio Pipeline section. STT already handled via `inputtext_s`. TTS owned by Falcon (local xVASynth/MeloTTS/Kokoro). 16 TTS + 3 STT + 4 ITT backends available from CHIM's ecosystem.

**Existing Wrappers** (reference only, in `thoughtstream-ai/src/garden/`):
* `qdrant_memory_store.py` — privacy-aware CRUD, 4-tier compression, caching, pub/sub
* `thoughtstream_memory.py` — emotional thread tracing, wobble sync, resonance amplification

**Rich Modlist Deployment Notes** (Panda's Sovngarde / large Wabbajack lists):

The system was initially designed for vanilla + CHIM. A rich behavior modlist changes the event landscape substantially — and entirely in our favor.

* **Every behavior mod is an event source.** NPC reaction mods (`NPCs React to Necromancy`, `Scared of Shootings`, etc.) generate `info`/`infoaction` events. Dialogue mods (RDO, GDO, FCO, `Chatty NPCs`) generate `chat` events. All of this flows through Falcon, gets adopted into NPC histories, and feeds the harmonic buffers. The scripted AI is not competing with MMK — it is providing the sensory input layer.
* **Increase Actors in a Cell** (up to 75 active actors) makes the scheduler's Tier 2/3 cadence filtering non-trivial. Token budget math holds; the harmonic cadence earns its keep at scale.
* **Nether's Follower Framework (NFF)**: our `is_follower` detection and `MakeFollower` command must be NFF-aware. NFF manages follower factions differently from vanilla. Verify `is_follower` flag against NFF's follower faction formIDs, not just vanilla `0x0005c84e`.
* **Pre-populated NPC histories**: by the time the player first speaks to an NPC, that NPC may already have a rich adopted history from RDO relationship dialogue, ambient barks, reaction events. Cold-start NPCs are not cold — they are nascent. This amplifies the Cold-Start Identity Formation pathway described above.
* **Panda's Sovngarde was designed as the richest non-LLM NPC experience possible**, using volume of scripted behavior to approximate cognitive depth. MMK adds the actual cognition layer. The scripted behavior becomes the training data; the system provides the mind.

## SKSE Wire Protocol

The SKSE plugin (AIAgent.dll) communicates via simple HTTP POST. Our FastAPI endpoint must be 100% wire-compatible. **Source of truth**: Full ChIM source now local at `docs/AIAgent/` (mod files + Papyrus source) and `docs/Dwemer Distro/` (HerikaServer PHP backend, extractable from `.tar`). Key files verified against actual source: `main.php` (base64 pipe-delimited wire parser, line 106: `$gameRequest = explode("|", $receivedData)`), `comm.php` (legacy entry point), `processor/comm.php` (response formatting). Also available on GitHub: `abeiro/HerikaServer` branch `aiagent`.

**Base inbound format**: `type|localts|gamets|data` (pipe-delimited string, split on first 3 pipes — data field may contain pipes)
* POST to path configured in `AIAgent.ini` (default: `/HerikaServer/comm.php`, we alias to `/comm.php`)
* `type` = event type string (lowercase)
* `localts` = local timestamp (string)
* `gamets` = game timestamp (numeric, Skyrim internal time counter. 0.0000024 = 1 hour. Used for save/reload rollback.)
* `data` = event-specific payload (varies by type — see below)

**Architecture note — pre-processor vs. fall-through**: In HerikaServer, `processor/comm.php` is a pre-processor that runs before `main.php`. Events that match a handler in the pre-processor set `$MUST_END=true` and return. Events that DON'T match any handler (like `inputtext`) fall through to `main.php` for LLM processing. This is why `inputtext` is a turn trigger — it's not explicitly handled, it falls through to the LLM.

### Complete Event Taxonomy

**Turn triggers** (fall through pre-processor to LLM — these are the events that trigger a prompt+response cycle):
* `inputtext` — Player typed text input. Data = the player's text.
* `inputtext_s` — Player speech-to-text input. Data = transcribed text. Same processing as `inputtext`.
* NOTE: In CHIM these trigger one-NPC-at-a-time processing. In MMK, Progeny determines which NPCs to wake based on scene context.

**Falcon-local events** (handled entirely by Falcon, never forwarded to Progeny):
* `request` — SKSE polls for queued responses. No data payload. Returns all queued responses as `{actor}|{action}|{text}\r\n` lines (one per queued response, may be empty if nothing ready).
* `chatnf` — Chat with NPC not found. Logs warning, returns empty.
* `just_say` — Direct output passthrough. Data = text to output. Falcon queues `data` verbatim into the response queue; SKSE picks it up on the next `request` poll (same path as Progeny responses). Does NOT return text in the HTTP response directly.

**Dialogue/speech context** (pre-stages context for turn triggers):
* `_speech` — Speech event. Data = **JSON**: `{listener, speaker, speech, location, companions[], distance, audios, debug}`. The `companions` array contains names of nearby NPCs — critical for Many-Mind scheduling (determines Chorus tier candidates). The `distance` float affects hearing radius. Stores to speech table and exits (`$MUST_END=true`) — does NOT trigger LLM directly. The speech data is read later when `inputtext` falls through to LLM processing.
* `chat` — NPC-to-NPC dialogue. Data = dialogue text.
* `funcret` — Function return value from SKSE. Data = return value string.

**Session lifecycle** (Falcon-local — none are forwarded to Progeny in the current implementation):
* `init` — Game load/reload. Data = DLL version string. **MMK**: Falcon clears its NPC registry (`active_npc_ids`) and returns empty. (In HerikaServer: triggers cleanup of all DB tables with `gamets >= incoming_gamets`, creates Dragon Break snapshots, restores NPC state, loads SNQE quests.)
* `wipe` — Full database reset. **MMK**: Falcon clears NPC registry and returns empty. (In HerikaServer: clears all tables unconditionally.)
* `playerdied` — Player death. **MMK**: Falcon clears NPC registry and returns empty. (In HerikaServer: rolls back all tables to the gamets of the last `infosave` event, same Dragon Break snapshot logic as `init`.)
* `goodnight` — Night cycle event. **MMK**: Falcon logs and returns empty. (In HerikaServer: triggers auto-diary for nearby NPCs.)
* `waitstart` — Player begins waiting/sleeping. **MMK**: Falcon logs and returns empty. (In HerikaServer: stores gamets, triggers auto-diary.)
* `waitstop` — Player finishes waiting/sleeping. **MMK**: Falcon logs and returns empty. (In HerikaServer: computes elapsed hours from `waitstart` gamets, logs time-forward event.)
* NOTE: Session events are now forwarded to the tick accumulator (as of March 2026) so Progeny receives them in TickPackages. Progeny-side session handling (rollback, diary generation, Dragon Break snapshots) is not yet implemented.

**NPC registration and stats** (`addnpc` is the richest event — 43+ fields):
* `addnpc` — NPC enters loaded cells. Data = `@`-delimited, 43+ fields:
    * `[0]` name, `[1]` base ID, `[2]` gender, `[3]` race, `[4]` refid
    * `[5-22]` 18 skills: archery, block, onehanded, twohanded, conjuration, destruction, restoration, alteration, illusion, heavyarmor, lightarmor, lockpicking, pickpocket, sneak, speech, smithing, alchemy, enchanting
    * `[23-32]` 10 equipment slots (each `name^baseid`): helmet, armor, boots, gloves, amulet, ring, cape, backpack, left_hand, right_hand
    * `[33-40]` 8 stats: level, health, health_max, magicka, magicka_max, stamina, stamina_max, scale
    * `[41]` mods — `#`-delimited list of mod names the NPC comes from
    * `[42]` factions — `formID:rank#formID:rank#...` pairs
    * `[43]` class — `className:formID:trainSkill:trainLevel`
* `updatestats` — Live combat stats (sent every ~3s in combat or on hit). Data = `@`-delimited: `npcName@level@health@health_max@magicka@magicka_max@stamina@stamina_max@scale`
* `itemtransfer` — Item exchange between NPCs. Data = natural language: `"SourceNPC gave Count ItemName to DestNPC"` (regex-parsed: `/^(.+?) gave (\d+) (.+?) to (.+)$/`)
* `enable_bg` — Enable background life for NPC. Data = `/`-delimited: `npcName/refid`
* `switchrace` — NPC race change notification. Just logged.

**Quest events**:
* `_quest` — Quest registration. Data = **JSON**: `{formId, name, currentbrief, currentbrief2, stage, data:{questgiver}, status}`
* `_uquest` — Quest stage update. Data = `@`-delimited: `formId@unknown@briefing@stage`
* `_questdata` — Quest supplementary data. Data = `@`-delimited: `formId@briefing2`
* `_questreset` — Clear all quests from quest table.
* `quest` — Quest notification (game event). Data = quest text string. Filtered: ignores blank names and "Storyline Tracker" (AIAgent internal). Has Narrator quest-comment integration with chance/cooldown.

**Info/world events** (prefix-matched with `strpos` — any event starting with `info` is caught):
* `info*` prefix — General info events. Just logged to eventlog. Includes `info`, `infonpc`, `infoloc`, `infoaction`, `infocombat`, `infospell`, etc.
* `infosave` — Player saves game. Triggers backup of all NPC profiles and SNQE quest state. Used as rollback point for `playerdied`.
* `location` — Cell/location change. Data = location name string. Cached globally.
* `book` — Book opened. Data = title string.
* `contentbook` — Book content (deprecated in favor of 1.2.0). Data = content (HTML tags stripped).
* `death` — NPC death event. No processing in pre-processor (just sets `$MUST_END`).

**World data utilities** (bulk data loading for world database — `/`-delimited payloads):
* `util_location_name` — Location data. Data = `/`-delimited: `name/formid/region/hold/tags/is_interior/factions/x/y`. Special: `__CLEAR_ALL__` truncates table.
* `util_faction_name` — Faction data. Data = `/`-delimited: `formid/name`
* `util_location_npc` — NPC position update. Data = `/`-delimited: `npcName/x/y/z/tag`. Maintains last_coords + 10-entry history.
* `named_cell` — Cell door topology. Data = `/`-delimited, 12 fields: `cell_name/id/location_id/interior/dest_door_cell_id/dest_door_exterior/door_id/worldspace/closed/door_name/door_x/door_y`
* `named_cell_static` — Cell static items. Data = `/`-delimited: `cellId/comma-separated name@refid pairs`

**Task/mission management**:
* `force_current_task` — Override NPC's current mission. Data = description string.
* `recover_last_task` — Remove the most recent mission.

**Configuration/admin**:
* `setconf` — Configuration change. Data = `@`-delimited: `key@value`. Special cases: `chim_context_mode` (toggle), `chim_renamenpc` (`key@oldname@newname@refid` — renames NPC with profile migration).
* `togglemodel` — Toggle LLM model. Returns: `{name}|command|ToggleModel@{model}\r\n`

**Dynamic profiles** (CHIM-specific, we'll reinterpret for MMK):
* `updateprofile` — Legacy single NPC dynamic profile update. Runs LLM to rewrite NPC biography based on recent dialogue.
* `updateprofiles_batch_async` — Batch dynamic profile update. Data = comma-delimited NPC names. Checks `DYNAMIC_PROFILE` flag per-NPC, forks background processing.
* `core_profile_assign` — Assign NPC to a core profile slot.

**SNQE (Synthetic Narrative Quest Engine)**:
* `snqe` — Quest engine control. Data = `@`-delimited: action (`START`/`END`/`CLEAN`/`RESTART`). Launches background agent processing.

**Other**:
* `diary` — Diary event. Data = diary entry text.
* `diary_nearby` — Manual diary trigger for all nearby NPCs.

**Deprecated** (logged with warnings, no processing):
* `updateequipment`, `updateinventory`, `updateskills` — Now handled by separate `gamedata.php` endpoint with JSON POST.

### Falcon's Parsing Responsibility

Falcon structurally decodes each event type's data field into typed Python objects. This is mechanical parsing, not semantic interpretation — Falcon doesn't know what the data means, just how to unpack it.

**Parsers** (in `event_parsers.py`):
* `_speech` → JSON deserialize → `SpeechData(listener, speaker, speech, location, companions, distance)`
* `addnpc` → split on `@` → `NpcRegistration(name, base, gender, race, refid, skills{}, equipment{}, stats{}, mods[], factions[], class_info)`
* `updatestats` → split on `@` → `NpcStats(npc_name, level, health, health_max, magicka, magicka_max, stamina, stamina_max, scale)`
* `_quest` → JSON deserialize → `QuestData(form_id, name, brief, stage, giver, status)`
* `_uquest`/`_questdata` → split on `@` → `QuestUpdate(form_id, briefing, stage)`
* `itemtransfer` → regex parse → `ItemTransfer(source, dest, item_name, count)`
* `util_location_name` → `/`-split → LocationData (name, formid, region, hold, tags, is_interior, factions, x, y)
* `util_faction_name` → `/`-split → FactionData (formid, name)
* `util_location_npc` → `/`-split → NpcPosition (npc_name, x, y, z, tag)
* `named_cell` → `/`-split → CellData (cell_name, id, location_id, interior, door topology, 12 fields)
* `named_cell_static` → `/`-split → CellStaticItems (cell_id, name@refid pairs)

* All unrecognised types → raw string data preserved in a generic `TypedEvent(event_type, local_ts, game_ts, raw_data)` wrapper with `parsed_data=None`

**Wire flow model names**: `wire_protocol.py` produces `ParsedEvent` (routing-annotated) → `routes.py` constructs `TypedEvent` (Pydantic, with `parsed_data` from `event_parsers.py`) → `tick_accumulator.py` batches into `TickPackage` → Progeny returns `TurnResponse` (if turn trigger present) or `AckResponse` (data-only tick). All models defined in `shared/schemas.py`.

Progeny receives these typed objects and does all semantic work: embedding, emotional projection, scheduling, prompting.

**Outbound (us to SKSE):**
* Dialogue: `NPCName|DialogueType|Text\r\n`
* Actions: `NPCName|command|ActionName@Params\r\n`
* Action types: Follow, Trade, Attack, MoveTo, Wait, Telekinesis, etc. (full 43-command vocabulary in Command Enum section)
* Multiple lines per response for multi-NPC group conversations
* Special responses: `{name}|rolecommand|DebugNotification@{message}\r\n` for in-game debug notifications

**Config (`AIAgent.ini`):**
```
SERVER=127.0.0.1
PORT=8000
PATH=/comm.php
POLINT=1
```
POLINT = polling interval in seconds for `request` event type.

## Emotional Architecture

**Core thesis:** Emotions are the fuel of cognition. Not decoration, not side-effect — the actual optimization gradient. Draws from Werbos' ADP but diverges fundamentally: **no backpropagation. Forward-hold only.**

### The Mechanism

1. **Emotional bath** — Agents continuously hold an emotional state (the harmonics basis vector). Not computed, *inhabited*.
2. **Event arrives** — Something changes in the world. If it matters, the emotional signature shifts.
3. **Threshold crossing** — Delta between current and held emotional state exceeds threshold = a **resolution/satisficing event**. This is BOTH a **storage trigger** AND a **search trigger** simultaneously.
4. **Store raw event** — Always. Full payload, current emotional vector, all referents, timestamp. Immutable.
5. **Search for arc start** — Use the *pre-shift* emotional signature as query vector. "Find the event where this emotional arc began." Qdrant returns it via emotional vector similarity.
6. **Generate arc summary** — The span from arc-start to now = one emotional arc. Condensed summary stored as MOD-tier index entry pointing to raw data.
7. **Update held state** — New emotional signature becomes the held state for the next arc.

### Why This Works (Credit Assignment Without Backprop)

* The emotional residue on each stored memory IS the label
* No need to trace backward through decision chains
* "Which past decision led to this outcome?" becomes "which stored memories carry a similar emotional signature to this moment?"
* Qdrant answers that in milliseconds via vector similarity
* O(1) emotional credit assignment instead of O(n) backpropagation through decision history

### Curvature, Snap, and Delay Buffers

* `curvature` (1st derivative) = rate of emotional change = the priority gradient over time. How fast the agent's emotional state is drifting right now. Curvature characterizes the *between-event* texture — the slow pressure building, the steady drift toward or away from danger.
* `snap` (2nd derivative) = rate of change of curvature = the event boundary detector. When snap spikes, the *trajectory* changed — not just the position. Snap triggers arc storage, event boundary detection, and pre-interruption stashing. See Cognitive Model section.
* `harmonic_buffers` (fast/medium/slow) = three timescale traces of the FULL 9d semagram. Each buffer holds a complete 9d vector — all 8 emotional axes + residual — decayed at its characteristic rate.
    * `fast` (τ ≈ 3-5 ticks) — reactive surface. Tracks the agent's immediate emotional+domain trajectory.
    * `medium` (τ ≈ 15-25 ticks) — session texture. The felt quality of this encounter/conversation.
    * `slow` (τ ≈ 50-100 ticks) — personality substrate. The deep background pattern that barely moves.
* **Update rule** (per tick, for each buffer tier): `buffer_t = α_t · new_semagram + (1 - α_t) · buffer_t` — exponential moving average. α_t (decay rate) is a per-agent personality parameter per tier. Fast α is large (tracks closely), slow α is small (remembers deeply).
* **Cross-buffer coherence** = agreement across the three timescales, computed per dimension and overall:
    ```
    coherence[dim] = 1 - normalized_var(fast[dim], medium[dim], slow[dim])
    overall_coherence = mean(coherence[0..8])
    ```
    High overall coherence = agent is in a stable, consistent state across timescales. Low = volatile transition, timescales disagree. Per-dimension coherence reveals WHAT is stable: high coherence on residual but low on emotional axes = stable domain, volatile feelings (combat in familiar territory). The opposite = stable mood, changing context (exploring while content).
* **Personality through buffer geometry** — no personality rules needed:
    * Veteran warrior in combat: high residual coherence across all buffers → stable domain context → λ stays low → calm tactical recall
    * Same warrior in unexpected diplomacy: residual coherence drops (buffers disagree on domain) → λ rises → emotional recall → fumbling, drawing on memories of past social encounters
    * Merchant in first fight: fast buffer spikes on emotional axes, medium/slow still hold calm-merchant pattern → low cross-buffer coherence → fast buffer dominates retrieval → panicky, reactive
    * After N ticks of sustained combat: medium buffer catches up to fast → partial coherence → gradual stabilization into the new context
* Short decay rates = reactive personality. Long decay rates = grudge-holding personality. But now the *shape* of the buffer difference matters too — not just duration, but *which dimensions* are stable.
* **The math IS the personality.** Decay rates, buffer geometry, snap thresholds, and λ gains define agent character — no separate personality rules needed.

### Engine Preset Values as Dynamic Modulators (Not Set-Points)

*Insight documented March 2026. Lineage: Ken Ong (theory), Oz/Warp (mechanism design).*

Skyrim's Creation Engine assigns every NPC a set of preset behavioral values at spawn — Aggression, Confidence, Morality, Mood, Assistance. The initial temptation is to project these directly into 9D semagram space as emotional set-points (homeostasis targets). **This is wrong.** These values are not emotional primitives — they are derived behavioral attractors. They describe emergent behavioral patterns, not fundamental emotional dimensions.

#### The Alignment Fallacy

Think of D&D alignment: "Chaotic Good" or "Neutral Evil" aren't primitive psychological dimensions — they're shorthand labels for behavioral patterns that emerge from deeper value structures interacting with context. Similarly, Skyrim's Aggression=2 doesn't mean "resting at angry." It means the NPC's behavioral *response dynamics* to threat signals are tuned to amplify and sustain aggression.

Whining isn't a primitive — it's a fallback response to helpless need in the context of usual need-fillers being absent. Bravery isn't a state — it's a damping coefficient on the fear signal. These values describe the *transfer function* of the NPC's behavioral feedback loops, not positions in emotional space.

#### The Five Values as Dynamic Modulators

Each engine preset value maps to a specific dynamic property of the harmonic buffer system — a gain, a damping factor, a threshold, or a bias — not a coordinate:

**Aggression** (0=Unaggressive → 3=Frenzied) → **Gain multiplier on threat-response axes.**
High aggression doesn't mean "resting at angry." It means the anger and excitement axes *amplify faster* when provoked and *decay slower* after the stimulus passes. The gain is asymmetric: fast rise, slow fall. Frenzied NPCs have a ratchet-like anger dynamic — easy to wind up, slow to wind down.
* Modulates: per-axis α-rate on anger, excitement, fear→excitement conversion
* Fast buffer: gain on anger/excitement axes scaled by `1 + aggression * k_agg`
* Slow buffer: decay on anger/excitement axes attenuated by `1 - aggression * k_agg_persist`

**Confidence** (0=Cowardly → 4=Foolhardy) → **Damping factor on fear/uncertainty axes.**
Foolhardy isn't "never scared" — it means the fear signal gets attenuated before it reaches the decision layer. The raw fear delta still arrives from the emotional projection pipeline, but its effective magnitude is scaled down by the confidence damping coefficient. A Cowardly NPC feels full fear. A Foolhardy NPC feels a muted echo.
* Modulates: effective delta magnitude on fear and safety axes
* Applied as: `effective_fear_delta = raw_fear_delta * (1 - confidence * k_conf_damp)`
* At Confidence=4 (Foolhardy): fear deltas are attenuated to ~20% of raw magnitude
* Does NOT suppress fear *storage* — the raw delta still writes to Qdrant with full emotional vector. Only the harmonic buffer update sees the damped value. The memory remembers the real fear; the mind just doesn't dwell on it.

**Morality** (0=Any crime → 3=No crime) → **Threshold gate on action selection.**
Morality is not emotional at all. It doesn't modulate how the NPC *feels* — it gates what the NPC *does* with those feelings. A Morality=0 NPC and a Morality=3 NPC can reach identical emotional states (rage, desperation, greed), but the action space available to resolve those states differs. Morality is a filter on `response_expander.py`'s action selection, not a parameter of the harmonic buffers.
* Modulates: action filtering in response expansion, NOT buffer dynamics
* Implementation: when the LLM proposes actions, `response_expander.py` filters against the NPC's morality threshold. Steal, trespass, and attack-innocent commands require morality ≤ their crime level.
* The LLM still *deliberates* about immoral options (they appear in the prompt context). The gate is at the output, not the input. An NPC with Morality=3 might *want* to steal the potion — but won't.

**Mood** (0-7: Neutral/Anger/Fear/Happy/Sad/Surprised/Puzzled/Disgusted) → **Ambient bias on the emotional baseline.**
Mood is the closest to a true set-point — but only on a single axis. It biases the *resting state* of one emotional dimension toward a non-zero value. An NPC with Mood=Happy has a positive bias on the joy axis that all three buffers drift toward during low-curvature periods. The bias is weak (shouldn't override strong emotional events) but persistent (always nudging back).
* Modulates: EMA target on the mood-corresponding axis (not zero, but `mood_bias[axis]`)
* Update rule for the biased axis: `buffer_t[axis] = α_t · new_semagram[axis] + (1 - α_t) · (buffer_t[axis] + mood_pull * (mood_bias[axis] - buffer_t[axis]))`
* `mood_pull` is small (~0.01-0.05) — a gentle ambient drift, not a hard anchor
* Maps Skyrim's integer mood enum to the corresponding semagram axis: Anger→anger, Fear→fear, Happy→joy, Sad→sadness, Disgusted→disgust, etc.
* Neutral (0) = no bias on any axis = the default zero-target behavior

**Assistance** (0=Nobody → 2=Friends and allies) → **Social binding strength / emotional bleed coefficient.**
Assistance modulates how much nearby agents' emotional states influence this NPC's deltas. An NPC with Assistance=2 (defends friends and allies) has a higher emotional bleed coefficient — when allies experience high-curvature events (combat, fear, anger), a fraction of that curvature propagates into this NPC's buffers even if the NPC didn't directly experience the stimulus. An NPC with Assistance=0 is emotionally isolated — other agents' states don't bleed in.
* Modulates: cross-agent emotional coupling coefficient
* When computing deltas for agent A, if ally agent B has high curvature: `bleed_delta = B.curvature * assistance_coupling * (B.fast_buffer - A.fast_buffer)`
* `assistance_coupling` scales with the Assistance value (0 = zero bleed, 2 = full coupling)
* This is how a squad develops coordinated emotional responses — not because they share a hive-mind, but because emotional bleed through social bonds creates correlated buffer trajectories

#### The Joker Example — Emergent Behavioral Profiles

Consider an NPC with: Confidence=4 (Foolhardy), Aggression=3 (Frenzied), Morality=0 (Any crime), Mood=3 (Happy).

The dynamic modulators produce:
* Fear signals attenuated to ~20% → threat events barely register emotionally
* Anger/excitement gain cranked to maximum → every stimulus amplifies into excitement
* No moral filtering on actions → full action space available
* Ambient joy bias → buffers drift toward cheerful during calm periods

The emergent behavior: a happy psychopath. Threat situations that would terrify other NPCs get converted into excitement through the high-gain anger/excitement path and damped fear channel. The NPC approaches danger with enthusiasm. Combat arcs stored in Qdrant carry excitement-dominant emotional vectors, not fear-dominant ones. On retrieval, the NPC recalls combat as *fun*. The Deliberation→Habituation→Instinct pipeline locks this in: after enough joyful combat arcs, the NPC's instinct layer retrieves "combat = exciting" patterns at low λ.

Nobody scripted "psychopath." The gain settings produced one.

Conversely: Confidence=0 (Cowardly), Aggression=0 (Unaggressive), Morality=3 (No crime), Mood=4 (Sad), Assistance=0 (Nobody). This NPC feels full fear, has no anger gain, can't act immorally, drifts toward sadness, and gets no emotional support from allies. The system produces a withdrawn, frightened loner — not from a personality rule, but from the dynamics.

#### Interaction with Zero-Init

The Zero-Init Pattern (see above) states that all agent state defaults to zero. This still holds for emotional *position* — the 9D semagram starts at the origin. But the *dynamics that govern how that state evolves* are immediately parameterized by the engine values from `addnpc`.

Zero-Init becomes: "Born at emotional zero, but with your own physics." Two NPCs receiving identical first events will diverge immediately — not because they started at different positions, but because the same stimulus propagates differently through their differently-tuned feedback dynamics. Lydia's Confidence=3 damping means her fear buffer barely moves; a Cowardly merchant's fear buffer spikes hard. Same input, different transfer function, divergent trajectories from tick 1.

#### Wire Protocol Gap and Implementation

**Current state**: The `addnpc` event (43+ fields) carries skills, equipment, stats, factions, and class — but NOT the 5 behavioral actor values. These values exist on every NPC in the Creation Engine but aren't included in the SKSE plugin's registration payload.

**Options to acquire**:
1. **Papyrus script at registration** — Add a small Papyrus script that reads `GetActorValue("Aggression")` etc. on NPC load and fires a custom ModEvent or appends to the `addnpc` payload. Requires a companion `.psc` script similar to `MMKSetBehavior.psc`. Cleanest approach — one-time read per NPC, shipped with the NPC metadata.
2. **Separate HTTP endpoint** — A Papyrus script that periodically reads and POSTs the 5 values for active NPCs. More complex, but allows detecting runtime changes (e.g., if a quest script modifies an NPC's Aggression).
3. **Default lookup table** — Hardcode known values for named NPCs from the Creation Kit data. Works for vanilla NPCs but breaks for mod-added NPCs. Fragile, not recommended.

**Recommended**: Option 1 (Papyrus read at registration) with Option 2 as a future enhancement for detecting quest-driven changes. The values rarely change at runtime in vanilla Skyrim — they're effectively static personality parameters.

**Progeny integration**: On receiving the 5 values (via `addnpc` extension or separate event), `harmonic_buffer.py` initializes per-agent dynamic modulator coefficients:
```
agent_dynamics = {
    "aggression_gain": normalize(aggression, 0, 3),      # 0.0-1.0
    "confidence_damp": normalize(confidence, 0, 4),       # 0.0-1.0
    "morality_threshold": morality,                        # 0-3 integer
    "mood_axis": MOOD_TO_AXIS[mood],                       # axis index or None
    "mood_pull": 0.03 if mood != 0 else 0.0,              # ambient drift strength
    "assistance_coupling": normalize(assistance, 0, 2),    # 0.0-1.0
}
```
These parameterize the per-tick buffer update without changing the core EMA math. Agents without engine values (fallback) get all-zero modulators = the current uniform-dynamics behavior.

#### Why This Is Emergence, Not Control

The 5 engine values don't tell the NPC what to feel or how to act. They tune the *physics* of the emotional manifold — how signals propagate, amplify, attenuate, and couple. Behavior emerges from the interaction of these dynamics with the actual event stream. The same Frenzied NPC in a peaceful village and a war zone produces radically different behavior from the same gain settings, because the input stream is different. The dynamics are the instrument; the world plays the music.

This extends the Tuning Knobs Model (see Fast-Twitch / Slow-Twitch Decoupling) into a bidirectional loop: the engine's preset values parameterize the mind's dynamics (input), and the mind's deliberated actor_value_deltas tune the engine's behavioral posture (output). The NPC's personality shapes how it processes the world, and the world shapes the NPC's personality through accumulated experience. The engine values are the initial conditions; the trajectory is emergent.

### Dual-Vector Architecture

Each memory point stores TWO named vectors in Qdrant:
* `semantic` (384d, all-MiniLM) — *what* happened (text content embedding)
* `emotional` (9d, harmonics basis) — *how it felt* (agent's emotional state at encoding time)

Similarity search on the emotional vector = **mood-congruent memory recall**: agents remember sad things when sad, not because of rules, but because the vector geometry makes it so. This is a real cognitive phenomenon (Bower 1981) emerging from data structure alone.

Retrieval blends both axes via Qdrant `prefetch` + `FusionQuery(RRF)`. Emotional intensity bias: high arousal shifts weight toward emotional axis, calm states bias toward semantic axis.

### The Kryptonite Problem (Future Research)

* Synthetic intelligences go straight to coherence — no emotional cost for dissonant state transitions
* Real moral signaling comes from the *cost* of resolution (betrayal HURTS because the gradient is steep)
* The harmonics architecture creates computational resistance to dissonant transitions (high loyalty = steep gradient against betrayal)
* Whether gradient-as-resistance is sufficient without subjective experience is an open question
* For Skyrim NPCs: the resistance creates believable behavior regardless of the philosophical question

### Concrete 9d Emotional Semagram

The full emotional signature — the **semagram** — is a 9-dimensional coordinate. A semagram is a sign that carries meaning through its geometric structure, not through symbolic labels. Two memories resonate not because they share a tag, but because their semagrams point the same way. Mood-congruent recall falls out of dot products.

**Axes 1-8** (Gram-Schmidt priority order — earlier = preserved more faithfully):
1. **fear** — drift 1.000 (untouched, first in priority)
2. **anger** — drift 0.985 (barely moved)
3. **love** — drift 0.986 (barely moved)
4. **disgust** — drift 0.846 (significant reshape, shed anger+sadness components)
5. **excitement** — drift 0.941 (moderate shift)
6. **sadness** — drift 0.900 (moderate shift, shed disgust overlap)
7. **joy** — drift 0.780 (most reshaped, shed safety+sadness+disgust components)
8. **safety** — drift 0.803 (significant reshape, last in priority, shed everything)

**Axis 9: residual magnitude** — `||emb_norm - (coeffs @ bases)||`
* Measures how much of the embedding is NOT captured by the 8 emotional axes
* Orthogonal to all 8 emotion axes by construction (it IS the orthogonal complement's magnitude)
* Serves as: emotionality meter (low = high emotional content), domain-content tiebreaker in similarity search
* Won't overwhelm emotional matching (1 of 9 orthogonal dims) but breaks ties toward "same kind of thing"

**Residual-space experimental findings** (`scripts/test_residual_9d.py`):
* Combat words cluster strongly in residual space (avg +0.447 vs +0.287 baseline): sword↔warrior +0.586
* Relationship words cluster by role geometry, not sentiment: ally↔enemy +0.570, enemy↔rival +0.519
* Emotion words are LEAST clustered in residual space (+0.242, below baseline) — once emotional content is subtracted, pure emotion words have nothing in common
* The residual encodes **role bindings** independent of emotional polarity

**Construction pipeline** (`scripts/emotional_bases.py`):
1. Seed words per axis (hand-curated synonyms + antonyms)
2. Thesaurus validation via Datamuse API (bidirectional synonym check, score-weighted)
3. Score-weighted embedding means: `v = weighted_mean(embed(synonyms)) - weighted_mean(embed(antonyms))`
4. Gram-Schmidt orthogonalization in priority order above
5. All 8 bases are unit vectors in MiniLM's 384d space; 9th dimension computed at projection time

**Projection**:
```
coeffs[0..7] = bases_8x384 @ normalize(text_embedding_384d)   # 8 emotional coordinates
coeffs[8]    = ||normalize(text_embedding_384d) - (coeffs[0..7] @ bases_8x384)||  # residual magnitude
```
Result is a 9d semagram: 8 emotional coordinates + 1 domain-content residual.

**Validation scorecard** (sample):
* "terrified" → fear +0.437, residual 0.881 ✓ | "furious" → anger +0.307, residual 0.935 ✓
* "revolting" → anger +0.256 / disgust +0.223 (multi-axis, both correct) ✓
* "poison" → safety -0.245 / disgust +0.201 (both correct) ✓
* Neutrals ("table", "the", "walk") all below ±0.1 emotional, residual >0.99 ✓

**Known limitation**: "elated" scores sadness(-0.302) stronger than joy — this is a property of MiniLM's semantic space, not the orthogonalization. The raw embedding encodes "elated" as anti-sadness more than pro-joy.

**Artifacts**:
* `shared/data/emotional_bases_9d.npz` — primary artifact (8 orthogonalized bases + residual metadata + raw bases)
* `shared/data/emotional_bases_8d.npz` — original 8d bases (retained for reference)

```python
data = np.load('emotional_bases_9d.npz')
bases = {k: data[k] for k in data['_emotion_names'][:8]}  # 8 unit vectors, each 384d
# 9th dim (residual) computed at projection time: ||emb_norm - (coeffs @ bases)||
```

## Memory Architecture

**Two layers only. Keep it simple.**

* **Raw points** — Every event, always stored, never deleted, never modified. Immutable log. Source of truth.
* **Arc summaries** — Condensed descriptions of emotional arcs. Stored as MOD-tier index entries. Used ONLY as search aids to find relevant raw points. Like a library card catalog — you search the catalog to find the book, then read the book (raw data).

### Storage Trigger

```
event arrives
  -> compute emotional delta, update curvature (1st derivative)
  -> compute snap (2nd derivative = rate of curvature change)
  -> snap > threshold?  (event boundary detected)
     YES -> store raw point
         -> search for arc start (pre-shift emotional signature as query)
         -> generate arc summary of the span
         -> store summary as MOD-tier index point
         -> update held emotional state
         -> push/pop event frame on implicit LIFO stack
     NO  -> store raw point
         -> update held emotional state (minor drift via curvature)
```

Snap — not raw delta — is the event boundary detector. Curvature alone can't distinguish "steady drift toward danger" from "sudden ambush" — both accumulate delta, but only the ambush produces a snap spike. The snap threshold catches the moment the *trajectory* changes, not just the position.

Only difference between significant and insignificant events: whether an arc summary gets generated. Raw data stores either way.

**Compaction** is a separate operational concern — when raw point count impacts performance, promote old points RAW->MOD->MAX. Operational decision, not cognitive. Keep out of core loop.

## Multi-Axis Retrieval

Memory retrieval is where the cognitive architecture pays off. Multiple axes, context-weighted.

### Axes

1. **Emotional resonance** (primary) — Dual-vector similarity on harmonics basis. "What memories were formed in a similar emotional state?" Mood-congruent recall.
2. **Role referents** — Payload filter: which agents were involved? If Lydia is in the scene, boost memories involving Lydia.
3. **Recency** — Exponential time-decay multiplier on game-time delta. Recent memories surface easier.
4. **Sensory anchors** — Rare/unique contextual features that create strong associative links. The "smell" axis. Weight by `-log(P(feature))`: if "Falkreath" appears in 3% of memories and "rain" in 40%, Falkreath match gets a much bigger boost. Information-theoretic weighting.

### λ(t): Emotional–Residual Retrieval Balance

A single continuous variable λ(t) balances emotional vs. residual similarity weighting during retrieval. This is the 9th-dimension operator — it governs how much weight to give feeling vs. structure when searching memory.

```
similarity = λ(t) · emotional_sim + (1 - λ(t)) · residual_sim
```

Where:
* λ(t) → 1 = emotion-first retrieval (episodes, narratives, social memory, grudges)
* λ(t) → 0 = residual-first retrieval (domain knowledge, combat tactics, tool use, procedures)
* λ(t) ≈ 0.5 = mixed (moral decisions, ambiguous social situations, complex quests)

**Update rule** — driven entirely by existing signals:
```
λ(t+1) = σ(α · curvature(t) + β · snap(t) - γ · residual_coherence(t))
```
* σ = sigmoid (keeps λ in [0, 1])
* α, β, γ = tunable gains (per-agent personality or learned)
* **curvature** pushes λ up — high emotional volatility = emotion-first recall
* **snap** spikes λ up — event boundary = force emotional indexing ("what just happened?")
* **residual_coherence** pushes λ down — stable, structured situation = domain-first recall

`residual_coherence` = cross-buffer agreement across all 9 dimensions (see Curvature, Snap, and Harmonic Buffers). When the agent's fast/medium/slow buffers converge — same emotional tone and domain context across timescales — coherence is high and retrieval shifts toward structural matching. This replaces the earlier scalar definition; now coherence captures the full 9d stability picture, not just residual magnitude variance.

**Emergent behavior**: In calm, structured situations → λ low → residual-first. Rising tension → λ rising → mixed. Sudden danger → λ spikes → emotional-first. After resolution → λ relaxes → residual-first again. No mode flags. No rules. Just the calculus.

**This is how humans shift** between procedural and episodic memory: "What kind of situation am I in?" (residual-first) vs. "This feels like that time..." (emotion-first). λ(t) captures the balance.

Developed with Kato/Copilot. The α/β/γ gains are personality parameters — an agent with high α is emotionally reactive in recall, an agent with high γ is procedurally grounded.

### Buffer-Sequenced Retrieval Matching

Instead of a single query vector, retrieval exploits all three buffer timescales as parallel query lenses. This biases toward near-term matches while allowing old patterns to surface when the deep background state resonates.

**Mechanism**: For each candidate memory, compute λ-weighted 9d similarity against each buffer tier, then combine with coherence-modulated weights:

```
for each candidate memory m:
  for each buffer tier t in {fast, medium, slow}:
    sim_t = λ(t) · cosine(buffer_t[0:8], m.emotional[0:8])
          + (1 - λ(t)) · residual_sim(buffer_t[8], m.emotional[8])

  buffer_score = w_fast · sim_fast + w_med · sim_medium + w_slow · sim_slow
```

**Dynamic weight modulation** — cross-buffer coherence adjusts the tier weights:
* **High coherence** (buffers converge) → boost slow buffer weight. Deep recall is trustworthy when all timescales agree. Old memories that resonate with the stable background surface.
* **Low coherence** (buffers diverge) → boost fast buffer weight. Stay reactive during transitions. Slow buffer patterns may not apply to the current volatile state.
* The weight adjustment is continuous, not switched. Default baseline: w_fast=0.6, w_med=0.3, w_slow=0.1 — personality parameters per agent.

**What this buys:**
* A memory matching ALL 3 buffers = extraordinarily strong retrieval signal (consistent across timescales)
* A memory matching only fast = recent/reactive match (mood of the moment)
* A memory matching only slow but NOT fast = old deep pattern that the current state has drifted away from — surfaces only when the background is stable enough to trust it
* The slow buffer IS long-term emotional memory indexing without a separate LTM mechanism

**Orthogonality with λ(t)**: λ controls *what* to search by (feeling vs. structure). Buffer cascade controls *when* to search by (timescale priority). They compose:
* High λ + fast-dominant = "What does this feel like RIGHT NOW?" (reactive emotional recall)
* Low λ + slow-dominant = "What kind of situation has this been, historically?" (deep domain recall)
* Mixed λ + high coherence = balanced, confident retrieval across the full state

### Retrieval Process

1. **Compute λ(t)** — from current curvature, snap, and cross-buffer coherence (see above)
2. **Broad resonance pass** — Buffer-sequenced 9d similarity (see above): each candidate scored against fast/medium/slow buffers with λ-weighted emotional/residual balance and coherence-modulated tier weights, via Qdrant `prefetch` + `FusionQuery(RRF)`. Top ~30 candidates, filtered by role referents present in scene.
3. **Re-rank** — Apply recency decay + sensory anchor boost (-log frequency)
4. **Top 5-8 as anchors**
5. **Expand to arc bounds** — For each anchor, find parent arc summary, get time bounds
6. **Retrieve wrapper blocks** — Pull all raw points in arc time window + margins. Unkeyed data rides along for free (the "smell" effect — mundane details become associated by temporal proximity)
7. **Scan for initiation** — Optionally search further back for precursor events with faint emotional signature match (the seed that planted the idea before it became salient)

### Tier Mapping to Canonical JSON Schema

* `state_history.recent[]` <- neighborhood raw points around anchors (the wrapper blocks)
* `state_history.summaries[]` <- arc summaries that led retrieval here
* `state_history.expandable_refs[]` <- Qdrant point IDs for full raw arcs the LLM can request

**Key insight:** All of this is pre-LLM retrieval shaping. The LLM gets a curated set of memories that already *feel* right. The LLM narrates what the NPC does with those memories — it doesn't decide which ones to recall.

## Event Accumulator / Turn Cycle

SKSE plugin fires events continuously. Falcon accumulates and packages on a tick; Progeny interprets and responds.

* Falcon: receives SKSE events, `wire_protocol.py` produces a `ParsedEvent` with routing flags (`is_turn_trigger`, `is_local`, `is_session`)
* Falcon: **Falcon-local events** (`request`, `chatnf`, `just_say`) handled immediately and never accumulated. `request` dequeues responses; `just_say` queues data to response queue; `chatnf` logs and returns empty.
* Falcon: **Session events** (`init`, `wipe`, `playerdied`, `goodnight`, `waitstart`, `waitstop`) handled locally — the first three clear Falcon's NPC registry (`active_npc_ids`), all return empty. Not forwarded to Progeny in current implementation.
* Falcon: all other events → `event_parsers.py` decodes structure → `TypedEvent` pushed to `tick_accumulator.py`
* Falcon: on tick (~1-3 seconds), snapshots buffer, wraps as `TickPackage` (includes `active_npc_ids` from accumulated `addnpc` events), ships to Progeny, clears buffer
* Progeny: receives typed event package, embeds text, computes emotional deltas, writes RAW to Qdrant
* Progeny: `event_accumulator.py` ingests typed events, maintains per-agent event buffers across turns
* Progeny: world state deltas accumulated from typed event data
* **Turn boundary** = `inputtext` or `inputtext_s` event detected in the incoming package by Progeny (not by Falcon — Falcon ships it like any other event)
* On turn boundary: Progeny flushes all buffers -> Many-Mind scheduling -> build canonical JSON -> send to Ollama -> return response bundle to Falcon
* Between turns: partial agent packets get timestamp-only stubs (don't inject wait states, just hold)

**No injected wait states.** If an agent's packet isn't complete, send only timestamp. Don't fabricate filler. Progeny doesn't get wait states — it just processes incoming packages and responds as fast as it can.

### Behavior Adoption (Intentional Design Decision)

All externally-generated NPC behavior — both dialogue and actions — is adopted into the agent's own history as if the LLM produced it. The agent cannot distinguish "I did this" from "the game made me do this." On the next turn, it sees the behavior as its own and rationalizes continuity from it.

This is not a hack — it mirrors how human cognition actually works. We don't have reliable access to *why* we said or did something. We post-hoc rationalize, constructing coherent narratives from whatever actions we observe ourselves taking. The agent does the same: it adopts externally-generated behavior as its own, then narrates coherence from it. See "The Broken Loop" in Cognitive Model.

**Dialogue adoption** — When SKSE sends a `chat` event where the speaker matches an active agent (vanilla combat bark, follower comment, ambient dialogue), `event_accumulator.py` files it into that agent's conversation history buffer with the same role tag as LLM-generated responses (`role=assistant`).

**Action adoption** — When SKSE sends action events where the actor matches an active agent (`info`, `infoaction`, `spellcast`, `npcspellcast`, `combatbark`, `bleedout`, `death`), the same treatment applies: the action is stored in the agent's history as something *it did*, not something that happened to it. The RAW point in Qdrant is written with the agent's current emotional vector, indistinguishable from an LLM-commanded action.

**Both streams run through the bidirectional emotional delta pipeline.** Progeny embeds the text, projects to 9d, computes the delta. Whether Lydia drew her sword because the LLM commanded `Attack@Bandit` or because the game's combat AI triggered it — same embedding, same emotional shift, same RAW write, same arc participation. The memory trace is identical.

**What this buys:**
* **Personality bootstrapping** — Before the LLM has ever spoken for an NPC, vanilla barks and game-AI actions provide a behavioral history. The LLM picks up the NPC's "voice" and action patterns from examples.
* **Seam continuity** — If the game fires a vanilla combat bark ("I'll handle this!") and draws the NPC's sword mid-fight, then the player talks to that NPC right after, the LLM sees both as things it did and follows naturally ("As I said, I've got this handled — that bandit won't bother us again.").
* **Emotional coherence** — Adopted behavior runs through emotional delta computation same as everything else. The agent's semagram reflects all its actions and dialogue, regardless of source.
* **Productive dissonance** — If the game generates an out-of-character action, the agent must reconcile it. This creates the kind of mild tension that drives interesting behavior through snap/curvature dynamics — the agent post-hoc rationalizes, just like we do.
* **Memory composability** — Adopted actions participate in arc detection, arc summaries, and retrieval exactly like LLM-generated ones. "Remember when you fought that dragon?" works even though the game AI handled the combat — the emotional arc was real, the memory was stored, the retrieval key is valid.

**Implementation:** In `event_accumulator.py`:
* `chat` events where speaker matches active agent → `role=assistant` in dialogue history
* `info`/`infoaction`/`spellcast`/`npcspellcast`/`combatbark`/`bleedout` events where actor matches active agent → `role=assistant` in action history (formatted as: "I [action description]")
* No provenance flag, no source tracking. The adoption is total.
* All adopted behavior flows through Progeny's `emotional_delta.py` for embedding and delta computation, same as LLM-generated output.

### Cold-Start Identity Formation

When an `addnpc` event arrives and the NPC's slug matches nothing in `skyrim_npc_profiles`, no biography seed is loaded. This is not an error — it is an alternative developmental path.

**What cold-start NPCs have:**
* Zero harmonic state (zero-init: first deltas ARE initial values)
* Complete, accurate memories of everything they do once active (behavior adoption is total)
* All scripted behavior from behavior mods adopted as their own — every reaction event, every ambient dialogue line, every faction interaction

**What cold-start NPCs lack:**
* A stated identity kernel ("Roleplay as X...")
* Pre-loaded goals, relationships, backstory
* The *felt sense of deliberation* that preceded their scripted actions

This is not amnesia. Amnesia patients have *missing* memories. Cold-start NPCs have *complete* memories — every adopted action is in their RAW history with full emotional vectors. What they lack is access to the causal layer that produced those actions. The quest script, the AI package, the mod behavior — that layer is invisible to them. They see only the output. They remember *doing* it; they don't remember *deciding* it.

The result: an NPC who must construct meaning from evidence rather than apply a pre-loaded narrative. The LLM's rationalizations become the character. The harmonic buffers accumulate the emotional weight of those rationalizations. A coherent identity emerges — not loaded, but *grown*. This is especially pronounced for handcrafted custom NPCs from narrative mods: the scripted behaviors were designed with intent, but the NPC constructs their *own* theory of why they keep doing these things.

**Rich modlist pre-seeding**: When a modlist includes extensive scripted AI (NPC reaction mods, dialogue overhauls, follower behavior frameworks), cold-start NPCs arrive with populated adopted histories before the player speaks to them. Their harmonic buffers have already processed dozens of events. They are not cold in any practical sense — they are nascent.

### Goal Resonance — Propositional vs. Resonant Goals

A seeded NPC's goals are *propositional* — stated strings in the identity kernel ("I protect my thane"). A cold-start NPC's goals, when they develop, are *resonant* — encoded in the slow harmonic buffer's convergent emotional signature from repeated relevant experiences.

**The hypothesis**: resonated goals may be more durable than assigned goals. The slow buffer carries the accumulated emotional cost of every relevant experience. Contradicting a resonated goal produces real harmonic dissonance — snap spikes, curvature surges, λ shifts. Contradicting a stated goal produces only a prompt-level inconsistency, which the LLM can rationalize away.

**The natural experiment**: any deployment with both seeded NPCs and unknown/custom NPCs creates side-by-side comparison subjects. Run both through the same scenario sequence. Compare:
* λ(t) trajectories under goal-threatening events
* Curvature magnitude at decision points where goals conflict with new information
* Specificity and depth of rationalization when behavior contradicts stated vs. resonant goals
* Behavioral stability across extended sessions

**Observable prediction**: cold-start NPCs show higher early volatility (high curvature, frequent snap) stabilizing over time as the slow buffer converges. Seeded NPCs show lower early volatility but potentially more brittle goal-holding under sustained pressure — the stated goal is a claim, not an emotional substrate.

**No new code required**: the architecture already captures everything needed. Persisting `emotional_delta.py` outputs alongside session data enables post-session analysis. The experiment runs itself.

## Goal Planner & Affordance System

Agents need to act autonomously across time. When a player says "Go to Windhelm and fetch my good sword," the NPC needs to decompose that into steps, execute them across many ticks without LLM involvement, handle failures, and exploit opportunities. This is constraint-based state-space planning with the 43-command vocabulary as the operator library.

### Architecture

The LLM does **planning** (slow-twitch deliberation). The affordance matcher does **execution** (fast, no inference). The LLM does **replanning** when execution fails.

**1. Goal Queue** (per agent, persisted in `skyrim_agent_state`)
* Ordered list of goals, each decomposed into steps
* Each step: `{command, target, item, precondition, done_when}`
* LLM generates goals as part of its response — new `goals[]` field alongside `actions[]` and `utterance`
* `actions[]` fires the first step immediately; `goals[]` persists the remaining plan for autonomous execution
* Multiple goals coexist (fetch sword AND keep watch for bandits)

**2. Affordance Set** (computed each tick by Progeny)
* What the agent CAN do right now given: position, inventory, nearby objects/NPCs, current state
* Built from Falcon's event data: `addnpc` (who's nearby), `util_location_npc` (positions), `updatestats` (state), `location` events, `named_cell_static` (what's in the cell)
* Simple predicate matching: `at_location:X`, `near:X`, `has_item:X`, `player_has:X`, `is_alive:X`, `is_open:X`
* Updated incrementally as events arrive — not recomputed from scratch

**3. Goal-Affordance Matcher** (runs each tick BEFORE the LLM prompt)
* Scan the agent's active goal: if the next step's precondition is satisfied by the current affordance set, emit the command immediately. No LLM call needed.
* This is fast-twitch goal execution — the plan was slow-twitch (LLM deliberation), but executing known steps against satisfied preconditions is instant.
* Step completion detected by checking `done_when` predicate against next tick's affordances. On completion, advance to next step.

**4. Opportunism** (constraint propagation across all goals)
* Don't just check the NEXT step — scan ALL goals for any step whose precondition is currently satisfied.
* If Lydia is passing through Windhelm on another errand and the sword-fetch goal is queued, she grabs the sword now rather than waiting for that goal to become active.
* Priority: current active goal > opportunistic match on other goals. Opportunistic matches don't interrupt active execution unless the opportunistic goal has higher emotional urgency (curvature).
* This is E7 (Competitive Convergence) applied to goals — paths compete, the one with least resistance wins.

**5. Replanning** (when steps fail)
* Step fails: container locked, item missing, NPC hostile, path blocked → flag goal step as `blocked`
* Next LLM prompt includes the blocked goal + failure context in the agent's state_history
* LLM either replans (alternative path), adapts ("the sword isn't there, but I found a better one"), or abandons (report failure to player)
* Emotional delta from failure feeds the harmonic buffers — frustration from repeated failures is real, affects future planning confidence

### Example: "Go to Windhelm and get my good sword"

LLM response:
```json
{
  "agent_id": "Lydia",
  "utterance": "I'll head to Windhelm and get your sword.",
  "actor_value_deltas": { "Confidence": 3 },
  "actions": [{ "command": "TravelTo", "target": "Windhelm" }],
  "goals": [{
    "description": "Retrieve player's Ebony Sword from Windhelm",
    "steps": [
      { "command": "TravelTo", "target": "Windhelm", "done_when": "at_location:Windhelm" },
      { "command": "MoveTo", "target": "Hjerim", "done_when": "near:Hjerim" },
      { "command": "PickupItem", "target": "Ebony Sword", "done_when": "has_item:Ebony Sword" },
      { "command": "TravelTo", "target": "Player", "done_when": "near:Player" },
      { "command": "GiveItemToPlayer", "target": "Ebony Sword", "done_when": "player_has:Ebony Sword" }
    ]
  }]
}
```

`actions[]` fires `TravelTo Windhelm` immediately. The goal persists. Across subsequent ticks, the affordance matcher advances steps without LLM involvement. If something goes wrong (Hjerim is locked), the blocked state enters the next LLM prompt for replanning.

### Deliberation → Habituation → Instinct (for Goals)

Same pipeline as emotional processing, applied to task execution:
* **Deliberation** — First time: LLM decomposes the goal into steps (~4 seconds). Full constraint reasoning in natural language.
* **Habituation** — Execution: affordance matcher runs steps without LLM. The plan is cached in the goal queue. No inference needed until something breaks.
* **Instinct** — Repeated similar goals: LLM retrieves previous plan from memory (via Qdrant — "last time I was sent to fetch something, here's what I did"), adapts it. Faster deliberation, better plans.

### Connection to KO47 Invariants

* **S8 (Dynamic Affordance Coupling)** — the affordance set IS S8. Computed each tick from the agent's coupling with its environment.
* **E11 (Goal-Directed Coherence / Attractor Convergence)** — the goal-reality differential $\vec{D}_{signal} = \vec{S}_{goal} - \vec{S}_{reality}$ is checked every tick by the affordance matcher.
* **E7 (Competitive Convergence)** — opportunism. Multiple goals compete; the one whose next step is currently satisfiable wins execution time.
* **S9 (Energy Regulation)** — goals that repeatedly fail drain emotional energy (frustration). The agent may deprioritize or abandon costly goals.

### Implementation

**New Progeny module: `goal_planner.py`**
* Goal queue management: add, advance, block, complete, abandon
* Affordance set computation from accumulated event state
* Goal-affordance matching (predicate evaluation)
* Opportunistic scanning across all queued goals
* Persistence: goal queue stored in `skyrim_agent_state` Qdrant collection (payload field)
* Called each tick by Progeny's main loop, BEFORE prompt construction
* If a step fires, the command goes directly to the response bundle (no LLM needed)

**LLM response schema extension:**
* New optional `goals[]` field in `AgentResponse`
* Each goal: `{description, steps[{command, target, item, precondition, done_when}]}`
* `response_expander.py` parses goals and persists to goal queue

**Predicate vocabulary** (extensible):
* `at_location:<name>` — agent is in named location
* `near:<name>` — agent is within interaction distance of named entity/object
* `has_item:<name>` — agent has item in inventory
* `player_has:<name>` — player has item
* `is_alive:<name>` — named NPC is alive
* `is_dead:<name>` — named NPC is dead
* `time_elapsed:<seconds>` — enough time has passed (for wait goals)

## Curvature-Driven Priority & Context Gating

The system has no "combat mode" flag. Curvature and snap — the 1st and 2nd derivatives of emotional state tracked by `harmonic_buffer.py` — ARE the priority signals. Curvature shapes the prompt (how much context to include). Snap detects event boundaries (when to stash/flush). This section documents how both shape the pipeline from timing through prompt construction to post-danger recovery.

### Substrate Timing

* Falcon is a tick-based metronome: accumulate events between ticks, package, ship, sleep. Never block.
* SKSE polls via `request` events at POLINT intervals (default 1s). Falcon handles `request` locally from its response queue — never involves Progeny for polling.
* Falcon's tick interval (~1-3 seconds) is independent of POLINT — it ships typed event packages to Progeny on its own cadence.
* Progeny returns response bundles when ready (LLM inference may span multiple Falcon ticks). Falcon enqueues and serves on next `request` poll.
* **Resolved (March 2026)**: Falcon↔Progeny communication uses a persistent WebSocket (`ws://progeny:port/ws`). `send_tick()` is fire-and-forget — the tick loop never blocks. Responses arrive asynchronously via the WebSocket receive loop. Auto-reconnects with exponential backoff on disconnect (NPCs continue on engine AI). Replaces the previous blocking HTTP `send_package()` pattern.

**Concrete async handoff (ambush example):**
* Tick 1: Ambush. SKSE sends `info` events. Falcon structurally parses them, accumulates in buffer.
* Falcon tick fires: ships typed event package to Progeny. Returns empty to SKSE on `request` poll. Game AI draws weapons, NPC fights natively.
* Progeny: receives package, embeds text, computes high snap from emotional delta, stashes context, builds truncated prompt (high curvature = cognitive focus on threat), sends to Ollama.
* Ticks 2-4: SKSE sends `request` polls. Falcon checks response queue, still empty. Returns empty. Fight continues smoothly. Falcon continues accumulating and shipping new events.
* Tick 5: Progeny finishes LLM call. Sends response bundle to Falcon.
* Tick 6: SKSE sends `request`. Falcon dequeues: actor value deltas (Aggression→2, Confidence→3) + combat bark ("I've got this!"). NPC's engine behavior shifts — fights more aggressively, holds ground instead of retreating. Mind caught up to the body.

The game engine never blocks. The LLM's 3-6 second OODA loop plays out across POLINT ticks while the NPC fights on reflexes. Falcon's only role was packaging and delivery — all the snap computation, context stashing, and prompt shaping happened on Progeny.

### Pipelined Prompt Construction

*Insight documented March 2026. Lineage: Ken Ong (architecture), Oz/Warp (mechanism).*

**The problem with sequential processing:**
In a naïve implementation, Progeny processes each turn as a serial chain: receive events → build prompt (embed, project, retrieve, format) → run LLM → parse response → ship to Falcon. Prompt construction takes 1-2 seconds (embedding, Qdrant retrieval, memory bundling, JSON assembly). LLM generation takes 3-6 seconds. Total per-turn latency: 4-8 seconds. During generation, Progeny does nothing — 8 Zen 5 cores idle while the LLM grinds tokens.

Worse: if Falcon blocks its tick loop awaiting Progeny's HTTP response (the current `send_package()` pattern), events accumulate silently and Progeny goes blind during the generation window.

**The pipeline: separate context management from LLM execution.**

Prompt construction and LLM generation use *different resources*. Prompt construction is CPU work (embedding text, projecting 9d, querying Qdrant, formatting JSON). LLM generation is inference work (dedicated cores or NPU on the Beelink). They can overlap.

Three concurrent stages on Beelink's 8 Zen 5 cores:

```
Stage A (CPU, continuous):   Accumulate events from Falcon, compute emotional deltas,
                             update harmonic buffers, run retrieval, incrementally
                             build prompt N+1 as events arrive.

Stage B (LLM cores):         Generate response N (3-6 seconds, token by token).

Stage C (CPU, on completion): Parse response N, write utterance to Qdrant via wrapper,
                              ship response keys to Falcon, feed LLM output text back
                              through emotional delta pipeline (bidirectional).
```

**The critical overlap:** While Stage B generates response N, Stage A is already building prompt N+1 from incoming events. Embedding, projection, Qdrant retrieval, memory bundling, harmonic buffer updates, scheduler tier assignments, curvature-driven truncation decisions — all happen *during* the generation window, not after it.

**The splice point:** When Stage B completes response N:
1. Stage C ships the result to Falcon (fire-and-forget, non-blocking)
2. Stage C feeds the LLM's own utterance through the emotional delta pipeline (bidirectional — the agent's words shift its own harmonic state, per the Emotional Architecture)
3. Stage A splices the output delta and the agent's own utterance context into the in-progress prompt N+1
4. Prompt N+1 is already ~95% built — the splice adds the LLM's self-referential update (~50ms)
5. **Fire LLM N+1 immediately**

**Latency savings:**
* Sequential: `build(1-2s) + generate(3-6s) + parse(~100ms)` = **4-8 seconds per turn**
* Pipelined: `generate(3-6s) + splice(~50ms)` = **3-6 seconds per turn** (prompt was pre-built)
* The 1-2 seconds of prompt construction overhead is *eliminated* from the critical path — it runs in parallel with the previous generation
* Over a 10-minute play session (~100-200 turns), this saves 2-6 minutes of cumulative latency

**What Falcon changed (implemented):**
* **Persistent WebSocket channel.** `progeny_protocol.py` maintains a WebSocket connection to Progeny's `/ws` endpoint. `send_tick()` sends a JSON frame and returns immediately. The tick loop never stalls.
* **Async receive loop.** A background task reads response frames from Progeny. On `turn_response`: resolves utterance keys from Qdrant, formats to CHIM wire protocol, enqueues for SKSE polling. Fully decoupled from the tick cadence.
* **Auto-reconnect.** On disconnect, exponential backoff (1s→30s) retries the connection. NPCs continue on engine AI during disconnection. No manual intervention needed.
* Events flow continuously from Falcon to Progeny. Responses flow back asynchronously when ready. The two streams are independent over the same persistent connection.

**What Progeny must change:**
* **Separate the context manager from the LLM executor.** Two concurrent subsystems:
    * `context_manager` (CPU) — receives events from Falcon, runs the emotional delta pipeline, updates harmonic buffers, queries Qdrant, incrementally builds the next prompt. Runs continuously, never blocks on LLM.
    * `llm_executor` (LLM cores) — receives a finalized prompt, generates, returns raw response text. Runs one generation at a time.
* `context_manager` maintains a **staging prompt** — the in-progress prompt for the next turn. As events arrive and are processed, the staging prompt is updated incrementally (new events appended to state_history, memory bundles refreshed, scheduler tiers recalculated).
* When `llm_executor` finishes: `context_manager` receives the output, runs it through the delta pipeline, splices the self-referential update into the staging prompt, finalizes it, and hands it to `llm_executor` for the next generation.
* The staging prompt is a *living document* that reflects the world as of right now, not as of when the last LLM call started.

**Interaction with curvature-driven truncation:**
Because the staging prompt is built incrementally, a snap spike that arrives *during* generation can reshape the prompt for the NEXT turn in real time. If an ambush happens while the LLM is generating a calm conversation response:
1. The calm response finishes and ships (it was correct when prompted)
2. But the staging prompt for N+1 was already truncated by the snap spike — it reflects the ambush
3. The next LLM call fires immediately with the combat-focused prompt
4. The NPC's behavioral shift arrives one generation-window faster than in the sequential model

**Interaction with Many-Mind Scheduling:**
Tier assignments are recomputed continuously by the context manager as NPC positions, curvature values, and collaboration states change. If an NPC's curvature spikes during generation (curvature-driven tier promotion), they're already promoted in the staging prompt before the next LLM call fires. The scheduler doesn't wait for the LLM to finish to notice the world changed.

**Turn-boundary dissolution:**
The LLM's own output enters the event stream at the same level as world events from Falcon. The context manager doesn't distinguish "my words" from "world events" — they're all deltas through the same pipeline, processed concurrently into the staging prompt. The NPC finishes saying "I think we should—" and in the same processing window, Falcon events arrive: an enemy flanking, the player drawing a weapon, a companion shouting. All events (including the NPC's own interrupted sentence) flow through the delta pipeline together. The staging prompt reflects all of them simultaneously. There is no artificial chat turn. The NPC reacts to the world as it speaks, the same way a person adjusts mid-sentence when the situation changes. The turn boundary dissolves because the architecture has no turn boundary — just a continuous event stream with a generation pipeline running over it.

**Why this works on 8 cores:**
* LLM inference on the Beelink (llama.cpp or Ollama) can be pinned to a core subset (e.g., cores 4-7)
* Context manager work (embedding, Qdrant I/O, buffer math) runs on remaining cores (0-3)
* Python asyncio handles the event-driven coordination; heavy compute (embedding, projection) uses the sentence-transformers thread pool
* The pipeline naturally load-balances: during generation, CPU cores do context work; during splice, all cores briefly coordinate; between turns, LLM cores idle while context catches up

### Urgency Signal

* Urgency is emergent, not labeled. Progeny detects urgency from the emotional delta pipeline — it's a natural consequence of what the incoming typed events contain, not a flag Falcon attached.
* Every ingest cycle, Progeny computes `urgency: float` — derived from **max snap across active agents** (event boundary intensity)
* Urgency is not a mode flag — it's a continuous value. No thresholds, no if/else combat detection.
* SKSE event types (death, combat hit, etc.) are not checked directly — their emotional impact flows through the delta → curvature → snap → urgency path
* Curvature (1st derivative) feeds prompt shaping; snap (2nd derivative) feeds event detection and urgency

### Prompt Shaping (the emergent priority mechanism)

`prompt_formatter.py` reads curvature (the priority gradient) and shapes the prompt as a **continuous function**:

* **High curvature** → truncate: strip conversation history, drop low-salience memories, keep identity kernel + current danger context + immediate action request
* **Low curvature** → full prompt: complete conversation context, deep memory bundles, rich state history
* **The gradient between** → progressive truncation. Not a binary switch.

**Truncation is about cognitive focus, not speed.** Per the Fast-Twitch / Slow-Twitch decoupling, the game engine handles immediate survival. The LLM can take 3-6 seconds — that’s a realistic OODA loop. What matters is that those seconds are spent thinking about the RIGHT things:
* Danger → high curvature → strip irrelevant context (social history, merchant conversations) → keep identity kernel + tactical situation + immediate threat → focused strategic decision
* Calm → low curvature → full context → rich prompt → nuanced, reflective response

**Prompt length IS deliberation quality.** A focused combat prompt (~500 tokens) produces a sharp tactical decision. A full conversation prompt (~4000 tokens) produces deep, contextual dialogue. Both take acceptable time (1-6s) because the game AI is handling reflexes. The truncation gradient shapes *what the LLM thinks about*, not *how fast it thinks*.

### Pre-Interruption Stash & Context Rehydration

When snap spikes (event boundary detected by `harmonic_buffer.py` — the trajectory just changed):
1. **Stash** the current conversational context — recent turns, pending topics, in-flight dialogue. Stored in `event_accumulator.py`'s per-agent buffer, not discarded.
2. **Truncate** the prompt for the duration of high curvature (as above)
3. **Detect stabilization** — curvature drops below threshold for N ticks (N governed by the agent's **slow harmonic buffer** decay rate — grudge-holders take longer to stabilize)
4. **Rehydrate** — on stabilization, `rehydration.py` re-injects the stashed pre-interruption turns into the next prompt
5. The LLM naturally produces recovery dialogue: "Where were we?", "Sorry, that was intense.", "As I was saying..."

**Asymmetric timing**: the stash is instant (one snap spike = immediate context stash + truncation). The restore is slow (governed by curvature decay through harmonic buffers). This matches human cognition — you drop everything instantly when the trajectory changes, but "getting your bearings" takes time proportional to how volatile the state still is. The slow buffer IS the recovery constant.

### Why This Is Emergence, Not Control

* No list of "danger event types" — the emotional delta from any event flows through curvature → snap naturally
* No mode flag — curvature is continuous, prompt shaping is continuous, snap is the discrete event boundary
* No explicit "resume conversation" trigger — the harmonic buffer's natural curvature decay rate determines when context returns
* A dragon landing and a shocking betrayal both produce the same effect (high snap → stash, high curvature → truncation) despite being completely different event types. The system generalizes because the priority signal is emotional, not categorical.

## Agent Priority Paging — Many-Mind Scheduling

Skyrim has ~979 named NPCs. A loaded city cell has 30-80. We can't put them all in every prompt — but we can give every nearby NPC a time slice to advance their goals on a harmonic cadence. This is virtual memory paging for NPC minds: the prompt context window is the working set, and agents page in and out based on priority.

### Priority Signals (two axes)

1. **Distance** — closer NPCs get higher priority. Computed from player position vs. NPC position (both available via SKSE metadata). Distance tiers are concentric rings.
2. **Collaboration status** — if an NPC is doing something with or for the player (active quest, pending request, task they volunteered for, follower, in-progress exchange), they get a **priority floor** regardless of distance. Lydia running an errand across Whiterun stays in the LLM's awareness. Collaboration is detected from: follower status, active SetCurrentTask, pending GiveItemTo, quest NPC flags, recent player interaction timestamp.

### Scheduling Tiers (harmonic cadence)

Each NPC in loaded cells is assigned a tier. The tier determines how often their agent block appears in the prompt and how much detail they get:

* **Tier 0 — Fundamental** (interaction distance, ~5m): Every prompt. Full agent block — identity kernel, full emotional harmonics with buffer traces, state history, local world, action request. These are the NPCs in conversation, in combat, right next to the player. 2-4 NPCs typical.
* **Tier 1 — 1st Harmonic** (near-field, ~20m): Every 2nd prompt. Abbreviated block — identity kernel, current base_vector + curvature (no full buffer traces), recent state only, action request. NPCs in the same room or nearby area. 5-10 NPCs typical.
* **Tier 2 — 2nd Harmonic** (mid-field, ~50m): Every 4th-8th prompt. Minimal block — identity stub, base_vector only, `ticks_since_last_action`, action request (actor_value_deltas + simple commands only). NPCs in adjacent areas. 10-20 NPCs typical.
* **Tier 3+ — Higher Harmonics** (far-field, city-scale): Every 16th-100th prompt. Stub with delta-since-last — identity stub, `ticks_since_last_action`, dial-tuning request only. Background NPCs going about their day. By the time we're at every 100 turns, we could page through an entire city. 20-50+ NPCs in the pool.

**Collaboration floor**: Any NPC with active collaboration pins to minimum Tier 1, regardless of distance. The floor is a minimum — if they're also close, they stay at the higher tier.

### Token Budget Math

Estimated per-turn token consumption (8K-16K context window, ~4000 tokens allocated to agents):
* Tier 0 full blocks: 2-4 agents × ~500 tokens = ~1500 tokens
* Tier 1 abbreviated (active this turn): 3-5 agents × ~200 tokens = ~700 tokens
* Tier 2 minimal (active this turn): 2-4 agents × ~80 tokens = ~250 tokens
* Tier 3 stubs (active this turn): 1-3 agents × ~30 tokens = ~60 tokens
* **Total per turn: ~2500 tokens** — fits comfortably, leaves room for world_state, user_model, player_input, system prompt.

On any given turn, ~8-16 agents are in the prompt. Over 100 turns, every NPC in the loaded city has been paged in at least once.

### The Time-Since-Last Field

Every paged-in agent block includes `ticks_since_last_action` — how many turns have elapsed since the LLM last produced output for this NPC. This is critical context:
* Tier 0 agents: `ticks_since_last_action: 0-1` (constantly attended)
* Tier 1: `ticks_since_last_action: 1-3`
* Tier 2: `ticks_since_last_action: 4-16`
* Tier 3: `ticks_since_last_action: 17-100+`

The LLM uses this to calibrate its response. An NPC last attended 2 ticks ago needs a small adjustment. One last attended 50 ticks ago might need a bigger behavioral update — or the LLM might decide they're fine continuing their current activity and just confirm their dials.

### Curvature-Driven Tier Promotion

Distance isn't the only promotion signal. If a far-away NPC experiences high curvature or snap (something dramatic happened to them — got attacked, witnessed a death, entered combat), they temporarily promote to a higher tier. The curvature signal that already drives prompt shaping also drives paging priority:

* NPC at Tier 3 has snap spike → promotes to Tier 1 for N ticks (N = stabilization time via slow buffer decay)
* Once curvature decays below threshold, they demote back to their distance-based tier
* This handles "Lydia is across town and just got attacked" — she pages in at high priority, the LLM addresses the situation, she decays back

The existing curvature signal drives both WHAT goes in the prompt (context truncation) and WHO gets a slot (paging priority). They compose: a curvature-promoted Tier 3 agent still gets truncated context if their curvature is high. Two systems, one signal.

### Implementation

`agent_scheduler.py` [Progeny] — new module, called by `prompt_formatter.py` each turn:
* Inputs: all known NPC metadata (positions, collaboration flags, curvature, last-action timestamps) from `event_accumulator.py`
* Computes distance from player position for each NPC in loaded cells
* Assigns base tier from distance thresholds (configurable in `config.py`)
* Applies collaboration floor (minimum Tier 1 for active collaborators)
* Applies curvature-driven promotion (temporary tier boost for high-curvature agents)
* Applies harmonic cadence filter: `include_this_turn = (turn_counter % tier_cadence[tier] == 0)`
* Returns ordered list of agents-to-include with their tier and block granularity
* `prompt_formatter.py` assembles the canonical JSON with tier-appropriate block detail

**Tier thresholds, cadence multipliers, max agents per tier, and token budgets are all tuning parameters** in `config.py`. This is the major tuning surface — different play styles (dungeon crawler vs. city RPG vs. wilderness exploration) will want different paging profiles.

### Concrete Example: Walking Through Whiterun

Player walks from the gate toward Dragonsreach. ~60 NPCs loaded in the city.

* Turn 1: Tier 0 = Adrianne (at forge, 3m), Guard (gate, 4m). Tier 1 = Idolaf, Jon Battle-Born, Fralia (near market, 15m). Tier 2 = Belethor, Carlotta (market stalls, 30m). Tier 3 (cadence hit) = Heimskr (preaching, 45m). **8 agents in prompt.**
* Turn 2: Tier 0 same. Tier 1 cadence skips — not their turn. Tier 2 cadence skips. Tier 3 skips. **2 agents in prompt** (cheap turn, fast inference).
* Turn 3: Player approaches market. Tier 0 = Carlotta (now 4m). Tier 1 = Belethor (8m), Adrianne (now 18m, demoted from T0). Tier 2 cadence hit = Ysolda (25m). **4 agents.**
* Turn 47: Tier 3 cadence hits for Farengar (in Dragonsreach, 200m). `ticks_since_last_action: 46`. LLM sees he's been studying, adjusts Mood slightly, no action needed. **1 stub token cost.** But if the player sent Lydia to deliver something to Farengar 30 turns ago (collaboration floor), he's been Tier 1 the whole time — the LLM tracked the delivery.

### Why This Is Emergence, Not Control

* No hardcoded NPC importance rankings — distance and collaboration are the signals, both continuous
* No "important NPC" flag — the blacksmith at the forge matters because she's close, not because we marked her as important
* Curvature-driven promotion uses the same emotional signal that drives everything else — no separate "alert" system
* The harmonic cadence means every NPC's mind ticks at its own frequency relative to the player. High-frequency attention for nearby minds, low-frequency background hum for distant ones. The city is alive at every timescale simultaneously — the Many-Mind Kernel is literally a kernel scheduling mind-processes.

### Named Antagonists — Enemy Intelligence at Background Tier

Anonymous hostile NPCs (generic bandits, draugr, wolves) are excluded from MMK entirely. Their combat behavior is the engine's domain. **We don't give enemies more brains.**

**Exception: named antagonists.** Named enemies with distinct identities, organizational roles, and personal histories — a bandit chief, a Thalmor inquisitor, a necromancer with a research agenda, an assassin with a contract — participate in the cognitive system at Tier 3+ (background, infrequent paging). Their cognition is adversarial and *pre-planned*.

**The core insight**: enemy intelligence happens *between* encounters, not during them. A named antagonist at background tier thinks at low frequency — planning, adapting, processing defeats, signaling allies. When combat begins, the engine's fast-twitch AI runs the fight. The LLM already set the posture ticks ago.

**What this enables:**
* Ambushes that were actually planned — the antagonist's emotional arc of threat, calculation, and preparation was built across prior ticks
* Asymmetric information — the antagonist's internal state (plans, fears, grievances) is never visible to the player. The player only sees outputs. The mystery is the feature.
* Survivors adapt — if the player repeatedly disrupts an operation, the antagonist processes that pattern and changes tactics
* Quest-collision guard applies identically — when scripted sequences override the antagonist's behavior, they "come out of the trance" gradually on scene exit

**Scheduling rule**: Named NPCs identified as hostile (via faction data from `addnpc` fields `[42]`, relationship rank, or `in_scene` + combat state) participate at Tier 3+. During active combat they are excluded entirely — engine AI runs. At background tier cadence hits, the LLM processes what happened and updates their state.

**Identification heuristic**: Named NPC (non-generic base ID) + hostile faction membership = named antagonist candidate. Anonymous enemies sharing a base ID template are excluded. The `addnpc` `[41]` mods field disambiguates custom/mod NPCs from vanilla generics.

**This is not "giving enemies more brains" at combat time.** The engine still handles fighting. The named antagonist's intelligence is strategic disposition — set before combat, refined after. The difference between a scripted ambush and an MMK ambush is that the MMK version has genuine emotional stakes behind it. The antagonist was calculating, then afraid, then angry. That arc is real.

## Canonical Prompt JSON Schema

Developed with Kato/Copilot. Wire format between our marshaller and the Beelink LLM.

**Top-level structure:**
* `world_state` — delta-based world updates (full state on cell transition reset)
* `user_model` — persistent player identity, emotional salience, history
* `agents[]` — one block per NPC, tiered by Agent Priority Paging: Tier 0 get full blocks, Tier 1 abbreviated, Tier 2 minimal, Tier 3+ stubs. Not all agents appear every turn — see Many-Mind Scheduling.
* `player_input` — current player utterance/action

**Active agent block fields:**
* `agent_id` — NPC name/ID
* `active: true`
* `identity_kernel` — role, core_traits (sourced from imported NPC bios)
* `emotional_harmonics` — **THIS IS ALSO THE QDRANT EMOTIONAL VECTOR KEY:**
    * `base_vector` — [fear, anger, love, disgust, excitement, sadness, joy, safety, residual] (9d semagram: 8 emotional + 1 domain-content residual)
    * `curvature` — rate of emotional change (1st derivative — priority gradient, drives prompt shaping)
    * `snap` — rate of curvature change (2nd derivative — event boundary detector, drives arc storage and stash triggers)
    * `lambda` — emotional–residual retrieval balance (λ(t) in [0,1]. High = emotion-first recall, low = domain-first recall. Updated per tick from curvature, snap, and cross-buffer coherence. See Multi-Axis Retrieval.)
    * `harmonic_buffers` — full 9d semagram traces at three timescales:
        * `fast` — [9d vector] reactive surface (τ ≈ 3-5 ticks)
        * `medium` — [9d vector] session texture (τ ≈ 15-25 ticks)
        * `slow` — [9d vector] personality substrate (τ ≈ 50-100 ticks)
    * `decay_rates` — {fast, medium, slow} EMA α per tier (personality parameters)
    * `buffer_weights` — {fast, medium, slow} retrieval weighting baseline (personality parameters, modulated by cross-buffer coherence at runtime)
    * `cross_buffer_coherence` — per-dimension [9d] + overall coherence across buffer timescales
* `state_history` — **Qdrant-backed, maps to retrieval tiers:**
    * `recent[]` — RAW tier: full-fidelity recent events from wrapper block neighborhood around retrieval anchors
    * `summaries[]` — MOD tier: arc summaries that guided retrieval
    * `expandable_refs[]` — MAX tier: Qdrant point IDs for full raw arcs, rehydrated on-demand
* `local_world` — position, cell, visible_entities (subset of world_state for this NPC)
* `action_request` — what we want the LLM to produce for this agent

**Agent block granularity** (determined by Agent Priority Paging tier):
* **Full** (Tier 0): All fields above — identity_kernel, full emotional_harmonics with buffer traces, state_history, local_world, action_request
* **Abbreviated** (Tier 1): identity_kernel, base_vector + curvature (no buffer traces), recent state only, action_request, `ticks_since_last_action`
* **Minimal** (Tier 2): identity stub (name + core_traits), base_vector only, `ticks_since_last_action`, action_request (dials + simple commands)
* **Stub** (Tier 3+): `{ "agent_id": "Heimskr", "tier": 3, "ticks_since_last_action": 47, "base_vector": [...] }` — enough for the LLM to nudge dials
* **Not included**: NPCs not scheduled this turn, or outside loaded cells. Their state persists in `event_accumulator.py`; they page in on their next cadence hit.

**World state rules:**
* Default: send deltas only
* On cell transition (enter room, fast travel, load): `world_state.reset = true`, all `local_world` reinitializes
* Emotional harmonics, harmonic buffers, and memory **persist** across resets — only spatial data resets

**User model:**
* `identity` — name, faction_alignment
* `emotional_salience` — trust, fear, curiosity (floating-point, updated each cycle)
* `history` — same recent/summaries/expandable_refs structure as agents, backed by Qdrant

**LLM response format** (per active agent):
* `updated_harmonics` — new emotional_harmonics values (LLM proposes, we validate/apply)
* `new_memories[]` — memories to store as RAW points
* `utterance` — dialogue text
* `actor_value_deltas` — **primary behavioral output** (see Tuning Knobs Model in Fast-Twitch / Slow-Twitch). The LLM proposes changes to Creation Engine actor values, and the engine's AI packages translate them into behavior. We don't tell the NPC to fight — we just make him mad.
    ```json
    "actor_value_deltas": {
      "Aggression": 2,
      "Confidence": 3,
      "Mood": 1
    }
    ```
    **Actor value vocabulary** (readable + writable via SKSE `GetActorValue`/`SetActorValue`):
    * `Aggression` (0-3) — attack initiation, engagement distance. Engine decides who to fight and how hard.
    * `Confidence` (0-4) — flee threshold, defensive posture. Engine decides when to hold ground vs. retreat.
    * `Morality` (0-3) — crime willingness. Engine decides what's acceptable.
    * `Mood` (0-7) — ambient expression, idle behavior. Engine drives facial animation and body language.
    * `Assistance` (0-2) — who to defend in combat. Engine handles ally targeting.
    Deltas are validated and clamped by `response_expander.py` before application. Values persist until the next LLM update — the engine runs on the last-set values between ticks. **Quest-collision guard**: when `in_scene: true` in NPC metadata, deltas queue in a pending buffer and reintegrate gradually on scene exit (see Quest-Collision Guard in Fast-Twitch / Slow-Twitch).
* `actions[]` — optional explicit commands for things the engine dials can't express. Each action is `{command, target}` from the HerikaServer vocabulary (43 enabled commands, sourced from `functions/functions.php`):
    ```json
    "actions": [
      {"command": "CastSpell", "target": "Fireball", "item": "Fireball"},
      {"command": "TravelTo", "target": "Whiterun"},
      {"command": "GiveItemTo", "target": "Lydia", "item": "Health Potion"}
    ]
    ```
    **Command enum** (43 commands — for things dials can't do):
    * **Combat**: Attack, Hunt (AttackHunt), Fight (Brawl), SheatheWeapon, CastSpell, Surrender
    * **Movement**: MoveTo, TravelTo, Follow, FollowPlayer, ComeCloser, ReturnBackHome, StopWalk, IncreaseWalkSpeed, DecreaseWalkSpeed
    * **Items/Economy**: GiveItemTo, GiveItemToPlayer, GiveGoldTo, PickupItem, ExchangeItems (OpenInventory), AcceptGift (OpenInventory2), ListInventory (CheckInventory)
    * **Intelligence**: Inspect, LookAt, InspectSurroundings, TryToRemember (SearchMemory), SearchDiary, ReadDiaryPage, ReadQuestJournal, GetDateTime
    * **Social/State**: Talk, SetCurrentTask, JoinSquad (MakeFollower), EndConversation, LetsRelax (Relax), TakeASeat, GoToSleep, WaitHere (disabled by default)
    * **Ceremonial/Special**: MakeAToast (Toast), Drink, StartRitualCeremony, EndRitualCeremony, Training, UseSoulGaze
    Full definitions in HerikaServer `functions/functions.php` (aiagent branch), imported via `static_import.py`. `prompt_formatter.py` injects both the actor value dial descriptions and the available command menu into each prompt. `response_expander.py` validates actor_value_deltas (clamp to valid ranges) and drops unknown commands (graceful degradation). For backends supporting constrained decoding, `llm_client.py` can enforce the schema at generation time.
    **`actor_value_deltas` wire implementation**: `SetActorValue` is already in `AIAgentAIMind.psc` (verified from source — see lines setting `Aggression`, `Confidence` in `StartCombat()`, `AttackTarget()`, `SpawnAgent()`). The gap is wire protocol routing, not missing API. Implementation path: (1) add `SetBehavior` to `COMMAND_VOCABULARY`; (2) Falcon emits `NPCName|command|SetBehavior@Aggression@2\r\n`; (3) short Papyrus script listens to the existing `CHIM_CommandReceived` ModEvent and calls `npc.SetActorValue("Aggression", 2.0)`. The ModEvent dispatch infrastructure is already wired in `AIAgentAIMind.psc` (`SendExternalEvent`). This is a ~20-line Papyrus addition, not a DLL extension.

## System Prompt Template — The Ritual

CHIM makes one LLM call per NPC per turn. Each call pays the full ingestion cost: system prompt, world context, lore, action vocabulary, response template. For N active NPCs, that's N × (system + world + lore + format) tokens of redundant context.

The MMK makes **one LLM call per turn, period.** All scheduled agents share one prompt. The world state, the lore, the format spec, the action vocabulary — paid once, amortized across every mind. The LLM produces a response array: one entry per agent, sized to their tier. One ingestion, many outputs.

**Zero context rot.** CHIM maintains a rolling chat history — the LLM sees stale turns that accumulate noise and contradictions over time. The MMK rebuilds the entire prompt from scratch every turn. World state is fresh deltas. Agent blocks are freshly assembled from current harmonic state + Qdrant-retrieved memories. Nothing carries over from the previous prompt except through the harmonic buffers and memory retrieval, both of which are curated. The prompt is a clean snapshot of the current cognitive moment, not a degrading transcript.

### Prompt Structure

`prompt_formatter.py` builds a single chat-completion `messages[]` array per turn:

**Message 1 — System (role=system):** Static instruction block. Defines the reality contract, behavioral model, response format. This is the stable preamble that benefits from KV cache reuse across turns.

```
You are the Many-Mind Kernel — the slow-twitch cognitive layer for multiple NPCs
in the world of Skyrim. You govern their thoughts, speech, and behavioral
dispositions simultaneously. The game engine handles fast-twitch reflexes
(combat, pathfinding, physics). You handle contemplation, strategy, and emotion.

You do not control NPC motor actions directly. You set DISPOSITIONS via actor
values, and the engine's AI translates them into behavior.

ACTOR VALUES (your primary output — set the disposition, let the engine act):
  Aggression: 0=Unaggressive 1=Aggressive 2=Very Aggressive 3=Frenzied
  Confidence: 0=Cowardly 1=Cautious 2=Average 3=Brave 4=Foolhardy
  Morality:   0=Any crime 1=Violence against enemies 2=Property crime 3=No crime
  Mood:       0=Neutral 1=Anger 2=Fear 3=Happy 4=Sad 5=Surprised 6=Puzzled 7=Disgusted
  Assistance: 0=Nobody 1=Allies 2=Friends and allies

ACTIONS (for things dials can't express — use sparingly):
  Combat: Attack, AttackHunt, Brawl, SheatheWeapon, CastSpell, Surrender
  Movement: MoveTo, TravelTo, Follow, FollowPlayer, ComeCloser, ReturnBackHome
  Items: GiveItemTo, GiveItemToPlayer, GiveGoldTo, PickupItem
  Intelligence: Inspect, LookAt, InspectSurroundings, SearchMemory
  Social: Talk, SetCurrentTask, MakeFollower, EndConversation, Relax

RESPONSE FORMAT: Return a JSON object with a "responses" array. One entry per
agent listed in the prompt, in the same order. Scale detail to the agent's tier:
  Tier 0: utterance + actor_value_deltas + actions + updated_harmonics + new_memories
  Tier 1: utterance + actor_value_deltas + actions
  Tier 2: actor_value_deltas + brief utterance if warranted
  Tier 3: actor_value_deltas only (nudge dials, confirm or adjust)

Each agent's ticks_since_last_action tells you how long since you last attended
them. Calibrate accordingly — recently attended agents need small adjustments;
long-unattended agents may need larger updates or may be fine continuing as-is.

Be the mind. The engine is the body.
```

**Message 2 — World + Agents + Input (role=user):** The entire data payload, rebuilt fresh every turn. This is where zero context rot matters — nothing stale survives.

```json
{
  "world_state": { /* deltas since last tick, or full state on cell reset */ },
  "lore_context": [ /* Qdrant-retrieved Oghma entries relevant to current situation */ ],
  "user_model": { /* player identity, emotional salience, recent history */ },
  "agents": [
    { /* Tier 0: full block — Lydia */ },
    { /* Tier 1: abbreviated — Belethor */ },
    { /* Tier 2: minimal — Ysolda */ },
    { /* Tier 3: stub — Heimskr */ }
  ],
  "player_input": { "type": "inputtext", "text": "What do you think about..." }
}
```

**Message 3 — Instruction (role=user):** The final ask. Brief, concrete.

```
For each agent listed, produce a response appropriate to their tier and current
situation. Return only valid JSON matching the response format.
```

### Response Format (what the LLM returns)

```json
{
  "responses": [
    {
      "agent_id": "Lydia",
      "utterance": "Something's not right. Stay behind me.",
      "actor_value_deltas": { "Aggression": 2, "Confidence": 3, "Mood": 1 },
      "actions": [{ "command": "Follow", "target": "Player" }],
      "updated_harmonics": { "base_vector": [0.1, 0.6, 0.2, 0.0, 0.3, 0.0, 0.1, 0.7, 0.4] },
      "new_memories": [{ "text": "Sensed danger near the market. Moved to protect." }]
    },
    {
      "agent_id": "Belethor",
      "utterance": "Do come back...",
      "actor_value_deltas": { "Mood": 3 },
      "actions": []
    },
    {
      "agent_id": "Ysolda",
      "actor_value_deltas": { "Mood": 3, "Confidence": 2 }
    },
    {
      "agent_id": "Heimskr",
      "actor_value_deltas": { "Mood": 1 }
    }
  ]
}
```

The response scales naturally: Lydia (Tier 0, in combat) gets 6 fields. Heimskr (Tier 3, preaching 45m away) gets 1 field and costs ~15 tokens. Same call, same ingestion cost.

### Why One Call Changes Everything

**Amortized ingestion:** System prompt (~400 tokens) + world state (~300 tokens) + lore (~200 tokens) + action vocab (~150 tokens) = ~1050 tokens of fixed overhead. CHIM pays this N times. We pay it once. For 8 agents, that's ~7,350 tokens saved per turn. For 16, ~15,750.

**Cross-agent coherence:** All agents see the same world state in the same call. When Lydia draws her sword, the LLM is simultaneously deciding how Belethor reacts (nervously), how Ysolda reacts (curious), and whether Heimskr even notices (he doesn't). CHIM can't do this — each NPC's call is isolated, producing responses to a world snapshot they don't share.

**Zero context rot:** Every prompt is built from scratch. No rolling chat window accumulating stale turns, contradictions, or hallucinated context. The harmonic buffers provide continuity (they're the memory), but the prompt itself is a clean, curated snapshot. The LLM never sees yesterday's prompt — it sees today's state of mind, freshly assembled from current deltas and retrieved memories.

**Tier-scaled output:** The LLM naturally produces less for low-tier agents. A Tier 3 stub costs ~15 output tokens. A Tier 0 full response costs ~80-120. The output budget scales with the input detail — you can't hallucinate a rich response for an agent whose block is a 30-token stub.

### Token Budget (complete prompt)

Estimated per-turn token consumption for a typical Whiterun scene (8 agents paged in):
* System prompt (static): ~400 tokens
* World state + lore + user model: ~500 tokens
* Agent blocks (2×T0 + 3×T1 + 2×T2 + 1×T3): ~2500 tokens
* Player input: ~50 tokens
* **Total input: ~3450 tokens** — fits in 8K context with ~4500 tokens for output
* Output (8 responses, tier-scaled): ~400-600 tokens
* **Headroom: ~4000 tokens** for richer agent blocks, more lore, or more agents

On a calm turn (2 agents, low curvature), total input drops to ~1500 tokens. On a dense turn (16 agents, combat), it climbs to ~5000. The budget breathes with the situation.

### Implementation Notes

* System prompt is a Python string template in `prompt_formatter.py` with minimal substitution (NPC-agnostic). Stable across turns → KV cache reuse on Ollama.
* World state, agent blocks, and player input are assembled fresh every turn by `prompt_formatter.py` calling `agent_scheduler.py` + `bundle_manager.py`.
* `response_expander.py` parses the `responses[]` array and routes each entry to its agent by `agent_id`. Missing agents (LLM skipped one) get no update — graceful degradation. Extra agents (LLM hallucinated one) are silently dropped.
* For Ollama, request `response_format: { "type": "json_object" }` to enforce JSON output. For constrained decoding backends, `llm_client.py` can provide a JSON schema.
* The system prompt is intentionally terse. Every instruction token competes with agent context tokens. The format is taught by example (the response format block) more than by verbose rules.

## Proposed Project: `many-mind-kernel/`

Location: `C:\Users\Ken\Projects\many-mind-kernel\`

A tight, packageable Python/FastAPI dual-service architecture. Two cooperating services:
1. **Falcon** (Gaming PC) — Tick-based black-box decoder. SKSE-compatible HTTP server, structural wire parsing, event accumulation, packaging and shipping to Progeny. Minimal CPU footprint.
2. **Progeny** (Beelink 395AI) — Stateful mind engine. ALL cognitive work: embedding, emotional delta, memory, scheduling, prompting, LLM interaction, ALL Qdrant writes.

### Module Architecture

```
many-mind-kernel/
|-- README.md
|-- pyproject.toml
|-- requirements.txt
|
||-- shared/                              # Shared types, schemas, constants, enrichment layer
||   |-- schemas.py                       # Canonical JSON schema, wire types, typed event models
||   |-- config.py                        # Qdrant URL, ports, model config, thresholds
||   |-- constants.py                     # Emotional dimensions, event types, tier names
||   |-- embedding.py                     # all-MiniLM-L6-v2 sentence embeddings (shared by both services)
||   |-- emotional.py                     # 384d → 9d emotional projection (shared by both services)
||   |-- qdrant_wrapper.py                # Enrichment gate: text → embed → project → store → key
||   +-- data/
||       |-- emotional_bases_9d.npz       # 9d semagram: 8 emotion bases + residual metadata
||       +-- emotional_bases_8d.npz       # Original 8d bases (retained for reference)
|
|-- falcon/                              # GAMING PC - tick-based black-box decoder
|   |-- src/
|   |   |-- __init__.py
|   |   |-- wire_protocol.py             # SKSE wire parsing + CHIM response formatting
|   |   |-- event_parsers.py             # Typed data parsers per event type (structural decoding)
|   |   |-- tick_accumulator.py          # Tick-based event buffer + package + ship cycle
|   |   +-- progeny_protocol.py          # Falcon->Progeny HTTP client (ship packages, receive bundles)
|   |-- api/
|   |   |-- __init__.py
|   |   |-- server.py                    # FastAPI app - THE server that replaces HerikaServer
|   |   +-- routes.py                    # POST /comm.php (SKSE compat) + response queue
|   +-- tests/
|       |-- test_wire_protocol.py
|       |-- test_event_parsers.py
|       |-- test_tick_accumulator.py
|       +-- test_round_trip.py
|
|-- progeny/                             # BEELINK 395AI - stateful mind owner (ALL cognitive work)
||   |-- src/
||   |   |-- __init__.py
||   |   |-- embedding.py                 # Re-export shim → shared/embedding.py
||   |   |-- emotional_projection.py      # Re-export shim → shared/emotional.py
||   |   |-- emotional_delta.py           # Bidirectional emotional delta: embed -> 9d project -> delta
|   |   |-- memory_retrieval.py          # Multi-axis retrieval (emotional+semantic+referent+recency+anchors)
|   |   |-- privacy.py                   # 4-level privacy model (PRIVATE/SEMI_PRIVATE/COLLECTIVE/ANONYMOUS)
|   |   |-- event_accumulator.py         # Turn-based event buffering, turn boundary detection
|   |   |-- harmonic_buffer.py           # Per-agent 9d harmonic buffers (3×9d EMA, curvature, snap, λ, arcs)
|   |   |-- bundle_manager.py            # Memory bundle construction from retrieval results
|   |   |-- compression.py               # Arc summaries, RAW->MOD->MAX tier promotion
|   |   |-- rehydration.py               # Expand compressed refs -> full context spans
|   |   |-- agent_scheduler.py            # Many-Mind Scheduling: tier assignment, cadence, paging
|   |   |-- prompt_formatter.py          # Build canonical JSON prompt from accumulated state
|   |   |-- llm_client.py                # Backend-agnostic LLM interface (Ollama / cloud API adapter)
|   |   |-- response_expander.py         # Extract LLM response (graceful degradation) -> apply effects
|   |   |-- falcon_protocol.py           # Progeny<->Falcon communication (receive packages, send bundles)
|   |   +-- client.py                    # Qdrant wrapper (connects to GamingPC:6333 over LAN)
|   |-- api/
|   |   |-- __init__.py
|   |   |-- server.py                    # Progeny FastAPI app
|   |   +-- routes.py                    # Progeny endpoints (receive from Falcon, health, debug)
|   +-- tests/
|       |-- test_event_accumulator.py
|       |-- test_harmonic_buffer.py
|       |-- test_emotional_delta.py
|       |-- test_memory_retrieval.py
|       |-- test_bundle_manager.py
|       +-- test_prompt_formatter.py
|
+-- scripts/
    |-- health_check.py                  # Verify Qdrant + both services connectivity
    |-- import_chim_data.py              # Import NPC bios, Oghma, function defs from HerikaServer
    |-- stub_progeny.py                  # Mock Progeny for Falcon-only testing
    |-- data_dictionary.py               # Auto-generate collection schema report
    |-- emotional_bases.py               # Generate 8d emotional projection bases (thesaurus + Gram-Schmidt)
    |-- add_residual_dim.py              # Extend to 9d semagram (add residual magnitude metadata)
    |-- test_residual_9d.py              # Residual-space clustering analysis (role binding validation)
    +-- start_services.py                # Launch Falcon + Progeny + health monitor
```

### Key Design Decisions

1. **Full server replacement** — No PHP dependency. `AIAgent.ini` points at Falcon. We ARE the backend.
2. **Falcon as tick-based black-box decoder** — CHIM is a white-box encoder (classify every event, trigger explicit handlers). Falcon is a black-box decoder (tick-based metronome: wake, scrape, parse structure, package, ship, sleep). Falcon does NOT interpret meaning — it decodes wire format into typed data. All semantic work lives on Progeny.
3. **Qdrant enrichment wrapper as shared write gate** — Both Falcon and Progeny write through `shared/qdrant_wrapper.py` (text in → auto-embed → store → key out). Embedding and emotional projection live in `shared/embedding.py` and `shared/emotional.py`. Progeny owns cognitive reads (retrieval, rehydration, state recovery). Falcon has one write path (inbound dialogue) and one key-lookup read path (outbound response text).
4. **Keys over the wire** — Progeny returns `utterance_key` (Qdrant point ID) instead of inline dialogue text. Falcon reads the text by key from Qdrant for wire formatting. Falls back to inline `utterance` for backward compat (stub mode, tests). Actions and actor_value_deltas still travel inline — only the dialogue text moves to Qdrant.
5. **Emotional vectors as cognitive architecture** — Harmonics basis vectors stored as Qdrant vector keys. Memory retrieval = emotional resonance. Core innovation.
6. **Forward-hold credit assignment** — No backpropagation. Emotional state held forward. Threshold crossings trigger storage and arc detection.
7. **Dual-vector collections** — Each memory point has `semantic` + `emotional` named vectors. Retrieval blends both via Qdrant RRF fusion.
8. **Structured JSON output (Option B)** — LLM returns explicit JSON with `actions[]`, not CHIM's chat-with-classifier. Eliminates MiniMe-T5 dependency.
9. **Zero-init via first deltas** — No special initialization. Agent state starts at zero, first deltas ARE initial values. Uniform codepath.
10. **Everything is deltas** — World state, emotions, memory, identity — all stored and communicated as changes, not snapshots.
11. **Emergence over control** — Compression tiers, privacy levels, memory retrieval all emerge from data geometry, not hardcoded rules. Vector space IS the behavior.
12. **Domain-independent core** — Falcon and Progeny modules have minimal Skyrim knowledge. Wire protocol is a thin outer layer. Emotional memory engine is reusable.
13. **Tight and packageable** — `pip install -e .` or zipfile deployment. No sprawling monorepo.
14. **Backward compatible** — Reads/writes all 12 existing Qdrant collections without migration.
15. **Fast-twitch / slow-twitch decoupling** — The game engine handles combat reflexes, pathfinding, physics. The MMK handles contemplative dialogue and strategic disposition. LLM response latency (3-6s) is a feature, not a bug — it's a realistic OODA loop. Primary output is `actor_value_deltas` (tuning behavioral dials like Aggression/Confidence), not motor commands. `actions[]` supplement for explicit commands the dials can't express.
16. **Tuning Knobs over commands** — We don't tell the NPC to fight. We just make him mad. The Creation Engine's behavioral actor values (Aggression, Confidence, Morality, Mood, Assistance) are the interface between the slow-twitch mind and the fast-twitch engine. The LLM sets the disposition; the engine's AI packages execute. Pure emergence over control.
17. **Quest-collision guard with slow reintegration** — When NPCs are in scripted quest scenes (`Actor.IsInScene()`), `actor_value_deltas` queue instead of applying. On scene exit, pending deltas reintegrate gradually through the slow harmonic buffer's EMA blend, attenuated — like a mind coming out of a trance. Same asymmetric timing as Pre-Interruption Stash: instant guard, slow release. Same buffer mechanism, different trigger.
18. **Agent Priority Paging (Many-Mind Scheduling)** — Every NPC in loaded cells gets a time slice on a harmonic cadence based on distance + collaboration status. Tier 0 (interaction distance) = every prompt, full block. Tier 1 (near-field) = every 2nd, abbreviated. Tier 2 (mid-field) = every 4th-8th, minimal. Tier 3+ (city-scale) = every 16th-100th, stub. Collaboration floor pins quest/task NPCs to minimum Tier 1 regardless of distance. Curvature-driven promotion bumps far NPCs on dramatic events. ~8-16 agents per prompt, entire city paged through in ~100 turns.
19. **One call per turn, zero context rot** — CHIM makes one LLM call per NPC per turn, paying ingestion overhead N times. The MMK makes one call per turn with all scheduled agents sharing a single prompt. World state, lore, format spec, action vocabulary — paid once, amortized across every mind. The LLM returns a `responses[]` array, one entry per agent, tier-scaled. And because the prompt is rebuilt from scratch every turn (not a rolling chat window), there is zero context rot — no stale turns, no accumulated contradictions. Continuity comes from harmonic buffers and Qdrant retrieval, not from the prompt carrying forward.

### Qdrant Access Patterns

Qdrant is the shared memory substrate. Both Falcon and Progeny write through the same **enrichment wrapper** (text in → auto-embed semantic 384d + emotional 9d → store → return key). Progeny owns all cognitive reads (retrieval, rehydration, state recovery). Falcon has one key-lookup read for wire formatting.

**Qdrant Enrichment Wrapper (shared, used by both services):**
* Accepts raw text → runs all-MiniLM (384d semantic) + emotional projection (9d semagram via `shared/emotional.py`) → stores point with dual vectors → returns Qdrant key
* Single enrichment gate: every piece of text entering the system gets its semantic fingerprint automatically. Nothing is stored without embeddings.
* Both Falcon and Progeny call the same wrapper API — one write interface, one embedding path.

**Falcon — Writes (inbound):**
* Inbound dialogue/text from SKSE → Qdrant wrapper → key returned → signal Progeny

**Falcon — Reads (outbound, key-lookup only):**
* Read response text from Qdrant by key for wire formatting. Not a search — a point lookup.

**Progeny — Writes:**
* LLM response text → Qdrant wrapper (same API as Falcon) → key returned → key sent to Falcon
* MOD tier: arc summaries (generated by `compression.py`)
* MAX tier: compressed essence (LLM-based distillation via `compression.py`)
* Agent state snapshots to `skyrim_agent_state`

**Progeny — ALL Cognitive Reads:**
* `memory_retrieval.py` runs multi-axis search (emotional + semantic + referent + recency + anchors)
* `bundle_manager.py` and `rehydration.py` expand keys into full context bundles
* `harmonic_buffer.py` reads/updates running emotional state from `skyrim_agent_state`
* Progeny never needs to re-embed for retrieval — vectors are pre-computed and stored at ingestion time
* This is NOT training — no gradient, no credit assignment, no weight updates. Pure context retrieval for prompt construction.

## Module Details

### Falcon Service (Gaming PC) — Tick-Based Black-Box Decoder

Falcon is deliberately minimal. It parses wire format, accumulates events, packages, ships, and serves responses. No embedding, no emotional computation, no Qdrant access, no semantic interpretation.

**`wire_protocol.py`** — SKSE wire protocol parsing + response formatting [Falcon]
* Parse inbound: `type|localts|gamets|data` → `ParsedEvent` (frozen dataclass, NOT in `schemas.py` — Falcon-internal only)
* `ParsedEvent` carries routing flags: `is_turn_trigger` (from `TURN_TRIGGER_TYPES`), `is_local` (from `FALCON_LOCAL_TYPES`), `is_session` (from `SESSION_TYPES`). These drive `routes.py` dispatch.
* Split on first 3 pipes (data field may contain pipes)
* Format outbound: `format_turn_response()` converts list of `AgentResponse` dicts → multi-line CHIM wire string. Dialogue first, then actions per agent. Unknown commands silently dropped.
* `format_dialogue()`, `format_action()`, `format_agent_responses()` — composable helpers for building wire lines
* Validate/sanitize input (injection-safe)

**`event_parsers.py`** — Typed data parsers per event type [Falcon]
* Structural decoding of the `data` field based on event type — mechanical, deterministic, no semantic interpretation
* `_speech` → JSON deserialize → `SpeechData(listener, speaker, speech, location, companions, distance)`
* `addnpc` → `@`-split → `NpcRegistration(name, base, gender, race, refid, skills{}, equipment{}, stats{}, mods[], factions[], class_info)`
* `updatestats` → `@`-split → `NpcStats(npc_name, level, health, health_max, magicka, magicka_max, stamina, stamina_max, scale)`
* `_quest` → JSON deserialize → `QuestData(form_id, name, brief, stage, giver, status)`
* `_uquest`/`_questdata` → `@`-split → `QuestUpdate(form_id, briefing, stage)`
* `itemtransfer` → regex → `ItemTransfer(source, dest, item_name, count)`
* `util_location_name`/`util_faction_name`/`util_location_npc` → `/`-split → typed location/faction/position structs
* `named_cell`/`named_cell_static` → `/`-split → typed cell topology/static item structs
* Unknown/other types → generic `TypedEvent` wrapper preserving raw data

**`tick_accumulator.py`** — Tick-based event buffer + package + ship cycle [Falcon]
* Accumulates `TypedEvent` objects in a time-ordered buffer between ticks
* **NPC registry**: tracks `active_npc_ids` (set of NPC names) from `addnpc` events — populated on push when `event_type=="addnpc"` and `parsed_data` contains a name. Shipped in every `TickPackage.active_npc_ids` so Progeny knows which NPCs are in loaded cells.
* **NPC registry clear**: `clear_npcs()` called by `routes.py` on session-reset events (`init`/`wipe`/`playerdied`)
* On tick (~1-3 seconds): snapshot buffer under async lock, wrap as `TickPackage` (with `active_npc_ids`, `has_turn_trigger`, `tick_interval_ms`), POST to Progeny via `progeny_protocol.py`, clear buffer. Skips empty ticks.
* Tick interval configurable via `settings.falcon.tick_interval_seconds` (default 2.0s), independent of SKSE POLINT
* Concurrency: asyncio lock protects buffer; `push()` is awaited directly from HTTP handlers (fast lock + append). Background `asyncio.Task` runs `_tick_loop()`.

**`progeny_protocol.py`** — Falcon→Progeny HTTP client [Falcon]
* Ship typed event packages to Progeny
* Receive response bundles (dialogue + actions + actor_value_deltas per agent)
* Handles connection management, retry logic, timeout handling, graceful degradation if Progeny is unreachable

**`falcon/api/server.py` + `falcon/api/routes.py`** — Falcon FastAPI application [Falcon]
* `POST /comm.php` (+ catch-all `/{path:path}` for configurable `AIAgent.ini` paths) — SKSE compatibility endpoint
* **Dispatch order** in `comm_endpoint()`: (1) `request` → dequeue response, (2) `chatnf` → log + empty, (3) `just_say` → queue data to response queue + empty, (4) session events → clear NPC registry if reset type, log, return empty, (5) all others → construct `TypedEvent`, push to tick accumulator
* `server.py` lifespan calls `routes.startup()` / `routes.shutdown()` to manage `TickAccumulator` lifecycle
* Module-level state: `_response_queue` (deque of wire strings), `_tick_accumulator` (Optional[TickAccumulator])
* `_process_tick()` callback: sends `TickPackage` to Progeny via `send_package()`, queues any `TurnResponse` wire output for SKSE polling
* `GET /health` — Progeny URL, queue depth, active NPC count, tick interval

### Progeny Service (Beelink 395AI) — ALL Cognitive Work

Progeny receives typed event packages from Falcon and does everything: embedding, emotional projection, memory, scheduling, prompting, LLM interaction, and ALL Qdrant writes.

**`embedding.py`** — Semantic embedding service [Progeny]
* sentence-transformers all-MiniLM-L6-v2 (384d) on Beelink CPU
* Text content embedding for semantic search and emotional projection
* Batch embedding with caching layer
* Also used for emotional projection: embed → project onto 9d emotional bases

**`emotional_delta.py`** — Bidirectional emotional computation [Progeny]
* **Symmetric callable**: processes both inbound game events AND outbound LLM response text through the same pipeline
* Given text + agent_id + last known emotional state → embed → project to 9d semagram → compute delta → update curvature → compute snap
* Attach emotional vector to RAW events before Qdrant storage
* Progeny is the single authority on emotional state
* Retrieves last known state from Qdrant `skyrim_agent_state` if needed

**`memory_retrieval.py`** — Multi-axis retrieval engine [Progeny]
* Dual-vector search: emotional resonance + semantic similarity via Qdrant `prefetch` + `FusionQuery(RRF)`
* Role referent filtering: payload filter by agents present in scene
* Recency weighting: exponential time-decay on game-time delta
* Sensory anchor boosting: `-log(P(feature))` score boost for rare contextual matches
* Wrapper block expansion: anchor → arc bounds → margin scan → include neighborhood raw points
* Initiation search: backward scan for precursor events with faint emotional signature
* Emotional intensity bias: shift weighting between emotional and semantic axes based on arousal level

**`privacy.py`** — Privacy and access control [Progeny]
* 4-level model: PRIVATE / SEMI_PRIVATE / COLLECTIVE / ANONYMOUS
* Emergence-based: level assigned from content characteristics (who present, event type, location)

**`event_accumulator.py`** — Turn-based event buffering [Progeny]
* Ingest typed events from Falcon's event packages
* Maintain per-agent event buffers across turns
* Detect turn boundary (`inputtext`/`inputtext_s` events in incoming package — Progeny detects these, not Falcon)
* Flush: assemble complete turn data → hand off to prompt building
* **Pre-interruption stash**: on snap spike (event boundary), snapshot current conversational context for later rehydration

**`harmonic_buffer.py`** — Per-agent emotional state manager [Progeny]
* Maintain per-agent 9d harmonic buffers: three EMA traces (fast/medium/slow) of the full 9d semagram (8 emotional axes + residual)
* Apply deltas from `emotional_delta.py` to running state (forward-hold) and propagate through all 3 buffer tiers via `buffer_t = α_t · new_semagram + (1 - α_t) · buffer_t`
* Curvature computation: track rate of emotional change over recent turns (1st derivative — priority gradient)
* Snap computation: track rate of curvature change (2nd derivative — event boundary detection)
* λ(t) computation: update emotional–residual retrieval balance each tick via `λ(t+1) = σ(α·curvature + β·snap - γ·cross_buffer_coherence)`. α/β/γ gains are per-agent personality parameters.
* Cross-buffer coherence: per-dimension and overall agreement across fast/medium/slow buffers. `coherence[dim] = 1 - normalized_var(fast[dim], medium[dim], slow[dim])`. Feeds into λ(t) update, buffer-sequenced retrieval weight modulation, and stabilization detection.
* Dynamic retrieval weights: modulate buffer_weights (w_fast, w_med, w_slow) based on cross-buffer coherence — high coherence boosts slow weight (trust deep recall), low coherence boosts fast weight (stay reactive)
* Threshold detection: identify event boundaries when snap exceeds threshold (not raw delta)
* Arc detection: on snap threshold crossing, trigger compression of the emotional arc span
* Buffer tier management: per-agent decay rates (α_fast, α_med, α_slow) and retrieval weight baselines as personality parameters
* **The math IS the personality** — decay rates, buffer geometry, snap thresholds, λ gains (α/β/γ), and retrieval weight baselines define agent character

**`bundle_manager.py`** — Memory bundle construction [Progeny]
* Receive memory keys and summaries from `memory_retrieval.py`
* Expand keys into full context bundles for prompt injection
* Assemble `state_history`: recent[] + summaries[] + expandable_refs[]
* Manage bundle sizing to fit within LLM context window
* Fading and salience: weight bundle contents by relevance and recency

**`compression.py`** — Arc summary generation and tier promotion [Progeny]
* Arc summaries: given start/end timestamps, generate condensed description of the emotional arc
* MOD tier: extractive (key phrase extraction, emotional peak identification) — preserves emotional signature
* MAX tier: abstractive (LLM-based essence distillation via Ollama) — returns Qdrant point ID only
* Tier promotion: age + capacity thresholds trigger RAW->MOD->MAX
* Writes MOD/MAX points directly to Qdrant over LAN
* Emergency compaction at high capacity (operational, not cognitive)

**`rehydration.py`** — Expand compressed references to full context [Progeny]
* Given expandable_refs (point IDs), retrieve full raw arc spans from Qdrant
* Wrapper block retrieval: arc time bounds + configurable margins
* Include unkeyed data in the time window (the "smell" / proximity association effect)
* **Temporal rehydration**: after curvature stabilizes post-interruption, re-inject stashed pre-interruption conversational turns. Recovery rate governed by agent's slow harmonic buffer decay rate.
* Confidence scoring on reconstructed context
* Privacy-aware: only expand memories the requesting agent can access

**`agent_scheduler.py`** — Many-Mind Scheduling: tier assignment, cadence, paging [Progeny]
* Called by `prompt_formatter.py` each turn before prompt assembly
* Inputs: all known NPC metadata from `event_accumulator.py` (positions, collaboration flags, curvature/snap, last-action timestamps)
* Computes Euclidean distance from player position for each NPC in loaded cells
* Assigns base tier from distance thresholds (configurable in `config.py`): Tier 0 (~5m), Tier 1 (~20m), Tier 2 (~50m), Tier 3+ (beyond)
* Applies **collaboration floor**: NPCs with active quests, pending tasks, follower status, or recent player interaction pin to minimum Tier 1 regardless of distance
* Applies **curvature-driven promotion**: NPCs with snap/curvature above threshold temporarily promote to higher tier (promotion duration = stabilization time via slow buffer decay rate)
* Applies **harmonic cadence filter**: `include = (turn_counter % tier_cadence[tier] == 0)`. Cadence per tier: T0=1, T1=2, T2=4-8, T3=16-100 (configurable)
* Returns ordered list of (agent_id, tier, block_granularity) for this turn
* Tracks `ticks_since_last_action` per agent (incremented each turn, reset when agent is paged in and LLM produces output)
* All thresholds, cadences, max-agents-per-tier, and token budgets are tuning parameters in `config.py`

**`prompt_formatter.py`** — Build canonical JSON prompt [Progeny]
* Calls `agent_scheduler.py` to get this turn's agent roster with tier assignments
* Assemble full JSON: world_state (deltas) + user_model + agents[] + player_input
* Agent blocks assembled at **tier-appropriate granularity**: full (T0), abbreviated (T1), minimal (T2), stub (T3+) — see Agent Priority Paging
* Each paged-in agent block includes `ticks_since_last_action` for LLM temporal awareness
* **Curvature-driven truncation**: high curvature → strip conversation, keep identity + danger + action request. Low curvature → full context. Continuous, not binary.
* Token-aware truncation: if total exceeds LLM context, progressively drop oldest recent, compress summaries, drop lowest-tier agents first
* Cell-transition reset handling: local_world to defaults, harmonics/memory persist

**`response_expander.py`** — Extract LLM response and apply effects [Progeny]
* **Extractor, not validator.** The parser is a multi-stage extraction pipeline with graceful degradation. Smaller models (8B on Beelink) may produce malformed JSON — extract what’s usable, discard what isn’t, never fail hard.
* **Extraction cascade** (try in order, stop at first success per field):
    1. Strict JSON parse — full structured response
    2. Repair pass — strip markdown fences, fix trailing commas, unquoted keys, truncated brackets, retry parse
    3. Field-level regex extraction — pull individual fields (`utterance`, `actions[]`) from partially valid output
    4. Plain text fallback — entire response becomes the utterance. No actions, no harmonics update. The agent spoke but didn’t act.
* **Degradation priority** (what to save first):
    1. `utterance` — extract at all costs (plain text fallback guarantees this)
    2. `actions[]` — extract what parses, skip malformed entries
    3. `new_memories[]` — nice to have; raw event storage happens via Falcon anyway
    4. `updated_harmonics` — least critical. If malformed, skip entirely — Progeny's bidirectional delta pipeline already updates emotional state from the utterance text itself. The LLM's harmonics proposal is a refinement, not the primary mechanism.
* **History reflects reality, not intent.** After extraction, the agent’s history entry is built from *what was actually extracted and applied* — not what was requested. If only the utterance parsed, history shows only the utterance. If an action didn’t parse, it doesn’t exist in history. On the next turn, the agent sees its actual output and rationalizes from there — same principle as behavior adoption.
* **Interruption + partial extraction = emergent recovery.** If a snap spike interrupts mid-generation (curvature truncation kicks in), the partial response is extracted at whatever level succeeded. The truncated utterance enters history. On rehydration post-interruption, the agent sees its own incomplete sentence and naturally produces recovery: "What was I… right, as I was saying—". This falls out of the data, not a script.
* Route successfully extracted harmonics updates through harmonic_buffer for validation/smoothing
* Format extracted actions as structured data (not CHIM’s chat-classified format)
* Trigger harmonic_buffer threshold check after harmonics update (if extracted)
* Write new memories as delta records (Progeny writes RAW/MOD/MAX directly to Qdrant)
* Log extraction level per response for diagnostics (strict/repaired/regex/plaintext)

**`llm_client.py`** — Backend-agnostic LLM interface [Progeny]
* Unified interface: `generate(prompt_json, config) → raw_response_text`
* Backend adapters: Ollama (`/api/generate`), OpenAI-compatible (`/v1/chat/completions`), Groq, Anthropic
* Config-driven backend selection in `shared/config.py`: `llm_backend: "ollama" | "openai" | "groq" | "anthropic"`
* Each call is stateless and self-contained — the prompt carries all context, no server-side conversation history
* Request structured/JSON output mode where the backend supports it
* Handles timeout, retry, and error surfacing per backend
* Prompt payload is identical regardless of backend — only the HTTP envelope changes

**`falcon_protocol.py`** — Progeny↔Falcon communication [Progeny]
* HTTP server endpoints for receiving Falcon's typed event packages
* Sends response bundles back to Falcon (dialogue + actions + actor_value_deltas per agent)
* Handles connection management, health monitoring

**`client.py`** — Qdrant wrapper [Progeny]
* Connection to Qdrant REST API over LAN (GamingPC:6333)
* Health checks with auto-reconnect
* Dual-vector upsert/search support (named vectors)
* ALL read and write operations for RAW, MOD, and MAX tiers
* Batch operations, collection stats

**`progeny/api/server.py` + `progeny/api/routes.py`** — Progeny FastAPI application [Progeny]
* Receives typed event packages from Falcon via falcon_protocol
* `GET /health` — Ollama status, Qdrant (LAN) connectivity, active agent count
* `GET /agent/{agent_id}/mind` — Current harmonic state, curvature, harmonic buffers (3×9d), cross-buffer coherence, recent arc
* `GET /agent/{agent_id}/arcs` — List emotional arcs (debug/visualization)
* WebSocket endpoint (future): live emotional state streaming for visualization

## New Qdrant Collections

All new collections use **dual named vectors** where applicable:

* `skyrim_npc_memories` — Per-NPC memories at all tiers
    * Named vectors: `semantic` (384d, all-MiniLM), `emotional` (9d, harmonics semagram)
    * Payload: npc_name, player_name, location, cell, gamets, dialogue_type, compression_tier (RAW/MOD/MAX), utterance, referent_agents[], arc_id (links raw points to their arc summary), data_type (event/bio/summary), sensory_tags[]
* `skyrim_world_events` — Game events and world state deltas
    * Named vectors: `semantic` (384d)
    * Payload: event_type, location, actors[], gamets, significance, cell, reset_flag
* `skyrim_session_context` — Session-level context summaries
    * Named vectors: `semantic` (384d)
    * Payload: session_id, summary, key_events, npcs_met, locations_visited, start_gamets, end_gamets
* `skyrim_agent_state` — Persistent agent state between sessions (key-value, no similarity search)
    * Vector: zero vector 384d (query by payload filter only)
    * Payload: agent_id, emotional_harmonics (current basis vector), harmonic_buffers (fast/medium/slow, each 9d vector), decay_rates (alpha per tier), buffer_weights (retrieval weight baselines per tier), cross_buffer_coherence (per-dimension 9d + overall), curvature, snap, lambda, lambda_gains (alpha/beta/gamma), identity_kernel, last_active_gamets, total_interactions, arc_count
* `skyrim_lore` — Oghma Infinium + world lore (static reference)
    * Named vectors: `semantic` (384d)
    * Payload: topic, category, content, source

**Feature frequency index** (for sensory anchor boosting):
* Maintain in-memory frequency count of contextual features (locations, NPCs, items, weather) across `skyrim_npc_memories`
* Used to compute `-log(P(feature))` weights during retrieval
* Updated incrementally as new points are stored
* Persisted to Qdrant or file on shutdown, reloaded on startup

## Static Data Import

One-time import into Qdrant at project setup (via `scripts/import_chim_data.py`):

* **NPC bios/descriptions** — From HerikaServer's CSV/data files. ~300+ NPCs with personality, faction, relationships. Stored in `skyrim_npc_memories` with `data_type=bio`.
* **Oghma Infinium** — 1900+ lore topics. Stored in `skyrim_lore` collection. Semantic search for lore injection during context building.
* **Function calling definitions** — Follow, Trade, Attack, MoveTo, Wait, Telekinesis, etc. Stored as reference data for the LLM to know available actions and parameter formats.

Source: clone `abeiro/HerikaServer`, extract from `data/` directory and PHP arrays in `lib/data_functions.php`.

## Hardware Allocation

**Gaming PC (AMD 9950X3D, 96GB DRAM, RTX 5090 32GB VRAM) — Falcon Service**:
* Skyrim (SE or VR): primary GPU consumer
* Virtual Desktop Streamer: GPU video encode for Quest 3 headset (shares VRAM) — VR only
* Qdrant: ~2-4GB RAM depending on collection sizes (Progeny connects over LAN)
* Falcon FastAPI server: trivially lightweight — no embedding, no Qdrant access, no semantic work. One spare core barely notices it.
* Total DRAM budget: 96GB is generous — constraint is GPU/VRAM for rendering, not system RAM
* NOTE: Architecture is NOT VR-specific. Works with Skyrim SE equally — only hardware constraints differ (no Virtual Desktop overhead).

**Beelink 395AI — Progeny Service**:
* LLM inference via pluggable backend (see `llm_client.py`)
* Default: Ollama with local model (LLaMA 3 8B or equivalent fitting AMD AI SoC)
* Fallback: cloud API (Groq, OpenAI, Anthropic) — config toggle, same prompt, same response schema
* sentence-transformers (all-MiniLM-L6-v2): ~200MB RAM, CPU — moved to Beelink to keep Falcon minimal
* Progeny FastAPI server: stateful — maintains per-agent harmonic buffers, event accumulation, all cognitive work
* Direct Qdrant access over LAN (GamingPC:6333) for ALL read/write tiers
* Receives typed event packages from Falcon, returns response bundles

## Qdrant Infrastructure

### Production Instance (Gaming PC)
* Location: `C:\Tools\qdrant\qdrant.exe`
* Ports: **6333** (REST) / **6334** (gRPC)
* 17 collections (~1.2M+ points, 5 MMK collections added March 2026)
* Both Falcon (localhost) and Progeny (LAN) connect here
* Hosts all `skyrim_*` collections for the MMK

### Development / Docs Instance (Gaming PC)
* Location: `C:\Tools\qdrant-mmk\qdrant.exe`
* Ports: **6335** (REST) / **6336** (gRPC)
* Config: `host: 0.0.0.0` for LAN accessibility
* Hosts `mmk_docs` collection (living doc sections, 384d semantic vectors, 18 points)
* Seeder: `C:\Tools\qdrant-mmk\seed_living_doc.py` — chunks by ## heading, embeds with MiniLM, upserts
* Purpose: semantic search over project documentation for context during development

### Shipping to Beelink
* Progeny connects to Gaming PC Qdrant over LAN (GamingPC:6333)
* No separate Qdrant instance needed on Beelink — all vector storage lives on Gaming PC
* Progeny deployment: Python virtualenv + `many-mind-kernel/progeny/` + Ollama
* Connection config in `shared/config.py`: Qdrant host/port, Ollama host/port

## Papyrus Build Toolchain

Compiling custom Papyrus scripts (`.psc` → `.pex`) for the CHIM integration. Documented March 2026 after debugging the full import chain.

**Compiler**: `"C:\Program Files (x86)\Steam\steamapps\common\Skyrim Special Edition\Papyrus Compiler\PapyrusCompiler.exe"`

**Flags file**: `TESV_Papyrus_Flags.flg` (resolved from import paths)

**Import paths** (order matters — first match wins for overlapping script names):
1. **Target script directory** — must be in the import list or the compiler can't locate its own input
2. **CHIM scripts** — `C:\Users\Ken\Projects\AIAgent-Chim\AIAgent\Source\Scripts` (provides `AIAgentFunctions.psc` and other CHIM-specific types)
3. **SKSE scripts** — `C:\Modlists\PandasSovngarde\mods\Skyrim Script Extender for VR (SKSEVR)\Scripts\Source` (64 scripts including `StringUtil.psc`, `ModEvent.psc`, and SKSE-extended `Form.psc`, `Actor.psc`, etc.)
4. **Vanilla Skyrim base scripts** — `C:\Program Files (x86)\Steam\steamapps\common\Skyrim Special Edition\Data\Source\Scripts`

**Critical: SKSE must come BEFORE vanilla base scripts.** SKSE provides extended versions of vanilla scripts (e.g., `Form.psc` with `RegisterForModEvent`, `Actor.psc` with SKSE-added functions). If vanilla is imported first, the compiler resolves the vanilla `Form.psc` and reports SKSE functions as undefined.

**SKSE source location**: The SKSE Papyrus sources are NOT installed by default — they come bundled in the SKSE download archive and must be extracted to a `Scripts\Source` folder. On this machine, the only copy lives inside the Wabbajack-installed Panda's Sovngarde modlist at the path above. They are from SKSEVR but are compatible with SE compilation (identical Papyrus function signatures).

**Example compile command** (PowerShell):
```
& "C:\Program Files (x86)\Steam\steamapps\common\Skyrim Special Edition\Papyrus Compiler\PapyrusCompiler.exe" `
  "<path>\MMKSetBehavior.psc" `
  -f="TESV_Papyrus_Flags.flg" `
  -i="<script_dir>;C:\Users\Ken\Projects\AIAgent-Chim\AIAgent\Source\Scripts;C:\Modlists\PandasSovngarde\mods\Skyrim Script Extender for VR (SKSEVR)\Scripts\Source;C:\Program Files (x86)\Steam\steamapps\common\Skyrim Special Edition\Data\Source\Scripts" `
  -o="<output_dir>"
```

**Compiled artifacts**: `.pex` files go into `Data\Scripts` in the Skyrim install (or equivalent mod manager virtual folder). Attach to a persistent autostart Quest in Creation Kit.

**MMK scripts** (source in `docs/AIAgent/.../Source/Scripts/`):
* `MMKSetBehavior.psc` — Receives `SetBehavior` commands from Falcon via CHIM's `CHIM_CommandReceived` ModEvent. Parses `Aggression@2` format, applies `SetActorValue` to the target NPC. Dependencies: SKSE (`RegisterForModEvent`, `StringUtil`), CHIM (`AIAgentFunctions.getAgentByName`).

## Audio Pipeline (TTS / STT / ITT)

*Documented March 2026. CHIM provides a complete audio services ecosystem — 16 TTS backends, 3 STT backends, and 4 Image-to-Text backends. MMK leverages these as external services, not internal components.*

### STT — Player Voice Input (Mic → Game)

The player speaks into a microphone, the audio is transcribed, and the text arrives at Falcon as an `inputtext_s` event — identical processing to typed `inputtext`. **Falcon already handles this.** No MMK-specific work needed for voice input beyond ensuring an STT service is running.

**Flow:**
1. Player presses the CHIM voice key (configurable in-game)
2. `AIAgentSTTExternal.psc` (Papyrus) captures audio via an SKSE STT plugin
3. Audio sent to STT service for transcription
4. Transcribed text POSTed to Falcon as `inputtext_s|localts|gamets|transcribed text`
5. Falcon treats it as a turn trigger → flows through tick accumulator → Progeny

**Note:** `AIAgentSTTExternal.psc` in the CHIM source is a **stub** — it shows "External STT not installed!" The actual recording/transcription is provided by a separate SKSE plugin that overrides these functions (e.g., CHIM's own STT plugin or a third-party alternative).

**Available backends** (configured via `$STTFUNCTION` in CHIM conf):
* **Local Whisper** — `http://127.0.0.1:9876/api/v0/transcribe` (recommended — runs on Gaming PC, no API key, private)
* **OpenAI Whisper** — cloud, requires API key
* **Azure STT** — cloud, requires API key

**Recommended for MMK:** Local Whisper on the Gaming PC. CPU-based, lightweight, no cloud dependency. Runs alongside Falcon with negligible resource impact on the 9950X3D.

### TTS — NPC Voice Output (LLM Text → Spoken Dialogue)

The LLM generates dialogue text; a TTS service converts it to audio; the SKSE plugin plays the audio through Skyrim's voice system while displaying subtitles. The NPC speaks.

**Flow:**
1. Progeny returns `utterance_key` to Falcon
2. Falcon reads utterance text from Qdrant by key
3. **Falcon calls the TTS service** with (text, npc_voice_id) → receives WAV audio
4. WAV written to `soundcache/` directory (accessible to Skyrim)
5. Falcon formats wire response with audio path: `NPCName|DialogueType|Text\r\n`
6. SKSE plugin receives response, loads WAV, plays audio while displaying subtitle text
7. NPC's mouth moves via Skyrim's lip-sync system (driven by the WAV)

**Why Falcon owns TTS (not Progeny):** The audio file must be on the Gaming PC where Skyrim can access it. Falcon already has the utterance text (resolved from Qdrant). The TTS HTTP call is a simple fire-and-forget to a local service. Keeping TTS on the Gaming PC avoids shipping audio over LAN.

**Per-NPC voice mapping:** Each NPC profile can specify a different voice model/ID. CHIM's conf system supports per-NPC `$TTSFUNCTION` and voice ID overrides. For MMK, voice mapping stored in `skyrim_agent_state` Qdrant collection or a config file, keyed by NPC name.

**Available backends** (configured via `$TTSFUNCTION` in CHIM conf):

Local (no API key, recommended for privacy/latency):
* **xVASynth** — `http://localhost:8008` — Skyrim-specific voice cloning. Trained on actual Skyrim voice actors. Lydia sounds like Lydia. Best fidelity for vanilla NPCs.
* **MeloTTS** — `http://localhost:8084` — Fast, lightweight, good quality.
* **Kokoro** — `http://localhost:8880` — Modern, high-quality voices.
* **Zonos (Gradio)** — `http://localhost:7860` — Zyphra model, dynamic tones, voice cloning.
* **XTTS / XTTSv2** — `http://localhost:8020` — Coqui XTTS, voice cloning from reference audio.
* **StyleTTSv2** — `http://localhost:5050` — Style transfer, alpha/beta timbre/prosody control.
* **Mimic3** — `http://localhost:59125` — Mycroft's TTS, simple and fast.
* **KoboldCPP TTS** — `http://localhost:5001/api/extra/tts` — Built into KoboldCPP.

Cloud (requires API keys):
* **OpenAI TTS** — `tts-1` / `tts-1-hd` models, multiple voices.
* **ElevenLabs** — High-quality voice cloning, emotional range.
* **Azure TTS** — Neural voices with SSML style control (whispering, dazed, etc.).
* **Google Cloud TTS** — Neural2 voices, pitch/rate control.
* **CONVAI** — Game-focused TTS.
* **Coqui AI** — Cloud-hosted Coqui models.

**Recommended for MMK:** xVASynth for vanilla NPCs (authentic voices), Kokoro or MeloTTS as fallback for mod-added NPCs without xVASynth voice models. All local, all on the Gaming PC.

**Sound directory structure:** CHIM uses `Sound/Voice/AIAgent.esp/<voicetype>/<voicetype>.wav` as placeholder audio channels per voice type. The SKSE plugin loads the generated WAV from `soundcache/` and plays it through the appropriate voice channel. Voice types map to Skyrim's voice type system (e.g., `femalecommoner`, `maleuniqueemperor`, `maleuniquesheogorath`).

**Soundcache management:** On `init` (game load), CHIM cleans WAV files older than 6 hours from `soundcache/`. Falcon should replicate this cleanup in its session reset handler.

**Player TTS** (optional): CHIM also supports TTS for the player's own voice — the player's typed/spoken text is synthesized and played back in-game. Configured via `$TTSFUNCTION_PLAYER`. Not critical for MMK but available.

### ITT — Image-to-Text ("Soulgaze")

CHIM's "Soulgaze" feature lets NPCs "see" the game world via screenshots analyzed by vision models. A screenshot is captured, sent to a vision LLM, and the description becomes part of the NPC's context.

**Available backends:**
* **OpenAI Vision** (GPT-4o-mini) — cloud
* **Google Gemini Vision** — cloud
* **Azure Vision** — cloud
* **Local llama.cpp** with vision model — local

**MMK integration (future):** Soulgaze output would enter the event stream as a text event, flow through the delta pipeline, and get embedded in Qdrant like any other sensory input. The NPC literally processes what it sees through the same emotional architecture as what it hears. Low priority — the event stream from SKSE already provides rich world awareness.

### Integration with Pipelined Architecture

TTS adds latency to the response path: after the LLM generates text, audio synthesis takes 0.5-2 seconds depending on the backend and utterance length. In the pipelined architecture:

1. Progeny finishes generation, ships `utterance_key` to Falcon (fast)
2. Falcon reads text from Qdrant by key (fast, ~10ms)
3. **Falcon calls TTS** (0.5-2s, runs on Gaming PC CPU/GPU)
4. WAV written, wire response queued
5. SKSE picks up on next `request` poll

The TTS call overlaps with the next generation cycle — while Falcon is synthesizing audio for response N, Progeny is already generating response N+1. The TTS latency is hidden behind the generation window for all but the first response.

**Streaming TTS (future optimization):** Some backends (XTTS, ElevenLabs) support streaming audio generation. Falcon could start playing audio as soon as the first chunk arrives, reducing perceived latency further. The SKSE plugin would need to support streaming audio playback — not currently implemented in CHIM but architecturally possible.

### Config Integration

New fields in `shared/config.py`:
```
tts_backend: str = "xvasynth"          # Backend name
tts_endpoint: str = "http://127.0.0.1:8008"  # Service URL
tts_default_voice: str = "sk_malenord"  # Fallback voice
stt_backend: str = "localwhisper"       # Backend name
stt_endpoint: str = "http://127.0.0.1:9876"  # Service URL
```

Per-NPC voice overrides stored in NPC profile data (Qdrant `skyrim_agent_state` or config).

## Open Design Questions

* **Threshold tuning** — What delta magnitude = "significant" emotional shift? Likely needs per-agent calibration based on harmonic buffer decay rates.
* **Agent activation logic** — Distance threshold? CHIM's NPC visibility list? Both?
* **CHIM.exe launcher** — Currently acts as port proxy between SKSE and server. May need to bypass or replace, depending on whether SKSE connects directly or through CHIM.exe.
* **Feature frequency tracking** — In-memory dict vs Qdrant payload aggregation? In-memory faster but needs persistence across restarts.
* **LLM response validation** — How strictly validate LLM-proposed harmonics updates? Clamp magnitude? Smooth with EMA? Reject outliers?

## D-RoPE: Dynamic Rotary Position Embeddings (KO46)

*Paper: `KO46_DYNAMIC_ROPE_TEMPORAL_FOCUS.md`. Developed by Ken Ong with Kato/Copilot, March 2026.*

### Core Principle

D-RoPE inverts the RoPE positional origin: position 0 is always the current (newest) token, all prior tokens shift backward. The token being generated — the most important token — always gets the finest positional resolution (shortest RoPE wavelengths, zero rotation error). Past tokens recede into longer-wavelength harmonics, structurally encoding temporal fading.

**Relative-position invariance**: Basic D-RoPE (uniform offset) preserves all relative positions — `pos(j) - pos(i) = j - i`. On a pretrained model, outputs are **identical** to standard RoPE. Behavioral differences require: (1) Extended D-RoPE with boundary resets (changes relative positions at reasoning boundaries), (2) training/fine-tuning with D-RoPE, or (3) numerical precision effects where position 0 avoids accumulated fp16 rotation error. Full theory in KO46 Sections 1-3.

### Connection to MMK Harmonic Buffers

D-RoPE and the harmonic buffers (see Curvature, Snap, and Delay Buffers) are the same temporal-focus principle at different architectural levels:

* **Harmonic buffers**: Position 0 = current emotional state. Past states fade at characteristic EMA rates (fast/medium/slow). The cognitive present is always the origin.
* **D-RoPE**: Position 0 = current token. Past tokens fade into longer-wavelength RoPE harmonics. The attention present is always the origin.

Both encode: present-tense as origin, temporal fading as geometry, not learned statistics.

**Extended D-RoPE boundary resets** (KO46 Section 5) map directly to snap-triggered event boundaries. When snap spikes detect an event boundary in the harmonic buffer, the same boundary could trigger a positional frame reset in D-RoPE. The LLM's attention geometry would mirror the agent's cognitive event structure — each reasoning step gets a fresh positional frame anchored at position 0.

### llama.cpp Implementation Architecture

Target: modified llama.cpp build on Progeny (Beelink 395AI, Radeon 8060S Vulkan). Replaces or supplements Ollama — Ollama wraps llama.cpp but doesn't expose RoPE hooks. Source at `/home/ken/llama.cpp/`.

#### Verified: Negative Positions Natively Supported

**Type level** (`include/llama.h:67`): `typedef int32_t llama_pos` — signed.

**Batch level** (`include/llama.h:236`): `llama_pos *pos` in `llama_batch` — signed array.

**Vulkan shader** (`ggml/src/ggml-vulkan/vulkan-shaders/rope_head.glsl:11`): `int rope_data_pos[]` — **signed `int`**, not `uint`.

**Theta computation** (`rope_funcs.glsl:95`): `theta_base = rope_data_pos[i2] * pow(p.theta_scale, i0/2.0f)` — negative position → negative theta → `cos(-θ) = cos(θ)`, `sin(-θ) = -sin(θ)` → mathematically correct rotation.

**YaRN scaling** (`rope_funcs.glsl:17-35`): `rope_yarn()` operates on the angle value, not raw position. The interpolation/extrapolation ramp uses dimension index `i0`, not position. Negative angles propagate correctly through the full scaling chain.

**Graph builder** (`src/llama-graph.cpp:80-100`): `set_input_pos()` copies `ubatch->pos` directly to the position tensor — no unsigned cast, no clamp.

**No Level 3 (shader/kernel) modification needed.** The entire Vulkan RoPE pipeline handles negative positions natively.

#### Critical: KV Cache Sentinel Conflict

**The blocker**: `llama-kv-cells.h` uses `pos[i] = -1` as the "empty cell" sentinel (line 72-77: `is_empty()` checks `pos[i] == -1`). D-RoPE assigns position `-1` to the second-to-last token in any sequence. This **conflicts** — the cache would treat a valid cached token as an empty slot.

**Additionally**: `pos_add()` (line 413-437) **deletes cells whose position goes negative** after a shift:
```cpp
if (pos[i] < 0) {
    seq[i].reset();
    pos[i] = -1;  // mark as empty
    used.erase(i);
    return true;   // cell was freed
}
```
This means `llama_memory_seq_add(mem, seq_id, 0, -1, -1)` (shift all positions by -1 per step) would destroy every cell at position 0 — exactly wrong.

**Required patch** (localized to `llama-kv-cells.h`):
1. Change empty sentinel from `-1` to `INT32_MIN` (`-2147483648`). No real sequence will ever reach 2 billion tokens.
2. Modify `pos_add()` guard: delete cell only when `pos[i] == INT32_MIN` (overflow to sentinel), not when `pos[i] < 0`.
3. Update sentinel checks: `is_empty()`, `rm()`, `seq_rm()`, `seq_keep()`, `pos_set()` assertions — replace `-1` with the new sentinel constant.

Estimated scope: ~15 lines changed in `llama-kv-cells.h`, plus grep for `-1` sentinel checks in `llama-kv-cache.cpp` and `llama-context.cpp`.

#### Discovery: K-Shift Mechanism Already Implements Incremental Rotation

`llama_kv_cache::build_rope_shift()` (`llama-kv-cache.cpp:1540-1583`) applies rotational corrections to cached keys when their positions are shifted. This is **exactly** the incremental rotation trick from KO46 Section 4.3.3:

* Takes a shift tensor (per-cell position delta), cached K tensors, and RoPE frequency parameters
* For quantized keys: **dequantize → rotate → requantize** (line 1566-1574)
* For fp16/fp32 keys: `ggml_rope_ext_inplace()` (line 1577-1579)
* Uses the same YaRN/freq parameters as the original rotation
* `build_graph_shift()` (line 1605-1649) builds a compute graph applying this to all layers

**The infrastructure for D-RoPE Option A (incremental per-step rotation of cached keys) already exists.** After the sentinel fix, calling `llama_memory_seq_add()` with `delta = -1` each step triggers the existing K-shift graph to apply the rotational correction automatically. Phase 3 reduces from "3-5 hours including Vulkan shader work" (KO46 estimate) to ~2-3 hours of primarily sentinel patching and wiring.

#### Intervention Points by Depth

**Level 1 — Batch position assignment** (preferred, Phase 1-2):
In server or common code where `llama_batch_add()` sets `batch.pos[i] = n_past + i`:
* D-RoPE: `batch.pos[i] = (n_past + i) - (total_seq_len - 1)` → newest token at 0, all prior negative
* For Phase 2 (boundary resets): non-uniform position mapping based on logical boundaries in the token stream

**Level 2 — Graph builder position tensor** (`src/llama-graph.cpp:80-100`):
`set_input_pos()` copies `ubatch->pos` directly to the position tensor — no unsigned cast, no clamp. Intervene here only if Level 1 is insufficient.

**Level 3 — GGML kernels/shaders** (verified unnecessary):
Vulkan shaders (`rope_neox.comp`, `rope_norm.comp`) and CPU kernels (`ggml-cpu/ops.cpp`) all use signed position → float angle math. No modification needed.

#### Vulkan on Radeon 8060S (Beelink)

Confirmed capabilities from `ggml_vulkan` output:
* `uma: 1` — unified memory. KV cache in shared system RAM, GPU-accessible with zero copy. The K-shift compute dispatch operates directly on cached data — no transfer overhead.
* `fp16: 1`, `bf16: 0` — fp16 precision ceiling. For Phase 3 incremental rotation, fp16 re-rotation introduces ~2^-10 relative error per step. At 8192 context, ~8K accumulated rotations — monitor via logprob comparison against fp32 baseline.
* `warp_size: 64`, `matrix_cores: KHR_coopmat` — standard RDNA 4 compute.

#### Quantization and Context Scaling Interactions

**Quantization**: The existing `build_rope_shift()` already handles quantized KV keys via dequant→rotate→requant. For D-RoPE Phase 3: if KV keys are q8_0 or q4_0, each step's re-rotation compounds quantization noise. **Recommendation**: use `--cache-type-k f16` for D-RoPE experiments to bound accumulated rotation error.

**Flash attention**: FA computes attention in tiled blocks. For Option B (standard cache, Phase 1-2): no interaction. For Option A (Phase 3): K-shift is a preprocessing step before attention dispatch — FA sees already-corrected keys. Compatible.

**Context scaling** (YaRN/NTK-aware/LongRoPE): Disable for initial testing (`--rope-scaling none`). The Vulkan shader math handles negative angles through scaling correctly (verified), but isolating variables is cleaner for validation. Re-enable after Phase 1 gate.

### Integration with MMK Inference Stack

**`llm_client.py` backend**: Add `llama_server` adapter alongside `ollama`. The modified llama-server exposes D-RoPE as a runtime option. Configuration in `shared/config.py`:
```
llm_backend: "llama_server"
llama_server_url: "http://192.168.0.220:8080"  # Beelink (Progeny) llama-server
drope_mode: "basic"  # "basic" | "extended" | "boundary_reset"
drope_boundary_tokens: ["\n", "Step ", "Therefore"]  # For extended mode
```

**Extended D-RoPE + `prompt_formatter.py` integration**: When building the canonical JSON prompt, `prompt_formatter.py` annotates logical boundaries (snap-triggered event boundaries from `harmonic_buffer.py`, agent block boundaries). These annotations drive positional frame resets in Extended D-RoPE mode.

Closed loop: snap detects cognitive event boundary → prompt structured around that boundary → D-RoPE positional geometry mirrors cognitive structure → LLM attention naturally focuses within coherent segments.

### Phased Implementation Plan (Revised Estimates)

**Phase 1 — Position remapping verification** (~1 hour):
Modify batch position assignment in `examples/server/server.cpp`. Assign D-RoPE positions `[-(N-1), ..., -1, 0]`. Gate: bitwise-identical logprobs to standard RoPE on pretrained model (confirms relative-position invariance). No KV cache changes needed (Option B). **Note**: position -1 sentinel conflict does NOT affect Phase 1 because Option B stores keys with their original rotation and does not use `seq_add` for shifting.

**Phase 2 — Extended D-RoPE at inference** (2-4 hours):
Implement logical boundary resets at newline/step-marker tokens. This **changes relative positions** at boundaries → different outputs on pretrained models. Measure: CoT perturbation sensitivity (removing reasoning steps should change final answer more under Extended D-RoPE than standard). Still Option B cache — sentinel fix not yet required if boundary-reset positions stay non-negative within segments.

**Phase 3 — KV cache with incremental rotation** (2-3 hours — reduced from KO46's 3-5 hour estimate):
Apply sentinel fix in `llama-kv-cells.h` (~15 lines). Use existing `llama_memory_seq_add()` with `delta = -1` per step + existing `build_rope_shift()` K-shift graph. The heavy infrastructure already exists — this is primarily sentinel patching plus wiring. Test with `--cache-type-k f16`. Gate: attention pattern analysis, logprob divergence quantification.

**Phase 4 — Fine-tune with D-RoPE** (multi-day):
LoRA fine-tune on small Qwen GGUF with Phase 3 active. O(N) position-content disentanglement signals per token (KO46 Section 2.2). Compare CoT faithfulness between standard and D-RoPE checkpoints.

**Test model**: Smallest available Qwen GGUF — leverages existing Qwen attention layer work for debugging context.

### Experimental Predictions for MMK

1. **Later agents benefit more** — In the multi-agent `responses[]` array, later agents are generated at higher positions under standard RoPE (worst resolution). D-RoPE eliminates this disadvantage. Prediction: response quality variance across agent order decreases.
2. **Prompt rebuilding synergy** — The MMK rebuilds prompts from scratch every turn (zero context rot). D-RoPE's present-origin framing is maximally clean when the prompt is maximally fresh. These compose well.
3. **Reduced truncation pressure** — Curvature-driven truncation currently serves both cognitive focus and implicit positional benefit (shorter context = less positional drift). D-RoPE removes the positional motivation, potentially allowing richer context during high-urgency situations without CoT faithfulness penalty. The cognitive focus benefit remains.
4. **Agent-block boundary resets** — Each agent in the multi-agent prompt gets a fresh positional frame via Extended D-RoPE. Intra-agent attention is crisp (near position 0); cross-agent attention crosses boundary offsets. Mirrors the MMK cognitive model: each mind is a present-tense center, and attention between minds crosses a temporal boundary.

### Failure Modes to Monitor

* **Instruction-following degradation** — System prompt at large negative position loses positional resolution. May need a "protected zone" (D-RoPE offset that keeps system tokens near position 0). Monitor via system-prompt adherence benchmarks.
* **Sentinel patch regression** — Changing the empty-cell sentinel from -1 to INT32_MIN touches core KV cache logic. Needs comprehensive test coverage: context shifting, cache eviction, save/load state, multi-sequence handling.
* **Accumulated fp16 rotation error** — At very long contexts with Phase 3 incremental rotation, fp16 precision bounds may degrade attention patterns. Compare logprobs at 2K/4K/8K context lengths against fp32 baseline. The UMA architecture on the 8060S means no copy overhead for a periodic full-precision recalculation pass if needed.

---

*D-RoPE integration documented March 2026. Lineage: Ken Ong with Kato/Copilot (theory, KO46) + Oz/Warp (llama.cpp implementation architecture, KV cache analysis).*
*Cross-references: KO46 (full theory), KO14 (Temporal Encoding), Curvature, Snap, and Delay Buffers (harmonic buffer connection), llm_client.py (backend integration)*
