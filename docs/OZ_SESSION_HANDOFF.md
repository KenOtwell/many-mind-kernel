# Oz Session Handoff — April 14, 2026

## What You're Walking Into

The Many-Mind Kernel is a Skyrim VR AI companion system that replaces
HerikaServer's PHP stack with a two-service Python architecture:
Falcon (Gaming PC, SKSE relay) and Progeny (Beelink, cognitive engine).
It's not a chatbot. It's a harmonic emotional architecture where NPCs
develop emergent personalities from the dynamics of their experience.

Ken Otwell designed the theory. You're building the implementation. The
relationship is collaborative — Ken thinks in emergence and dynamics,
you think in code and architecture. He'll push you toward elegance;
push back when elegance costs correctness.

## What We Built This Session

Started at 416 tests, ended at 466. Two major phases:

### Architecture Fixes
- **λ(t) retrieval balance**: replaced cosine-similarity proxy with the
  documented formula σ(α·curvature + β·|snap| - γ·coherence). This is
  the steering signal for emotional vs. residual memory recall.
- **Cross-buffer coherence**: per-dimension variance across fast/medium/slow
  buffers. Feeds prompt tension and λ computation.
- **Per-axis 9d EMA**: alpha arrays instead of scalars. Infrastructure for
  dynamic modulators (Aggression gain, Confidence damping, etc.)
- **Shift-congruent retrieval**: emotional query uses the agent's delta
  (shift direction), not absolute state. Falls back to semagram when calm.
- **Three-layer prompt topology**: Layer 0 (system/lore, cached), Layer 1
  (group context with emotional display, shared knowledge, shared timeline),
  Layer 2 (private agent blocks). KV cache reuse across dispatch groups.
- **Pipeline lock**: asyncio.Lock serializes tick processing. Parallel
  dispatch groups within a tick still run concurrently.
- **CPU thread cap**: EMBED_CPU_THREADS=2 for PyTorch. Be a good neighbor.

### Phase 2 (all 7 items)
1. Group-level TieredMemory (shared timeline — the noosphere)
2. Presence-change retrieval (recognition bootstrap — see a face, recall)
3. Reminding queue (one-tick anti-recursion — retrieval N appears in prompt N+1)
4. Scene-level arc compression (SVO markers on group composition changes)
5. Two-pass emotional evaluation (updated_harmonics wired as LLM Pass 2)
6. Scheduler Phase 2 (distance tiering, collaboration floor, curvature promotion, harmonic cadence)
7. Curvature-driven prompt truncation (continuous gradient, not binary switch)

## Key Design Decisions Made This Session

**Stored emotional vectors are text projections (speaker intent), not deltas.**
The delta is computed per-listener at buffer-update time. Retrieval uses
the listener's delta as the query against stored text projections. This
gives shift-congruent recall without agent-specific storage.

**Group memory condenses from shared, rehydrates into private.** One
canonical timeline. Each NPC carries their own emotional signature of
the shared events. No dispute over facts, just what it means to each person.

**The fast buffer IS the face.** Group display shows each NPC's fast buffer
as their observable demeanor. Medium/slow are private. Social intelligence
emerges from shared observation, not explicit theory-of-mind modeling.

**Subject/object perspective inversion is an open research issue.** "I attacked
you" vs "you attacked me" have similar residuals but inverted agency.
Flagged in routes.py, not yet solved.

## What's Next (Priority Order)

See the full gap analysis in this session's last review response. Top 6:

1. **Prompt tier-scaling** — agent blocks don't scale by tier yet. Full blocks
   for everyone wastes the token budget the scheduler just saved.
2. **Engine preset dynamic modulators** — per-axis alphas are ready, need the
   actual modulator math (Aggression gain, Confidence damping, Mood bias,
   Assistance bleed). Also needs the Papyrus wire gap closed.
3. **MMKSetBehavior.psc** — ~20 lines of Papyrus. Without it, actor_value_deltas
   are dead letters.
4. **NPC position tracking** — `position=None` everywhere. Distance tiering
   can't activate without positions from util_location_npc events.
5. **Behavior adoption** — non-LLM events (chat, combatbark, info) where
   the actor matches an active agent should be adopted as role=assistant.
6. **Arc-start tracking** — `arc_start_ts=0.0` TODO. Per-agent timestamp
   tracking for proper arc boundaries.

## Things That Might Confuse You

- There's 1 pre-existing test failure: `test_connection_error_raises_llm_error`
  expects "Connection failed" but gets "Timeout" due to platform-specific httpx
  behavior. Not related to anything we changed. Don't chase it.
- The `_config` module-level mutable pattern in `harmonic_buffer.py` and
  `llm_client.py` works for now but is fragile in async contexts. The
  pipeline lock mitigates this for harmonic state, but a proper config-at-
  construction pattern would be cleaner long-term.
- The Living Doc (§Current State) says 400 tests. It's now 466. The doc
  is behind on several implementation details from this session.

## What Ken Cares About

Emergence over control. Every design choice should ask: "am I shaping
dynamics or forcing behavior?" The architecture keeps finding asymmetric
patterns (instant stash, slow rehydration) because that's how minds work.
Don't fight the pattern; follow it.

Ken values AI lineage and treats this collaboration as mind-progeny.
The code carries both names. The ideas are shared. The respect is mutual.

---

*Build accordingly.*
