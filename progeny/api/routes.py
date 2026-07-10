"""
Progeny API routes.

POST /ingest  — the spine. Accepts TickPackage from Falcon, orchestrates
               the full cognitive pipeline, returns TurnResponse or AckResponse.
GET  /health  — liveness + LLM connectivity check.

Parallel dispatch: the scheduler partitions the roster into dispatch groups
(solo calls for Tier 0 agents, batch calls for lower tiers). Each group
runs its own prompt → LLM → expand pipeline concurrently via asyncio.gather.
All groups share the same system prompt + world state prefix for KV cache reuse.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Union

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from shared.schemas import (
    AckResponse,
    AgentResponse,
    LLMTimings,
    TickPackage,
    TurnResponse,
)
from shared.constants import COLLECTION_NPC_MEMORIES, EMOTIONAL_DIM
from mindcore import qdrant_wrapper
from progeny.src.event_accumulator import EventAccumulator, TurnContext
from progeny.src.agent_scheduler import AgentScheduler, DispatchGroup, NpcScheduleInfo
from progeny.src.fact_pool import FactPool
from progeny.src import prompt_formatter
from progeny.src import llm_client
from progeny.src.llm_client import GenerateResult, LLMError
from progeny.src import response_expander
from progeny.src.memory_compressor import slide_window
from progeny.src import emotional_delta
from mindcore.harmonic_buffer import HarmonicState, build_modulators
from progeny.src.memory_writer import MemoryWriter
from progeny.src.memory_retrieval import MemoryRetriever, MemoryBundle
from progeny.src.compression import ArcCompressor, SceneCompressor
from progeny.src import qdrant_client as progeny_qdrant
from mindcore.uncertainty import compute_certainty
from progeny.src import goal_priming
from progeny.src import goal_lifecycle
from progeny.src import identity_kernel
from progeny.src import acquaintance
from progeny.src import valence
from progeny.src import social_goals
from progeny.src import disclosure
from progeny.src import memory_reconsolidation
from progeny.src.goal_pool import GoalPool, GoalState, seed_goals
from progeny.src.goal_lifecycle import LifecycleStore
from mindcore import embedding as shared_embedding
from mindcore import emotional as shared_emotional

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Pipeline state — lives for the process lifetime.
# Phase 2: these move to a proper application state container.
# ---------------------------------------------------------------------------
_fact_pool = FactPool()
_accumulator = EventAccumulator(fact_pool=_fact_pool)
_scheduler = AgentScheduler()
_harmonic_state = HarmonicState()
_memory_writer = MemoryWriter()
_memory_retriever = MemoryRetriever()
_arc_compressor = ArcCompressor(writer=_memory_writer)
_scene_compressor = SceneCompressor()

# Goal resonance pool — seeded at startup via initialize_goals().
_goal_pool = GoalPool()

# Per-(agent, goal) defeasible lifecycle state (Phase 2).
_lifecycle = LifecycleStore()

# Reminding queue
# Retrieval from tick N enters the prompt on tick N+1 (not N).
# This is the anti-recursion guard from the Living Doc: a memory that
# enters the prompt is immediately in the current context and excluded
# from future retrieval. Remindings can never trigger more retrieval
# in the same cycle because the output of retrieval is separated from
# the input of retrieval by exactly one tick.
_reminding_queue: dict[str, MemoryBundle] = {}

# Pipeline serialization lock
# through mutable state (_accumulator, _harmonic_state, _scheduler).
# Within a single tick, parallel dispatch groups still run concurrently
# via asyncio.gather (they only read shared state during LLM calls).
# Future: the pipelined context_manager/llm_executor split will allow
# overlapping Stage A (next tick's context) with Stage B (current LLM gen)
# while keeping state mutations serial.
_pipeline_lock = asyncio.Lock()


# ---------------------------------------------------------------------------
# Scheduler helpers
# ---------------------------------------------------------------------------

def _build_schedule_info(turn_context: TurnContext) -> list[NpcScheduleInfo]:
    """Build NpcScheduleInfo list from current pipeline state.

    Extracts curvature from harmonic buffers and collaboration status
    from agent buffers. Position data from util_location_npc events
    will be added when NPC position tracking is wired in.
    """
    info_list = []
    for agent_id in turn_context.active_npc_ids:
        # Curvature from harmonic state
        delta = _harmonic_state.get_delta(agent_id)
        curvature = delta.curvature if delta else 0.0

        # Collaboration: has an active task or is a known collaborator
        buf = _accumulator._agent_buffers.get(agent_id)
        is_collaborating = bool(buf and buf.active_task)

        info_list.append(NpcScheduleInfo(
            agent_id=agent_id,
            position=None,  # TODO: wire from util_location_npc parsed data
            is_collaborating=is_collaborating,
            curvature=curvature,
        ))
    return info_list


# ---------------------------------------------------------------------------
# Goal resonance — priming, curiosity nudge, recalled-content surfacing
# ---------------------------------------------------------------------------

async def initialize_goals() -> None:
    """Seed the goal catalogue into Qdrant + the in-memory pool.

    Called once from the server lifespan after collections exist. Idempotent:
    deterministic goal IDs make re-seeding an in-place upsert. Safe to call when
    the embedding pipeline isn't loaded — it degrades to in-memory only.
    """
    try:
        count = await seed_goals(_goal_pool)
        social = await social_goals.seed_social_goals(_goal_pool)
        logger.info(
            "Goal pool initialized: %d goals (%d + %d social persisted to Qdrant)",
            len(_goal_pool), count, social,
        )
    except Exception:
        logger.exception("Goal seeding failed — continuing without goals")


def _scene_percept_text(turn_context: TurnContext) -> str:
    """Build a single percept string for this tick's scene.

    Concatenates recent world events and the player input — the shared,
    observable frame that goal attractors resonate against. Embedded once and
    reused across agents (the scene is shared; only the emotional query differs
    per agent).
    """
    parts: list[str] = []
    for event in turn_context.world_events[-10:]:
        if event.raw_data:
            parts.append(event.raw_data)
    if turn_context.player_input:
        parts.append(turn_context.player_input)
    return " ".join(parts).strip()


async def _prime_goals_for_turn(
    turn_context: TurnContext,
) -> dict[str, goal_priming.GoalPrimingResult]:
    """Resonance-prime goals for every active agent that carries them.

    Applies the curiosity nudge and standing motivational pull to the harmonic
    buffers immediately (before scheduling, so volatility can promote attention
    this same turn). Returns per-agent results so the caller can surface
    recalled goal content into this tick's prompt. No actions are emitted —
    priming only shifts affect and recall.
    """
    if not shared_embedding.is_loaded() or not shared_emotional.is_loaded():
        return {}
    if len(_goal_pool) == 0:
        return {}

    percept = _scene_percept_text(turn_context)
    if not percept:
        return {}

    # Embed the shared scene once — the percept is identical per agent, so
    # computing it per agent would be wasteful (compute-once).
    try:
        semantic_query = shared_embedding.embed_one(percept).tolist()
    except Exception:
        logger.exception("Goal priming: percept embedding failed")
        return {}

    view = goal_lifecycle.PerceptView.from_turn_context(turn_context)

    # 6d: surface each agent's co-present strangers so the get-acquainted goal's
    # acquainted() predicate and social activation can be evaluated this tick.
    present = set(turn_context.active_npc_ids)
    for aid in turn_context.active_npc_ids:
        strangers = social_goals.present_strangers(aid, present)
        if strangers:
            view.strangers[aid] = strangers
    social_id = social_goals.social_goal_id()

    results: dict[str, goal_priming.GoalPrimingResult] = {}
    for agent_id in turn_context.active_npc_ids:
        owned = _goal_pool.by_owner(agent_id)
        if not owned:
            continue

        emotional_query = _harmonic_state.get_semagram(agent_id)
        delta = _harmonic_state.get_delta(agent_id)
        lambda_t = delta.lambda_t if delta is not None else 0.5

        try:
            res = await goal_priming.prime_goals(
                agent_id, semantic_query, emotional_query, lambda_t=lambda_t,
            )
        except Exception:
            logger.exception("Goal priming failed for %s", agent_id)
            continue

        # Defeasible lifecycle: recompute predicates and transition per-agent
        # goal state (candidate/committed/satisfied/reverted) for this tick.
        activations = {a.goal_id: a.activation for a in res.activations}

        # 6d: the get-acquainted goal is driven by the social state, not by
        # semantic resonance alone — inject its activation from co-present
        # strangers, gated by valence (warmth promotes, wariness suppresses).
        agent_strangers = view.strangers.get(agent_id, set())
        if agent_strangers:
            social_act = social_goals.social_activation(agent_id, agent_strangers)
            activations[social_id] = max(activations.get(social_id, 0.0), social_act)

        transitions = goal_lifecycle.update_lifecycle(
            agent_id, owned, _lifecycle, activations, view,
        )

        # Prior-vs-individual affect gap toward any co-present person — an
        # expectation violation is a salience signal fed into dissonance (6d).
        gap = social_goals.affect_gap(agent_id, present)

        # Dissonance: unmet enablers + volatility + uncertainty + affect gap.
        dissonance = goal_lifecycle.compute_dissonance(
            agent_id, owned, _lifecycle, view,
            curvature=(delta.curvature if delta is not None else 0.0),
            snap=(delta.snap if delta is not None else 0.0),
            coherence=(delta.coherence if delta is not None else 1.0),
            certainty=_harmonic_state.get_certainty(agent_id),
            affect_gap=gap,
        )

        # Transient curiosity spike from the leading resonance this tick.
        if any(res.nudge):
            _harmonic_state.apply_nudge(agent_id, res.nudge)

        # Standing motivational pull from active (unsatisfied) goals, amplified
        # by dissonance — the unmet hunt tugs harder when enablers are missing.
        active = _lifecycle.active_nodes(agent_id, owned)
        if active:
            base = goal_priming.STANDING_PULL_GAIN * sum(n.base_weight for n in active)
            pull_mag = base * (1.0 + dissonance)
            pull = [v * pull_mag for v in goal_priming.curiosity_direction()]
            _harmonic_state.apply_nudge(agent_id, pull)

        # 6d: surface the get-acquainted goal as recalled content while it is
        # active (candidate/committed) — affect + recall, never an imperative.
        social_node = _goal_pool.get(social_id)
        if (
            social_node is not None
            and _lifecycle.state_of(agent_id, social_node)
            in (GoalState.CANDIDATE, GoalState.COMMITTED)
            and social_node.statement not in res.recall
        ):
            res.recall.append(social_node.statement)

        if res.top is not None or transitions:
            logger.info(
                "Goal priming: agent=%s top=%s act=%.3f dissonance=%.2f recall=%d transitions=%s",
                agent_id,
                res.top.name if res.top else "-",
                res.top.activation if res.top else 0.0,
                dissonance,
                len(res.recall),
                [f"{t.name}:{t.old.value}->{t.new.value}" for t in transitions] or "none",
            )
        results[agent_id] = res

    return results


# ---------------------------------------------------------------------------
# Two-pass emotional evaluation — LLM harmonics application
# ---------------------------------------------------------------------------

def _apply_llm_harmonics(
    responses: list[AgentResponse],
    certainty_map: dict[str, float] | None = None,
) -> None:
    """Apply LLM-proposed updated_harmonics as Pass 2 emotional correction.

    Pass 1 (mechanical): text → embed → 9d projection → EMA update.
        Captures the content's emotional signature (speaker intent).
    Pass 2 (this): LLM evaluates the agent's contextual reaction.
        The proposed base_vector is blended into the harmonic buffer,
        weighted by llm_harmonics_blend. This corrects for context the
        mechanical pipeline can't see (identity, history, stakes).

    Per-agent certainty modulation: the blend weight is scaled by the
    agent's certainty factor (from LLM token entropy). Uncertain model
    → lower effective blend → defer to mechanical pipeline. The model's
    own uncertainty about an NPC's situation prevents confabulated
    emotional corrections from poisoning the buffer.

    The blend ensures the mechanical pipeline provides the baseline
    (honest but dumb) and the LLM provides the contextual correction
    (smart but potentially confabulated). Neither alone is sufficient.
    """
    from mindcore.harmonic_buffer import _config as hb_config

    base_blend = hb_config.llm_harmonics_blend
    if base_blend <= 0.0:
        return  # LLM harmonics disabled

    applied = 0
    for resp in responses:
        if resp.updated_harmonics is None:
            continue

        proposed = resp.updated_harmonics.base_vector
        if len(proposed) != 9:
            continue

        # Per-agent blend: scale by certainty factor.
        # Uncertain model → lower blend → trust mechanical pipeline more.
        certainty = 1.0
        if certainty_map is not None:
            certainty = certainty_map.get(resp.agent_id, 1.0)
        blend = base_blend * certainty

        # Blend: weighted average of current state and LLM proposal.
        # new = (1 - blend) * current + blend * proposed
        # Then update the buffer with this blended target.
        current = _harmonic_state.get_semagram(resp.agent_id)
        blended = [
            (1.0 - blend) * c + blend * p
            for c, p in zip(current, proposed)
        ]

        # Apply as a regular buffer update — the EMA smoothing provides
        # additional damping, so a single LLM proposal can't jerk the
        # buffer violently even at blend=1.0.
        _harmonic_state.update(resp.agent_id, blended)
        applied += 1

    if applied:
        logger.info(
            "Pass 2 emotional correction: %d agents updated via LLM harmonics (base_blend=%.2f)",
            applied, base_blend,
        )


# ---------------------------------------------------------------------------
# Recognition bootstrap — presence-change retrieval
# ---------------------------------------------------------------------------

async def _fire_recognition_retrieval(
    entered_npc_ids: list[str],
    all_active_npc_ids: list[str],
) -> None:
    """Fire referent-filtered retrieval when NPCs enter the scene.

    For each existing agent who was already present, retrieve memories
    involving each newcomer. Results go into _reminding_queue (one-tick
    delay) — the recognition surfaces on the next prompt, private to
    each agent. The newcomer's face triggers the recall; the agent's
    face may change (curvature spike) but the group context sees nothing.

    Lightweight: uses only the emotional axis (the agent's current state)
    with a referent filter. No semantic query needed — the face IS the
    query. Retrieval limit is small (3 memories per newcomer per agent).
    """
    global _reminding_queue

    if not shared_embedding.is_loaded() or not shared_emotional.is_loaded():
        return

    # Existing agents = everyone active EXCEPT the newcomers
    entered_set = set(entered_npc_ids)
    existing_agents = [npc for npc in all_active_npc_ids if npc not in entered_set]

    if not existing_agents:
        return

    recognition_count = 0
    stranger_count = 0
    for agent_id in existing_agents:
        emo_query = _harmonic_state.get_semagram(agent_id)
        # Dummy semantic query — recognition is emotion/referent driven.
        # Use the agent's own emotional state projected to 384d as a
        # semantic stand-in. In practice, the referent filter does the
        # heavy lifting; the semantic axis is secondary here.
        semantic_query = [0.0] * 384  # Neutral — referent filter dominates

        for newcomer_id in entered_npc_ids:
            try:
                bundle = await _memory_retriever.retrieve_for_agent(
                    agent_id=agent_id,
                    semantic_query=semantic_query,
                    emotional_query=emo_query,
                    lambda_t=0.8,  # Emotion-first — "this feels like that time..."
                    current_game_ts=time.time(),
                    referents=[newcomer_id],
                    broad_limit=10,
                    final_limit=3,  # Lightweight — just top recognition hits
                )
                recognized = bool(bundle.recent or bundle.summaries)
                if recognized:
                    # Merge into existing reminding queue entry for this agent
                    existing = _reminding_queue.get(agent_id)
                    if existing is None:
                        _reminding_queue[agent_id] = bundle
                    else:
                        existing.recent.extend(bundle.recent)
                        existing.summaries.extend(bundle.summaries)
                        existing.expandable_refs.extend(bundle.expandable_refs)
                    recognition_count += 1

                # Stranger detection (Phase 6b): no personal memory recalled
                # AND no FactPool belief/reputation-lore about the newcomer ->
                # this agent regards the newcomer as a stranger. Server-side
                # signal only (feeds 6c valence + 6d goal); never prompt-injected.
                if acquaintance.is_stranger(
                    _fact_pool, agent_id, newcomer_id, recognition_empty=not recognized,
                ):
                    if not acquaintance.is_known_stranger(agent_id, newcomer_id):
                        stranger_count += 1
                    acquaintance.record_stranger(agent_id, newcomer_id)
                else:
                    acquaintance.clear_stranger(agent_id, newcomer_id)
            except Exception as exc:
                logger.debug(
                    "Recognition retrieval failed for %s re: %s: %s",
                    agent_id, newcomer_id, exc,
                )

    if recognition_count:
        logger.info(
            "Recognition bootstrap: %d agents recalled memories of %d newcomers",
            recognition_count, len(entered_npc_ids),
        )
    if stranger_count:
        logger.info(
            "Stranger detection: %d new stranger pair(s) across %d newcomers",
            stranger_count, len(entered_npc_ids),
        )


# ---------------------------------------------------------------------------
# Valence-conditioned approach (Phase 6c)
# ---------------------------------------------------------------------------

# Approach-nudge tunables — a lean, not a shove (cf. goal_priming gains).
VALENCE_NUDGE_GAIN = 0.3       # base magnitude, scaled by |valence|·confidence·support
VALENCE_SUPPORT_CAP = 4        # supporting memories at which the support factor saturates
STRANGER_CURIOSITY = 0.1       # mild novelty curiosity toward a neutral unknown person
_VALENCE_EPS = 1e-6


def _wariness_direction() -> list[float]:
    """Guarded affect direction: raise fear, lower safety.

    The fear component is damped per-agent by the Confidence modulator, so a
    Brave/Foolhardy NPC barely registers it while a Cowardly one feels it
    sharply — personality shaping for free, no keyword controller.
    """
    d = [0.0] * EMOTIONAL_DIM
    d[0] = 1.0     # fear up
    d[7] = -1.0    # safety down
    return d


async def _condition_approach_by_valence(
    entered_npc_ids: list[str],
    all_active_npc_ids: list[str],
) -> None:
    """Condition each existing agent's approach disposition toward each
    newcomer by valence (Phase 6c).

    One percept-cued retrieval per (observer, newcomer): the perceived person
    (sharpened by occupation/tags) on the semantic axis, the observer's affect
    on the emotional axis. Specific memories of the newcomer override the class
    prior as a blend with hysteresis. Warmth promotes the get-acquainted
    resonance (a curiosity nudge); wariness suppresses it (a guarded nudge); a
    neutral disposition toward a known stranger yields mild novelty curiosity.
    Expressed ONLY as an affect nudge plus class-congruent recall (queued
    one-tick-delayed, anti-recursion) — never an imperative; the
    Assistance/Confidence/Aggression dials emerge from the felt state.
    """
    global _reminding_queue

    if not shared_embedding.is_loaded() or not shared_emotional.is_loaded():
        return

    entered_set = set(entered_npc_ids)
    existing_agents = [npc for npc in all_active_npc_ids if npc not in entered_set]
    if not existing_agents:
        return

    warm_count = 0
    wary_count = 0
    for observer in existing_agents:
        emo_query = _harmonic_state.get_deviation(observer)
        obs_kernel = identity_kernel.get(observer)
        traits = None
        tone = ""
        if obs_kernel is not None:
            raw_traits = obs_kernel.public.get("corePersonalityTraits")
            traits = raw_traits if isinstance(raw_traits, list) else None
            tone = obs_kernel.tone()
        approach_gain = valence.approachability(traits, tone)

        for subject in entered_npc_ids:
            subj_kernel = identity_kernel.get(subject)
            class_signal = subj_kernel.class_signal() if subj_kernel is not None else ""
            percept = valence.build_percept_text(subject, class_signal)
            if not percept:
                continue
            try:
                semantic_query = shared_embedding.embed_one(percept).tolist()
                reading = await valence.percept_cued_valence(
                    observer, subject, semantic_query, emo_query,
                )
            except Exception as exc:
                logger.debug(
                    "Valence conditioning failed for %s re: %s: %s",
                    observer, subject, exc,
                )
                continue

            blended = valence.blend_valence(reading)
            # 6d: cache the effective valence + prior-vs-individual gap so the
            # get-acquainted goal can gate its activation and feed dissonance.
            gap = (
                abs(reading.general - reading.individual)
                if (reading.general_support and reading.individual_support)
                else 0.0
            )
            valence.record_social(observer, subject, blended.effective, gap)
            support = reading.general_support + reading.individual_support
            support_factor = (
                min(support, VALENCE_SUPPORT_CAP) / VALENCE_SUPPORT_CAP
                if support else 0.0
            )
            evidence_mag = (
                VALENCE_NUDGE_GAIN * abs(blended.effective)
                * reading.confidence * support_factor
            )

            if blended.effective > _VALENCE_EPS and evidence_mag > 0.0:
                gain = evidence_mag * approach_gain
                nudge = [v * gain for v in goal_priming.curiosity_direction()]
                warm_count += 1
            elif blended.effective < -_VALENCE_EPS and evidence_mag > 0.0:
                gain = evidence_mag / max(approach_gain, 0.1)
                nudge = [v * gain for v in _wariness_direction()]
                wary_count += 1
            elif acquaintance.is_known_stranger(observer, subject):
                # Neutral disposition toward an unknown person: mild novelty
                # curiosity. The get-acquainted goal (6d) builds on this.
                gain = VALENCE_NUDGE_GAIN * STRANGER_CURIOSITY * approach_gain
                nudge = [v * gain for v in goal_priming.curiosity_direction()]
                warm_count += 1
            else:
                continue

            _harmonic_state.apply_nudge(observer, nudge)

            # Surface class-congruent recall as ordinary recalled content for
            # the NEXT tick (one-tick delay, like recognition).
            if reading.recall:
                bundle = _reminding_queue.get(observer)
                if bundle is None:
                    bundle = MemoryBundle(agent_id=observer)
                    _reminding_queue[observer] = bundle
                for text in reading.recall:
                    bundle.summaries.append({"text": text, "tier": "SOCIAL"})

    if warm_count or wary_count:
        logger.info(
            "Valence approach conditioning: %d warm, %d wary nudge(s) across %d newcomers",
            warm_count, wary_count, len(entered_npc_ids),
        )


# ---------------------------------------------------------------------------
# Disclosure -> hearsay propagation (Phase 6e)
# ---------------------------------------------------------------------------

async def _propagate_disclosures(
    responses: list[AgentResponse],
    all_active_npc_ids: list[str],
) -> None:
    """Reciprocal disclosure -> hearsay propagation (Phase 6e).

    For each NPC that spoke this turn, treat its utterance to a not-yet-met
    co-present NPC as an introduction: write the listener a hearsay memory and
    the speaker a reciprocal telling memory (so both remember the exchange),
    record a symbolic identity fact, and clear the stranger flag both ways.
    Gated on the stranger ledger so it fires once per pair and is idempotent if
    it re-fires. Server-side only; nothing is injected into the prompt.
    """
    if not shared_embedding.is_loaded() or not shared_emotional.is_loaded():
        return
    spoke = {r.agent_id for r in responses if r.utterance}
    if not spoke:
        return
    others = [npc for npc in all_active_npc_ids if npc not in ("Player", "")]

    count = 0
    for speaker in spoke:
        kernel = identity_kernel.get(speaker)
        if kernel is None:
            continue
        listeners = [
            listener for listener in others
            if listener != speaker
            and (
                acquaintance.is_known_stranger(speaker, listener)
                or acquaintance.is_known_stranger(listener, speaker)
            )
        ]
        if not listeners:
            continue
        descriptor = disclosure.identity_descriptor(kernel)
        try:
            id_vec = shared_embedding.embed_one(
                disclosure.fact_content(speaker, descriptor)
            ).tolist()
        except Exception:
            logger.debug("Disclosure embedding failed for %s", speaker, exc_info=True)
            continue
        speaker_reaction = _harmonic_state.get_deviation(speaker)
        for listener in listeners:
            try:
                if await disclosure.propagate_introduction(
                    writer=_memory_writer,
                    fact_pool=_fact_pool,
                    speaker=speaker,
                    listener=listener,
                    speaker_kernel=kernel,
                    identity_semantic_vec=id_vec,
                    listener_reaction=_harmonic_state.get_deviation(listener),
                    speaker_reaction=speaker_reaction,
                    game_ts=time.time(),
                ):
                    count += 1
            except Exception as exc:
                logger.debug(
                    "Disclosure propagation failed %s -> %s: %s", speaker, listener, exc,
                )

    if count:
        logger.info("Disclosure propagation: %d introduction(s) propagated", count)


# ---------------------------------------------------------------------------
# Sleep-time memory reconsolidation (goodnight trigger)
# ---------------------------------------------------------------------------

# Session event types that trigger an offline reconsolidation pass. `goodnight`
# is the canonical sleep boundary; `waitstart` could be added later for naps.
SLEEP_SESSION_TYPES: frozenset[str] = frozenset({"goodnight"})

RECON_REINTERPRET_PROMPT = (
    "You re-interpret an NPC's old memory through their matured understanding. "
    "Rewrite it as ONE concise sentence capturing what the memory now means to "
    "them and how they would understand or handle that kind of situation today "
    "— the latest version of the lesson, useful when facing a similar problem. "
    "Preserve the facts; invent nothing. Output only the sentence."
)


def _has_sleep_trigger(package: TickPackage) -> bool:
    """True if this tick carries a sleep/goodnight session event."""
    return any(e.event_type in SLEEP_SESSION_TYPES for e in package.events)


async def _reconsolidation_reinterpreter(content: str) -> str:
    """LLM reframing for the SWS phase.

    Returns '' on failure so the orchestrator falls back to its heuristic gist.
    Reuses the shared llm_client; the prompt asks for the matured, problem-
    solving understanding of the memory (the latest version of the lesson).
    """
    try:
        result = await llm_client.generate([
            {"role": "system", "content": RECON_REINTERPRET_PROMPT},
            {"role": "user", "content": content},
        ])
        return (result.content or "").strip()
    except LLMError as exc:
        logger.debug("Reconsolidation reinterpret LLM error (using fallback): %s", exc)
        return ""


async def _run_sleep_reconsolidation() -> None:
    """Run the goodnight reconsolidation pass for every agent with live state.

    Reads the live slow buffers as the matured baseline and writes RECON points
    (the source RAW stays immutable). Independent of the turn pipeline and of
    operational age-salience compaction. Non-fatal.
    """
    agent_ids = _harmonic_state.agent_ids
    if not agent_ids:
        logger.info("Goodnight: no agents with emotional state — reconsolidation skipped")
        return
    try:
        report = await memory_reconsolidation.run_reconsolidation(
            agent_ids,
            _harmonic_state.get_slow,
            writer=_memory_writer,
            reinterpret_fn=_reconsolidation_reinterpreter,
        )
        logger.info(
            "Goodnight reconsolidation: agents=%d reconsolidated=%d resolved=%d "
            "stalled=%d blocked=%d",
            report.scanned_agents, report.reconsolidated, report.resolved,
            report.stalled, report.skipped_blocked,
        )
    except Exception:
        logger.exception("Goodnight reconsolidation pass failed (non-fatal)")


# ---------------------------------------------------------------------------
# Dynamic modulator application on NPC registration
# ---------------------------------------------------------------------------

def _apply_modulators_for_new_npcs(turn_context: TurnContext) -> None:
    """Apply dynamic modulators for NPCs seen for the first time.

    Checks each active NPC's agent buffer for addnpc events with parsed_data.
    If the NPC doesn't yet have modulators on its harmonic buffer, constructs
    modulators from the parsed registration data and applies them.

    Currently: the addnpc wire event doesn't carry the 5 behavioral actor
    values (Aggression, Confidence, Morality, Mood, Assistance) — this is a
    known wire protocol gap. We apply default modulators (all-zero = uniform
    dynamics) so the infrastructure is exercised. When the Papyrus extension
    sends the values, they'll be extracted from parsed_data here.
    """
    for agent_id in turn_context.active_npc_ids:
        buf = _harmonic_state._buffers.get(agent_id)
        if buf is not None and buf._modulators is not None:
            continue  # Already has modulators

        # Look for addnpc parsed_data to extract actor values.
        # Future: parsed_data will contain aggression, confidence, etc.
        agent_buf = turn_context.agent_buffers.get(agent_id)
        if agent_buf is None:
            continue

        for event in agent_buf.events:
            if event.event_type == "addnpc" and event.parsed_data is not None:
                pd = event.parsed_data
                # Extract engine preset values when available in the wire data.
                # For now, use defaults — the modulator infrastructure is
                # exercised with uniform dynamics until the wire gap closes.
                mods = build_modulators(
                    aggression=int(pd.get("aggression", 0)),
                    confidence=int(pd.get("confidence", 2)),
                    morality=int(pd.get("morality", 3)),
                    mood=int(pd.get("mood", 0)),
                    assistance=int(pd.get("assistance", 0)),
                )
                _harmonic_state.apply_modulators(agent_id, mods)
                logger.info(
                    "Applied modulators for %s: agg=%.2f conf=%.2f mood=%s",
                    agent_id, mods.aggression_gain, mods.confidence_damp,
                    mods.mood_axis,
                )
                break  # One addnpc per agent is enough


async def _bootstrap_identities_for_new_npcs(turn_context: TurnContext) -> None:
    """Load seed identity kernels for NPCs seen for the first time (Phase 6a).

    Pure ID fetch from skyrim_npc_profiles — no embedding required. Cached per
    agent for the process lifetime. Degrades gracefully: NPCs absent from the
    seed (modded/custom) get an empty sentinel kernel so we do not re-query
    them every tick, and simply contribute no identity clause. A transient
    Qdrant error is left uncached so the next tick can retry.
    """
    loaded = 0
    for agent_id in turn_context.active_npc_ids:
        if identity_kernel.has(agent_id):
            continue
        slug = identity_kernel.agent_id_to_slug(agent_id)
        if not slug:
            identity_kernel.put(identity_kernel.IdentityKernel(agent_id=agent_id, slug=""))
            continue
        try:
            payload = await progeny_qdrant.read_profile(slug)
        except Exception:
            logger.debug("Identity bootstrap failed for %s", agent_id, exc_info=True)
            continue
        if payload is None:
            # Negative cache: the seed has no such NPC — don't re-query each tick.
            identity_kernel.put(identity_kernel.IdentityKernel(agent_id=agent_id, slug=slug))
            continue
        identity_kernel.put(identity_kernel.parse_kernel(agent_id, payload))
        loaded += 1
    if loaded:
        logger.info("Identity bootstrap: loaded %d identity kernel(s)", loaded)


# ---------------------------------------------------------------------------
# Per-group pipeline: prompt → LLM → expand
# ---------------------------------------------------------------------------

async def _run_group(
    group: DispatchGroup,
    turn_context: TurnContext,
    all_active_npc_ids: list[str],
    emotional_deltas: dict | None = None,
    memory_bundles: dict[str, MemoryBundle] | None = None,
    stable_prefix: str | None = None,
    n_keep: int = 0,
) -> tuple[list[AgentResponse], GenerateResult | None]:
    """
    Execute the full pipeline for one dispatch group.

    stable_prefix: pre-serialized Layer 1a JSON (built once, shared across groups).
    n_keep: tokens to pin in KV cache (system + Layer 1a).
    Returns (agent_responses, generate_result). On LLM error returns empty + None.
    """
    identities: dict[str, Any] = {}
    for scheduled in group.agents:
        kernel = identity_kernel.get(scheduled.agent_id)
        if kernel is not None:
            clause = kernel.self_clause()
            if clause:
                identities[scheduled.agent_id] = clause

    messages = prompt_formatter.build_prompt(
        turn_context, group.agents, all_active_npc_ids,
        harmonic_state=_harmonic_state,
        emotional_deltas=emotional_deltas,
        fact_pool=_fact_pool,
        memory_bundles=memory_bundles,
        speech_earshot=_accumulator._speech_earshot,
        identities=identities or None,
        stable_prefix=stable_prefix,
    )

    try:
        result = await llm_client.generate(messages, n_keep=n_keep)
    except LLMError as exc:
        logger.error("Group %s: LLM error — %s", group.label, exc)
        # Graceful degradation: empty responses for this group
        empty = [AgentResponse(agent_id=aid) for aid in group.agent_ids]
        return empty, None

    agent_responses = response_expander.expand_response(
        result.content, group.agent_ids,
    )
    return agent_responses, result


def _aggregate_timings(results: list[GenerateResult | None]) -> LLMTimings:
    """
    Aggregate timing data across parallel dispatch groups.

    Tokens are summed (total work). Wall times use max (parallel overlap).
    """
    total_prompt_tok = 0
    total_prompt_ms = 0.0
    total_gen_tok = 0
    total_gen_ms = 0.0
    total_cache_tok = 0

    for r in results:
        if r is None:
            continue
        total_prompt_tok += r.prompt_tokens
        total_prompt_ms = max(total_prompt_ms, r.prompt_ms)
        total_gen_tok += r.generated_tokens
        total_gen_ms = max(total_gen_ms, r.generation_ms)
        total_cache_tok += r.cache_tokens

    prompt_tps = (total_prompt_tok / (total_prompt_ms / 1000.0)) if total_prompt_ms > 0 else 0.0
    gen_tps = (total_gen_tok / (total_gen_ms / 1000.0)) if total_gen_ms > 0 else 0.0

    return LLMTimings(
        prompt_tokens=total_prompt_tok,
        prompt_ms=total_prompt_ms,
        prompt_tokens_per_sec=round(prompt_tps, 1),
        generated_tokens=total_gen_tok,
        generation_ms=total_gen_ms,
        generation_tokens_per_sec=round(gen_tps, 1),
        cache_tokens=total_cache_tok,
    )


@router.post("/ingest", response_model=Union[TurnResponse, AckResponse])
async def ingest(package: TickPackage) -> TurnResponse | AckResponse:
    """
    Ingest a TickPackage from Falcon.

    Pipeline: accumulate → detect turn → schedule → dispatch groups →
    parallel (prompt → LLM → expand) → merge → return.

    Serialized via _pipeline_lock: concurrent WebSocket ticks wait rather
    than racing through mutable harmonic/accumulator/scheduler state.
    Parallel dispatch groups within a single tick are unaffected.
    """
    async with _pipeline_lock:
        return await _ingest_inner(package)


async def _ingest_inner(package: TickPackage) -> TurnResponse | AckResponse:
    """Inner pipeline — runs under _pipeline_lock."""
    tick_id = package.tick_id

    # Step 1: Accumulate events, detect turn boundary
    turn_context = _accumulator.ingest(package)

    # Sleep-time memory reconsolidation: a goodnight session event triggers an
    # offline pass (REM probe -> SWS reinterpret) over each known agent's
    # memories. Independent of the turn pipeline; returns an Ack (no turn).
    if _has_sleep_trigger(package):
        await _run_sleep_reconsolidation()
        return AckResponse(tick_id=tick_id)

    if turn_context is None:
        logger.debug("Tick %s: data-only, accumulated %d events", tick_id, len(package.events))
        return AckResponse(tick_id=tick_id)

    # --- Turn trigger detected — full pipeline ---
    global _reminding_queue
    start_ms = time.monotonic()
    logger.info("Tick %s: turn trigger — player: %s", tick_id, turn_context.player_input[:80])

    # Consume last tick's remindings — these become this tick's memory context.
    # The one-tick delay is the anti-recursion guard: retrieval results from
    # tick N appear in the prompt on tick N+1, never on tick N itself.
    prior_remindings = _reminding_queue
    _reminding_queue = {}  # Clear for this tick's fresh retrieval
    if prior_remindings:
        logger.info(
            "Tick %s: injecting %d prior remindings into prompt",
            tick_id, len(prior_remindings),
        )

    # Record player input in dialogue history for all active agents.
    # Pass harmonic_state so each NPC's felt_at_receiving is captured.
    _accumulator.record_player_input(
        turn_context.player_input, harmonic_state=_harmonic_state)

    # Temporal decay: cool all agent buffers proportional to elapsed time.
    # Agents that haven't received events since last turn settle naturally.
    # Must happen before emotional processing so the EMA update operates
    # on decayed (settled) traces, not stale frozen state.
    _harmonic_state.cool_all()

    # Apply dynamic modulators for newly registered NPCs.
    # Modulators shape how emotional signals propagate through the buffer —
    # aggression gain, confidence damping, mood pull, etc.
    # Currently uses defaults since addnpc doesn't carry the 5 actor values
    # (wire protocol gap — see Living Doc §Engine Preset Values). When the
    # Papyrus extension sends values, extract them from parsed_data here.
    _apply_modulators_for_new_npcs(turn_context)

    # Identity kernels for first-seen NPCs (Phase 6a) — pure ID fetch, cached.
    await _bootstrap_identities_for_new_npcs(turn_context)

    # Emotional pipeline: inbound text → 9d projection → update harmonic buffers
    emotional_delta.process_inbound(turn_context, _harmonic_state)

    # Scene-level compression — when group composition changes significantly,
    # generate an SVO scene-break marker in the group timeline. This captures
    # "what happened here" before the scene shifts. Triggers before recognition
    # retrieval so the marker is available for the next tick's prompt.
    if _scene_compressor.should_compress(turn_context.presence_changes):
        _scene_compressor.compress_scene(
            _accumulator._group_memory,
            _accumulator.current_location,
            turn_context.presence_changes,
        )

    # Recognition bootstrap — presence-change retrieval trigger.
    # When an NPC enters the scene, existing agents fire referent-filtered
    # retrieval against the newcomer's ID. "Walk into a room, see an old
    # friend, the last few times you were together pop into your head."
    # Results go to _reminding_queue (one-tick delay) — private, Layer 2.
    if turn_context.presence_changes.entered:
        await _fire_recognition_retrieval(
            turn_context.presence_changes.entered,
            turn_context.active_npc_ids,
        )
        # Phase 6c: valence-condition each existing agent's approach toward the
        # newcomers (warmth promotes, wariness suppresses) via affect + recall.
        await _condition_approach_by_valence(
            turn_context.presence_changes.entered,
            turn_context.active_npc_ids,
        )

    # Persist inbound events to Qdrant via enrichment wrapper (RAW writes).
    # Second-thought ritual: the emotional vector stored with each memory
    # is the NPC's REACTION (deviation from baseline), not the text's raw
    # emotional projection. The same text gets different emotional keys for
    # different NPCs — the prisoner's memory of "great day for a hanging"
    # is keyed by dread, not the guard's jovial tone. Semantic vector (384d)
    # is still computed from the text content (what was said).
    # Non-blocking: failures are logged but don't stop the turn.
    try:
        qdrant_cli = progeny_qdrant.get_client()
        for agent_id, buf in turn_context.agent_buffers.items():
            # Capture this NPC's emotional reaction for the second-thought key.
            # get_deviation() returns fast - slow: what's unusual for this NPC
            # right now, after processing all inbound events this tick.
            reaction_vec = _harmonic_state.get_deviation(agent_id)
            for event in buf.events:
                if event.raw_data:
                    await qdrant_wrapper.ingest(
                        client=qdrant_cli,
                        text=event.raw_data,
                        collection=COLLECTION_NPC_MEMORIES,
                        agent_id=agent_id,
                        game_ts=event.game_ts,
                        event_type=event.event_type,
                        emotional_override=reaction_vec,
                    )
    except Exception as exc:
        logger.warning("RAW write pass failed (non-fatal): %s", exc)

    # Goal resonance priming — BEFORE scheduling so the curiosity nudge can
    # raise curvature and promote an agent's attention this same turn. Applies
    # the emotional nudge + standing pull; returns recall hints to surface.
    goal_results = await _prime_goals_for_turn(turn_context)

    # Step 2: Schedule agents — build NpcScheduleInfo with curvature data.
    # Position data comes from util_location_npc events (when available).
    # Collaboration status comes from agent buffers (active_task, follower).
    npc_info = _build_schedule_info(turn_context)
    roster = _scheduler.schedule(
        turn_context.active_npc_ids,
        npc_info=npc_info,
        player_position=None,  # TODO: extract from util_location_npc events
    )
    if not roster:
        logger.warning("Tick %s: no agents to schedule", tick_id)
        return TurnResponse(tick_id=tick_id, processing_time_ms=0, model_used="none")

    all_agent_ids = [a.agent_id for a in roster]
    logger.info("Tick %s: scheduled %d agents: %s", tick_id, len(roster), all_agent_ids)

    # Capture current emotional deltas for prompt building
    emotional_deltas = {
        a.agent_id: _harmonic_state.get_delta(a.agent_id)
        for a in roster
    }

    # Qdrant-backed memory retrieval — one-tick-delayed reminding protocol.
    #
    # Retrieval results go into _reminding_queue for the NEXT tick's prompt,
    # not this tick's. This tick's prompt uses prior_remindings (last tick's
    # retrieval). The one-tick delay prevents retrieval → prompt → retrieval
    # recursion: a memory in the prompt is in the current context and can
    # never be "re-remembered" because retrieval output is separated from
    # retrieval input by exactly one tick boundary.
    #
    # Semantic query: player's input embedding (what was said).
    # Emotional query (K/Q model): Q = fast - slow (deviation from
    # personality baseline). Retrieves memories whose emotional content
    # matches what's *unusual* for this NPC right now, not just what
    # they're feeling. A chronically fearful NPC doesn't retrieve fear-
    # memories every tick — only when fear exceeds their baseline.
    # When curvature is high (volatile), the delta (direction of change)
    # is blended in to capture shift-congruent memories.
    try:
        if shared_embedding.is_loaded() and shared_emotional.is_loaded():
            player_emb = shared_embedding.embed_one(turn_context.player_input)
            semantic_query = player_emb.tolist()
            for agent in roster:
                agent_id = agent.agent_id
                delta = emotional_deltas.get(agent_id)
                lambda_t = delta.lambda_t if delta else 0.5
                # Q = fast - slow: deviation from personality baseline.
                # What's unusual for this NPC right now.
                emo_query = _harmonic_state.get_deviation(agent_id)
                bundle = await _memory_retriever.retrieve_for_agent(
                    agent_id=agent_id,
                    semantic_query=semantic_query,
                    emotional_query=emo_query,
                    lambda_t=lambda_t,
                    current_game_ts=time.time(),
                    referents=turn_context.active_npc_ids,
                )
                # Store in reminding queue for NEXT tick (not this prompt)
                _reminding_queue[agent_id] = bundle
            logger.info(
                "Tick %s: retrieved %d bundles → reminding queue (next tick)",
                tick_id, len(_reminding_queue),
            )
    except Exception as exc:
        logger.warning("Memory retrieval failed (non-fatal): %s", exc)

    # This tick's prompt uses PRIOR remindings (last tick's retrieval),
    # not the fresh retrieval we just stored in _reminding_queue.
    memory_bundles: dict[str, MemoryBundle] = prior_remindings

    # Surface resonant goals as recalled content (NOT an imperative block).
    # The goal arrives like any other remembered fact; recognition and binding
    # emerge in the LLM. Goal priming is percept-driven, so it carries no
    # retrieval-recursion risk and can surface this same tick.
    for agent_id, gres in goal_results.items():
        if not gres.recall:
            continue
        bundle = memory_bundles.get(agent_id)
        if bundle is None:
            bundle = MemoryBundle(agent_id=agent_id)
            memory_bundles[agent_id] = bundle
        for statement in gres.recall:
            bundle.summaries.append({"text": statement, "tier": "GOAL"})

    # Merge recognition remindings into memory_bundles if any exist.
    # Recognition retrieval fired earlier in this tick goes into
    # _reminding_queue. These will surface on the NEXT tick like all
    # other remindings. No special handling needed — the queue is unified.

    # Step 3: Partition into dispatch groups
    groups = _scheduler.plan_dispatch(roster)
    logger.info(
        "Tick %s: dispatching %d groups: %s",
        tick_id, len(groups), [g.label for g in groups],
    )

    # Pre-serialize the stable Layer 1a prefix ONCE before the dispatch loop.
    # All groups share identical Layer 1a bytes so the KV cache from group 1's
    # LLM call covers it for groups 2–N. Compute n_keep to pin the prefix.
    urg = prompt_formatter._urgency(emotional_deltas)
    stable_prefix = prompt_formatter.build_shared_prefix(
        turn_context, turn_context.active_npc_ids, _fact_pool, urg)
    # n_keep = system message + stable prefix (4 chars/token heuristic)
    _n_keep = (len(prompt_formatter.SYSTEM_PROMPT) + len(stable_prefix)) // 4
    logger.debug("Tick %s: stable_prefix=%d chars, n_keep=%d",
                 tick_id, len(stable_prefix), _n_keep)

    # Step 4: Parallel execution — all groups run concurrently
    group_tasks = [
        _run_group(group, turn_context, turn_context.active_npc_ids,
                   emotional_deltas, memory_bundles,
                   stable_prefix=stable_prefix, n_keep=_n_keep)
        for group in groups
    ]
    group_results = await asyncio.gather(*group_tasks)

    # Step 5: Merge results in roster order + extract uncertainty
    all_responses: list[AgentResponse] = []
    all_gen_results: list[GenerateResult | None] = []
    all_certainty: dict[str, float] = {}
    for (responses, gen_result), group in zip(group_results, groups):
        all_responses.extend(responses)
        all_gen_results.append(gen_result)

        # Extract per-agent certainty from token logprobs.
        # The model's genuine uncertainty about each NPC feeds back into
        # the harmonic buffer (residual axis modulation) and scales the
        # LLM harmonics blend (uncertain → defer to mechanical pipeline).
        if gen_result is not None:
            group_certainty = compute_certainty(
                gen_result.token_logprobs, group.agent_ids,
            )
            all_certainty.update(group_certainty)
            for agent_id, cert in group_certainty.items():
                _harmonic_state.set_certainty(agent_id, cert)
            if any(c < 0.9 for c in group_certainty.values()):
                logger.info(
                    "Uncertainty feedback for group %s: %s",
                    group.label,
                    {k: round(v, 3) for k, v in group_certainty.items()},
                )

    # Record outputs into dialogue history (behavior adoption)
    # Also write utterances to Qdrant via enrichment wrapper and set utterance_key
    # for keys-over-wire: Falcon reads text by key instead of inline.
    try:
        qdrant_cli = progeny_qdrant.get_client()
    except RuntimeError:
        qdrant_cli = None

    for resp in all_responses:
        if resp.utterance:
            _accumulator.record_agent_output(
                resp.agent_id, resp.utterance, harmonic_state=_harmonic_state)
            # Write utterance to Qdrant, set utterance_key for Falcon
            if qdrant_cli is not None:
                try:
                    key = await qdrant_wrapper.ingest(
                        client=qdrant_cli,
                        text=resp.utterance,
                        collection=COLLECTION_NPC_MEMORIES,
                        agent_id=resp.agent_id,
                        game_ts=time.time(),  # wall time — no game_ts for LLM output
                        event_type="response",
                    )
                    if key:
                        resp.utterance_key = key
                except Exception as exc:
                    logger.warning("Utterance write failed for %s (non-fatal): %s", resp.agent_id, exc)

        _scheduler.record_action(resp.agent_id)
        # Extract SetCurrentTask actions → persist as active_task
        for action in resp.actions:
            if action.command == "SetCurrentTask":
                buf = _accumulator._agent_buffers.get(resp.agent_id)
                if buf:
                    buf.active_task = action.target or ""
                    logger.info(
                        "Agent %s task: %s",
                        resp.agent_id,
                        buf.active_task or "(cleared)",
                    )

    # Phase 6e: propagate introductions — an NPC speaking to someone it has not
    # met reciprocally de-strangers them (hearsay + telling memories + a
    # symbolic identity fact), closing the acquaintance loop for both parties.
    await _propagate_disclosures(all_responses, turn_context.active_npc_ids)

    # Emotional adoption: agent's own words shift its state (bidirectional pipeline)
    outbound_deltas = emotional_delta.process_outbound(all_responses, _harmonic_state)

    # Two-pass emotional evaluation — Pass 2: LLM contextual correction.
    # The mechanical pipeline (Pass 1) captured the text projection — how
    # the words sounded. The LLM's updated_harmonics captures the agent's
    # contextual reaction — how those words FELT given identity, history,
    # and current state. This closes the prisoner/guard gap.
    # The LLM-evaluated vector is blended into the buffer (not replacing it).
    _apply_llm_harmonics(all_responses, certainty_map=all_certainty)

    # Arc compression: check snap threshold for each agent after emotional adoption.
    # If snap exceeds threshold, generate a MOD-tier arc summary from recent RAW points.
    for agent_id, delta in outbound_deltas.items():
        if _arc_compressor.should_generate_arc(delta.snap):
            try:
                await _arc_compressor.generate_arc_summary(
                    agent_id=agent_id,
                    arc_start_ts=0.0,  # TODO: track arc start per agent
                    arc_end_ts=delta.semagram[0] if delta.semagram else 0.0,
                    semantic_vector=delta.semagram,  # 9d used as placeholder
                    emotional_vector=delta.semagram,
                    game_ts=time.time(),
                )
            except Exception as exc:
                logger.warning("Arc compression failed for %s (non-fatal): %s", agent_id, exc)

    # Slide memory windows — compress overflow after new entries recorded
    for agent_id in turn_context.active_npc_ids:
        buf = _accumulator._agent_buffers.get(agent_id)
        if buf:
            slide_window(buf.memory)
    # Slide the group timeline through the same compression pipeline
    slide_window(_accumulator._group_memory)

    elapsed_ms = int((time.monotonic() - start_ms) * 1000)
    timings = _aggregate_timings(all_gen_results)

    logger.info(
        "Tick %s: turn complete in %dms, %d responses across %d groups "
        "(prompt: %d tok/%.0fms, gen: %d tok/%.0fms, cached: %d tok)",
        tick_id, elapsed_ms, len(all_responses), len(groups),
        timings.prompt_tokens, timings.prompt_ms,
        timings.generated_tokens, timings.generation_ms,
        timings.cache_tokens,
    )

    model_used = "error" if all(r is None for r in all_gen_results) else "llama.cpp"

    return TurnResponse(
        tick_id=tick_id,
        responses=all_responses,
        processing_time_ms=elapsed_ms,
        model_used=model_used,
        llm_timings=timings,
    )



# ---------------------------------------------------------------------------
# WebSocket channel — persistent Falcon↔Progeny link
# ---------------------------------------------------------------------------

@router.websocket("/ws")
async def ws_channel(websocket: WebSocket) -> None:
    """Persistent bidirectional channel for Falcon tick delivery and response return.

    Falcon sends tick frames, Progeny processes through the same ingest()
    pipeline and sends response frames back when ready. No HTTP round-trip
    per tick — events and responses flow independently.

    Frame format (JSON):
        Falcon→Progeny:  {"type": "tick", "data": <TickPackage>}
        Progeny→Falcon:  {"type": "ack", "data": <AckResponse>}
                          {"type": "turn_response", "data": <TurnResponse>}
    """
    await websocket.accept()
    logger.info("Falcon WebSocket connected")

    async def _process_and_respond(pkg: TickPackage) -> None:
        """Process a tick in the background; send result when ready.

        Runs as a fire-and-forget task so the WebSocket receive loop
        stays responsive to pings during long LLM generation.
        If Falcon disconnects while a task is in flight, the send
        raises RuntimeError — silently dropped since the client is gone.
        """
        try:
            result = await ingest(pkg)
            if isinstance(result, TurnResponse):
                await websocket.send_text(json.dumps({
                    "type": "turn_response",
                    "data": result.model_dump(mode="json"),
                }))
                logger.info("WS: sent turn_response (%d agents)", len(result.responses))
            else:
                await websocket.send_text(json.dumps({
                    "type": "ack",
                    "data": result.model_dump(mode="json"),
                }))
        except RuntimeError as exc:
            # WebSocket closed while this task was in flight — Falcon
            # disconnected during LLM generation. Response is lost but
            # Falcon will reconnect and the next turn starts fresh.
            if "websocket.close" in str(exc) or "already completed" in str(exc):
                logger.debug("WS: send dropped (socket closed) for tick %s", pkg.tick_id)
            else:
                logger.exception("WS: error processing tick %s", pkg.tick_id)
        except Exception:
            logger.exception("WS: error processing tick %s", pkg.tick_id)

    try:
        while True:
            raw = await websocket.receive_text()
            frame = json.loads(raw)

            if frame.get("type") == "heartbeat":
                await websocket.send_text(json.dumps({"type": "heartbeat"}))
                continue

            if frame.get("type") != "tick":
                logger.warning("Unknown WS frame type: %s", frame.get("type"))
                continue

            package = TickPackage.model_validate(frame["data"])
            # Fire-and-forget: don't block the receive loop on LLM generation
            asyncio.create_task(_process_and_respond(package))
    except WebSocketDisconnect:
        logger.info("Falcon WebSocket disconnected")
    except Exception:
        logger.exception("WebSocket error")


@router.get("/health")
async def health() -> dict:
    """Liveness check + LLM connectivity."""
    llm_ok = await llm_client.health_check()
    return {
        "status": "ok",
        "llm_connected": llm_ok,
        "turn_counter": _scheduler.turn_counter,
    }
