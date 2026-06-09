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
from typing import Union
from uuid import uuid4

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from shared.schemas import (
    AckResponse,
    AgentResponse,
    LLMTimings,
    TickPackage,
    TurnResponse,
)
from shared.constants import COLLECTION_NPC_MEMORIES
from shared import qdrant_wrapper
from progeny.src.event_accumulator import EventAccumulator, TurnContext
from progeny.src.agent_scheduler import AgentScheduler, DispatchGroup, NpcScheduleInfo
from progeny.src.fact_pool import FactPool
from progeny.src import prompt_formatter
from progeny.src import llm_client
from progeny.src.llm_client import GenerateResult, LLMError
from progeny.src import response_expander
from progeny.src.memory_compressor import slide_window
from progeny.src import emotional_delta
from progeny.src.harmonic_buffer import HarmonicState, build_modulators
from progeny.src.memory_writer import MemoryWriter
from progeny.src.memory_retrieval import MemoryRetriever, MemoryBundle
from progeny.src.compression import ArcCompressor, SceneCompressor, DEFAULT_SNAP_THRESHOLD
from progeny.src import qdrant_client as progeny_qdrant
from progeny.src.uncertainty import compute_certainty
from shared import embedding as shared_embedding
from shared import emotional as shared_emotional

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
    from progeny.src.harmonic_buffer import _config as hb_config

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
                if bundle.recent or bundle.summaries:
                    # Merge into existing reminding queue entry for this agent
                    existing = _reminding_queue.get(agent_id)
                    if existing is None:
                        _reminding_queue[agent_id] = bundle
                    else:
                        existing.recent.extend(bundle.recent)
                        existing.summaries.extend(bundle.summaries)
                        existing.expandable_refs.extend(bundle.expandable_refs)
                    recognition_count += 1
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


# ---------------------------------------------------------------------------
# Per-group pipeline: prompt → LLM → expand
# ---------------------------------------------------------------------------

async def _run_group(
    group: DispatchGroup,
    turn_context: TurnContext,
    all_active_npc_ids: list[str],
    emotional_deltas: dict | None = None,
    memory_bundles: dict[str, MemoryBundle] | None = None,
) -> tuple[list[AgentResponse], GenerateResult | None]:
    """
    Execute the full pipeline for one dispatch group.

    Returns (agent_responses, generate_result). On LLM error, returns
    empty responses and None result (graceful degradation per group).
    """
    messages = prompt_formatter.build_prompt(
        turn_context, group.agents, all_active_npc_ids,
        harmonic_state=_harmonic_state,
        emotional_deltas=emotional_deltas,
        fact_pool=_fact_pool,
        memory_bundles=memory_bundles,
        speech_earshot=_accumulator._speech_earshot,
    )

    try:
        result = await llm_client.generate(messages)
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

    # Record player input in dialogue history for all active agents
    _accumulator.record_player_input(turn_context.player_input)

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

    # Step 4: Parallel execution — all groups run concurrently
    group_tasks = [
        _run_group(group, turn_context, turn_context.active_npc_ids,
                   emotional_deltas, memory_bundles)
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
            _accumulator.record_agent_output(resp.agent_id, resp.utterance)
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
