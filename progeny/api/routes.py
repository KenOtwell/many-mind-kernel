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

from fastapi import APIRouter

from shared.schemas import (
    AckResponse,
    AgentResponse,
    LLMTimings,
    TickPackage,
    TurnResponse,
)
from progeny.src.event_accumulator import EventAccumulator, TurnContext
from progeny.src.agent_scheduler import AgentScheduler, DispatchGroup
from progeny.src import prompt_formatter
from progeny.src import llm_client
from progeny.src.llm_client import GenerateResult, LLMError
from progeny.src import response_expander
from progeny.src.memory_compressor import slide_window
from progeny.src import emotional_delta
from progeny.src.harmonic_buffer import HarmonicState

logger = logging.getLogger(__name__)

router = APIRouter()

# ---------------------------------------------------------------------------
# Pipeline state — lives for the process lifetime.
# Phase 2: these move to a proper application state container.
# ---------------------------------------------------------------------------
_accumulator = EventAccumulator()
_scheduler = AgentScheduler()
_harmonic_state = HarmonicState()


# ---------------------------------------------------------------------------
# Per-group pipeline: prompt → LLM → expand
# ---------------------------------------------------------------------------

async def _run_group(
    group: DispatchGroup,
    turn_context: TurnContext,
    all_active_npc_ids: list[str],
    emotional_deltas: dict | None = None,
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
    """
    tick_id = package.tick_id

    # Step 1: Accumulate events, detect turn boundary
    turn_context = _accumulator.ingest(package)

    if turn_context is None:
        logger.debug("Tick %s: data-only, accumulated %d events", tick_id, len(package.events))
        return AckResponse(tick_id=tick_id)

    # --- Turn trigger detected — full pipeline ---
    start_ms = time.monotonic()
    logger.info("Tick %s: turn trigger — player: %s", tick_id, turn_context.player_input[:80])

    # Record player input in dialogue history for all active agents
    _accumulator.record_player_input(turn_context.player_input)

    # Emotional pipeline: inbound text → 9d projection → update harmonic buffers
    emotional_delta.process_inbound(turn_context, _harmonic_state)

    # Step 2: Schedule agents
    roster = _scheduler.schedule(turn_context.active_npc_ids)
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

    # Step 3: Partition into dispatch groups
    groups = _scheduler.plan_dispatch(roster)
    logger.info(
        "Tick %s: dispatching %d groups: %s",
        tick_id, len(groups), [g.label for g in groups],
    )

    # Step 4: Parallel execution — all groups run concurrently
    group_tasks = [
        _run_group(group, turn_context, turn_context.active_npc_ids, emotional_deltas)
        for group in groups
    ]
    group_results = await asyncio.gather(*group_tasks)

    # Step 5: Merge results in roster order
    all_responses: list[AgentResponse] = []
    all_gen_results: list[GenerateResult | None] = []
    for responses, gen_result in group_results:
        all_responses.extend(responses)
        all_gen_results.append(gen_result)

    # Record outputs into dialogue history (behavior adoption)
    for resp in all_responses:
        if resp.utterance:
            _accumulator.record_agent_output(resp.agent_id, resp.utterance)
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
    emotional_delta.process_outbound(all_responses, _harmonic_state)

    # Slide memory windows — compress overflow after new entries recorded
    for agent_id in turn_context.active_npc_ids:
        buf = _accumulator._agent_buffers.get(agent_id)
        if buf:
            slide_window(buf.memory)

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



@router.get("/health")
async def health() -> dict:
    """Liveness check + LLM connectivity."""
    llm_ok = await llm_client.health_check()
    return {
        "status": "ok",
        "llm_connected": llm_ok,
        "turn_counter": _scheduler.turn_counter,
    }
