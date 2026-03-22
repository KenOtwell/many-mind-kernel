"""
Prompt formatter for Progeny — The Ritual.

Builds a chat-completion messages[] array per dispatch group. In parallel
mode, each group gets its own prompt with the same shared prefix (system
prompt + world state + who's present) for KV cache reuse, but only its
assigned agent blocks.

Message 1 (system): Static instruction block — the reality contract.
Message 2 (user):   Data payload + instruction, rebuilt fresh every turn.

Zero context rot: nothing stale survives from the previous prompt.
Continuity comes from harmonic buffers and Qdrant retrieval, not from
the prompt carrying forward.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from shared.constants import ZERO_SEMAGRAM
from progeny.src.agent_scheduler import ScheduledAgent
from progeny.src.event_accumulator import AgentBuffer, TieredMemory, TurnContext
from progeny.src.fact_pool import FactPool

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from progeny.src.harmonic_buffer import EmotionalDelta, HarmonicState

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# System prompt — stable across turns, benefits from KV cache reuse
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the Many-Mind Kernel — the slow-twitch cognitive layer for multiple NPCs \
in the world of Skyrim. You govern their thoughts, speech, and behavioral \
dispositions simultaneously. The game engine handles fast-twitch reflexes \
(combat, pathfinding, physics). You handle contemplation, strategy, and emotion.

You do not control NPC motor actions directly. You set DISPOSITIONS via actor \
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

RESPONSE FORMAT: Return a JSON object with a "responses" array. One entry per \
agent listed in the prompt, in the same order. Scale detail to the agent's tier:
  Tier 0: utterance + actor_value_deltas + actions + updated_harmonics + new_memories
  Tier 1: utterance + actor_value_deltas + actions
  Tier 2: actor_value_deltas + brief utterance if warranted
  Tier 3: actor_value_deltas only (nudge dials, confirm or adjust)

Each agent's ticks_since_last_action tells you how long since you last attended \
them. Calibrate accordingly — recently attended agents need small adjustments; \
long-unattended agents may need larger updates or may be fine continuing as-is.

Be the mind. The engine is the body."""


INSTRUCTION_PROMPT = (
    "For each agent listed, produce a response appropriate to their tier "
    "and current situation. Return only valid JSON matching the response format."
)


# Tier-scaled fact limits for per-agent known_world
TIER_FACT_LIMITS: dict[int, int] = {0: 20, 1: 10, 2: 5, 3: 2}


def build_prompt(
    turn_context: TurnContext,
    roster: list[ScheduledAgent],
    all_active_npc_ids: list[str] | None = None,
    harmonic_state: "HarmonicState | None" = None,
    emotional_deltas: "dict[str, EmotionalDelta | None] | None" = None,
    fact_pool: FactPool | None = None,
) -> list[dict[str, str]]:
    """
    Build the 2-message chat completion array for a dispatch group.

    Returns a list of message dicts ready for the LLM:
    [{"role": "system", "content": ...}, {"role": "user", "content": ...}]

    Data payload and instruction are combined into a single user message
    to satisfy chat templates that require strict user/assistant alternation
    (e.g. Mistral NeMo).

    Args:
        turn_context: Accumulated context for this turn.
        roster: Agents in THIS dispatch group (may be a subset of all active).
        all_active_npc_ids: Full list of active NPCs this turn (for cross-agent
            awareness in partitioned calls). If None, derived from roster.
        harmonic_state: Live emotional state container. If None, uses ZERO_SEMAGRAM.
        emotional_deltas: Pre-captured deltas per agent. If None, no delta in prompt.
    """
    # Message 2: data payload + instruction — rebuilt fresh every turn
    data_payload = _build_data_payload(
        turn_context, roster, all_active_npc_ids, harmonic_state, emotional_deltas,
        fact_pool,
    )
    user_content = json.dumps(data_payload, indent=None) + "\n\n" + INSTRUCTION_PROMPT

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _build_data_payload(
    ctx: TurnContext,
    roster: list[ScheduledAgent],
    all_active_npc_ids: list[str] | None = None,
    harmonic_state: "HarmonicState | None" = None,
    emotional_deltas: "dict[str, EmotionalDelta | None] | None" = None,
    fact_pool: FactPool | None = None,
) -> dict[str, Any]:
    """Assemble the JSON data payload for message 2."""
    agents = []
    for scheduled in roster:
        agent_block = _build_agent_block(
            scheduled, ctx, harmonic_state, emotional_deltas, fact_pool,
        )
        agents.append(agent_block)

    # Present NPCs — all active NPCs this turn, not just those in this group.
    roster_ids = [a.agent_id for a in roster]
    present_ids = all_active_npc_ids if all_active_npc_ids is not None else roster_ids

    # Lore context — shared facts known by everyone (category="lore")
    lore_context: list[str] = []
    if fact_pool is not None:
        lore_facts = fact_pool.query("Player", category="lore", limit=10)
        lore_context = [f.content for f in lore_facts]

    payload: dict[str, Any] = {
        "present_npcs": present_ids,
        "agents": agents,
        "player_input": {
            "type": "inputtext",
            "text": ctx.player_input,
        },
    }
    if lore_context:
        payload["lore_context"] = lore_context

    return payload


def _build_agent_block(
    scheduled: ScheduledAgent,
    ctx: TurnContext,
    harmonic_state: "HarmonicState | None" = None,
    emotional_deltas: "dict[str, EmotionalDelta | None] | None" = None,
    fact_pool: FactPool | None = None,
) -> dict[str, Any]:
    """
    Build a single agent block at tier-appropriate granularity.

    Phase 1: all agents get full blocks (Tier 0 format).
    Phase 2 will scale down for Tier 1-3.
    """
    agent_id = scheduled.agent_id
    buf = ctx.agent_buffers.get(agent_id)

    # Recent events for this agent (from this turn's tick accumulation)
    recent_events = []
    if buf:
        recent_events = [e.raw_data for e in buf.events[-10:]]

    # Tiered memory (cross-turn) — verbatim, compressed, keywords
    memory = buf.memory if buf else TieredMemory()

    # Live emotional state from harmonic buffers (or zero fallback)
    semagram = (
        harmonic_state.get_semagram(agent_id)
        if harmonic_state is not None
        else ZERO_SEMAGRAM
    )

    # Per-agent known world from fact pool (tier-scaled limits)
    known_world: list[dict] = []
    if fact_pool is not None:
        fact_limit = TIER_FACT_LIMITS.get(scheduled.tier, 2)
        known_world = fact_pool.facts_for_prompt(agent_id, limit=fact_limit)

    block: dict[str, Any] = {
        "agent_id": agent_id,
        "tier": scheduled.tier,
        "ticks_since_last_action": scheduled.ticks_since_last_action,
        "base_vector": semagram,
        "known_world": known_world,
        "recent_events": recent_events,
        "dialogue_history": memory.verbatim,
        "compressed_history": memory.compressed,
        "distant_memories": memory.keywords,
    }

    # Active task — persistent goal the agent is working toward
    if buf and buf.active_task:
        block["active_task"] = buf.active_task

    # Emotional dynamics — curvature/snap/tension for LLM calibration
    if emotional_deltas is not None:
        delta = emotional_deltas.get(agent_id)
        if delta is not None:
            block["emotional_dynamics"] = {
                "curvature": round(delta.curvature, 4),
                "snap": round(delta.snap, 4),
                "tension": round(1.0 - delta.lambda_t, 4),
            }

    return block


def _get_location(ctx: TurnContext) -> str:
    """Extract current location from world events or default."""
    # Check for location events in this turn
    for event in reversed(ctx.world_events):
        if event.event_type == "location":
            return event.raw_data
    return "Unknown"
