"""
Prompt formatter for Progeny — The Ritual.

Three-layer prompt topology optimized for KV cache reuse:

  Layer 0 (system): Lore + rules + response format. Static across turns
      — benefits from full KV cache reuse between turns.
  Layer 1 (group context): Scene shared by all present NPCs. Location,
      shared events, shared facts (ATMS: all-bits-set), and the group
      emotional display (fast buffers of all present NPCs). Identical
      across dispatch groups within a tick — KV cache reuse within turn.
  Layer 2 (agent blocks): Private per-agent data. Full harmonic state,
      private memories/history, private knowledge (ATMS: only-my-bit),
      goals, emotional dynamics. Varies per dispatch group.

The group display — each NPC's fast buffer as their observable "face" —
gives every agent social awareness without explicit theory-of-mind.
The fast buffer IS the non-verbal channel: who looks tense, who just
flinched, who's calm. Medium/slow buffers stay private (internal).

Zero context rot: nothing stale survives from the previous prompt.
Continuity comes from harmonic buffers and Qdrant retrieval, not from
the prompt carrying forward.
"""
from __future__ import annotations

import json
import logging
from typing import Any

from shared.constants import EMOTIONAL_AXES, ZERO_SEMAGRAM
from progeny.src.agent_scheduler import ScheduledAgent
from progeny.src.event_accumulator import AgentBuffer, TieredMemory, TurnContext
from progeny.src.fact_pool import FactPool
from progeny.src.memory_retrieval import MemoryBundle

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mindcore.harmonic_buffer import EmotionalDelta, HarmonicState

logger = logging.getLogger(__name__)

# Compact axis labels for the group display — saves tokens vs full names
_DEMEANOR_AXES = [a[:3].upper() for a in EMOTIONAL_AXES]  # FEA,ANG,LOV,...


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

PROMPT STRUCTURE:
  group_context: shared scene — what everyone present can see and sense.
    group_display: each NPC's observable demeanor (their emotional "face").
    Axes: FEA=fear ANG=anger LOV=love DIS=disgust EXC=excitement SAD=sadness \
JOY=joy SAF=safety RES=residual. Tension = how volatile they appear.
  agents[]: private per-agent data — their inner state, memories, goals.
    Only YOU (as that agent's mind) see their full internal state.
    Other agents see only the group_display surface.

SPEAKING ROLES:
  Each agent has a speaking_role: "addressee", "bystander", or "group_addressed".
  Only the addressee's voice is heard in-game. Choose carefully.
  - addressee: The player is talking to you. Respond fully with utterance, \
actor_value_deltas, actions, and all other fields appropriate to your tier.
  - bystander: The player is talking to someone else. Do NOT produce an \
utterance — your voice won't be heard this turn. Instead, update your internal \
state: actor_value_deltas, updated_harmonics, actions (if you'd physically \
react), new_memories. You're still thinking and feeling — you just aren't \
speaking. On a future turn you may be addressed and can reference what you \
observed (visible in group_context.shared_recent).
  - group_addressed: The player spoke to the group without targeting anyone \
specifically. You may respond with an utterance, but differentiate your \
perspective — don't echo what another NPC would likely say. Contribute your \
unique viewpoint, knowledge, or reaction. If you have nothing distinct to add, \
brief acknowledgment or just actor_value_deltas is fine.

RESPONSE FORMAT: Return a JSON object with a "responses" array. One entry per \
agent listed, in order. Scale detail to tier:
  Tier 0: utterance + actor_value_deltas + actions + updated_harmonics + new_memories
  Tier 1: utterance + actor_value_deltas + actions
  Tier 2: actor_value_deltas + brief utterance if warranted
  Tier 3: actor_value_deltas only (nudge dials, confirm or adjust)

Each agent's ticks_since_last_action tells you how long since you last attended \
them. Calibrate accordingly.

Be the mind. The engine is the body."""


INSTRUCTION_PROMPT = (
    "For each agent listed, produce a response appropriate to their tier "
    "and current situation. Return only valid JSON matching the response format."
)


# ---------------------------------------------------------------------------
# Addressee detection — who is the player talking to?
# ---------------------------------------------------------------------------

def _resolve_addressee(
    player_input: str,
    roster: list[ScheduledAgent],
    speech_earshot: dict | None = None,
) -> str | None:
    """Determine which NPC the player is addressing.

    Resolution order:
      1. Name match in player input (fuzzy: case-insensitive substring)
      2. Last NPC who spoke to the player (from _speech earshot context)
      3. None → group-addressed (no specific target identified)

    Returns the agent_id of the addressee, or None if the player is
    addressing the group (triggers group_addressed role for all NPCs).
    """
    if not roster:
        return None

    roster_ids = [a.agent_id for a in roster]
    input_lower = player_input.lower()

    # 1. Name mentioned in player input
    for agent_id in roster_ids:
        # Match on first name or full name, case-insensitive
        name_lower = agent_id.lower()
        # Try full name and first token (handles "Afelir Morel" → "afelir")
        first_name = name_lower.split()[0] if " " in name_lower else name_lower
        if name_lower in input_lower or first_name in input_lower:
            return agent_id

    # 2. Last speaker to the player (from _speech earshot)
    if speech_earshot is not None:
        last_speaker = speech_earshot.get("speaker", "")
        if last_speaker and last_speaker in roster_ids:
            return last_speaker

    # 3. No specific target → group-addressed
    return None


# Tier-scaled fact limits for per-agent private knowledge
TIER_FACT_LIMITS: dict[int, int] = {0: 20, 1: 10, 2: 5, 3: 2}

# Curvature truncation thresholds — continuous gradient from calm to crisis.
# Below LOW: full prompt (deep memory, full history, lore).
# Above HIGH: maximum truncation (anchors only, strip history, focus tactical).
# Between: linear interpolation.
CURVATURE_TRUNCATION_LOW = 0.1    # Below this → full prompt
CURVATURE_TRUNCATION_HIGH = 0.5   # Above this → maximum truncation


def build_prompt(
    turn_context: TurnContext,
    roster: list[ScheduledAgent],
    all_active_npc_ids: list[str] | None = None,
    harmonic_state: "HarmonicState | None" = None,
    emotional_deltas: "dict[str, EmotionalDelta | None] | None" = None,
    fact_pool: FactPool | None = None,
    memory_bundles: dict[str, MemoryBundle] | None = None,
    speech_earshot: dict | None = None,
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
        fact_pool, memory_bundles, speech_earshot,
    )
    user_content = json.dumps(data_payload, indent=None) + "\n\n" + INSTRUCTION_PROMPT

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]


def _urgency(emotional_deltas: "dict[str, EmotionalDelta | None] | None") -> float:
    """Compute the scene urgency from max curvature across all agents.

    Returns a value in [0, 1]: 0 = calm, 1 = crisis.
    Used by both Layer 1 and Layer 2 to scale prompt depth.
    """
    if not emotional_deltas:
        return 0.0
    max_curv = max(
        (d.curvature for d in emotional_deltas.values() if d is not None),
        default=0.0,
    )
    # Map to [0, 1] via the truncation thresholds
    if max_curv <= CURVATURE_TRUNCATION_LOW:
        return 0.0
    if max_curv >= CURVATURE_TRUNCATION_HIGH:
        return 1.0
    return (max_curv - CURVATURE_TRUNCATION_LOW) / (
        CURVATURE_TRUNCATION_HIGH - CURVATURE_TRUNCATION_LOW
    )


def _build_data_payload(
    ctx: TurnContext,
    roster: list[ScheduledAgent],
    all_active_npc_ids: list[str] | None = None,
    harmonic_state: "HarmonicState | None" = None,
    emotional_deltas: "dict[str, EmotionalDelta | None] | None" = None,
    fact_pool: FactPool | None = None,
    memory_bundles: dict[str, MemoryBundle] | None = None,
    speech_earshot: dict | None = None,
) -> dict[str, Any]:
    """Assemble the JSON data payload for message 2.

    Three-layer structure with curvature-driven truncation:
      group_context (Layer 1): shared scene. Truncated under high urgency
          (strip history to anchors, drop lore, keep display + events).
      agents[] (Layer 2): private per-agent blocks. Truncated under high
          urgency (drop deep history, keep recent events + dynamics).
      player_input: the current player utterance/action.

    Truncation is about cognitive focus, not speed. High curvature means
    the agent should think about the RIGHT things (tactical situation),
    not about everything (social history, lore, old memories).
    """
    roster_ids = [a.agent_id for a in roster]
    present_ids = all_active_npc_ids if all_active_npc_ids is not None else roster_ids

    # Urgency: 0.0 = calm (full prompt), 1.0 = crisis (maximum truncation)
    urg = _urgency(emotional_deltas)

    # Addressee detection — who is the player talking to?
    addressee_id = _resolve_addressee(ctx.player_input, roster, speech_earshot)

    # --- Layer 1: Group context (shared by all present NPCs) ---
    group_context = _build_group_context(
        ctx, present_ids, harmonic_state, emotional_deltas, fact_pool, urg,
    )

    # --- Layer 2: Private agent blocks ---
    agents = []
    for scheduled in roster:
        bundle = (memory_bundles or {}).get(scheduled.agent_id)
        agent_block = _build_agent_block(
            scheduled, ctx, present_ids, harmonic_state, emotional_deltas,
            fact_pool, bundle, urg,
        )
        # Speaking role: addressee responds fully, bystanders stay quiet,
        # group_addressed means everyone can speak but should differentiate.
        if addressee_id is None:
            agent_block["speaking_role"] = "group_addressed"
        elif scheduled.agent_id == addressee_id:
            agent_block["speaking_role"] = "addressee"
        else:
            agent_block["speaking_role"] = "bystander"
        agents.append(agent_block)

    payload: dict[str, Any] = {
        "group_context": group_context,
        "agents": agents,
        "player_input": {
            "type": "inputtext",
            "text": ctx.player_input,
        },
    }

    return payload


# ---------------------------------------------------------------------------
# Layer 1: Group context — shared scene visible to all present NPCs
# ---------------------------------------------------------------------------

def _build_group_context(
    ctx: TurnContext,
    present_ids: list[str],
    harmonic_state: "HarmonicState | None" = None,
    emotional_deltas: "dict[str, EmotionalDelta | None] | None" = None,
    fact_pool: FactPool | None = None,
    urgency: float = 0.0,
) -> dict[str, Any]:
    """Build the shared group context (Layer 1) with curvature truncation.

    Urgency 0.0 (calm): full history, lore, shared knowledge, display.
    Urgency 1.0 (crisis): anchors only + display + tick events.
    The gradient is continuous — not a binary switch.

    Always included regardless of urgency: location, present_npcs,
    group_display (the room's emotional state), shared_events (what
    just happened this tick). These are the tactical essentials.
    """
    group: dict[str, Any] = {
        "location": _get_location(ctx),
        "present_npcs": present_ids,
    }

    # Group timeline — truncated by urgency.
    # Calm: full depth (verbatim + compressed + anchors).
    # Crisis: anchors only (SVO recognition triggers, minimal tokens).
    gm = ctx.group_memory
    if urgency < 0.5:
        # Calm to moderate: include verbatim and compressed
        if gm.verbatim:
            group["shared_recent"] = gm.verbatim[-10:]
        if gm.compressed:
            group["shared_history"] = gm.compressed[-10:]
    # Anchors always included (cheap, ~15 tokens each)
    if gm.keywords:
        group["shared_anchors"] = gm.keywords[-10:]

    # Shared tick events — always included (what just happened)
    shared_events = [e.raw_data for e in ctx.world_events[-10:] if e.raw_data]
    if shared_events:
        group["shared_events"] = shared_events

    # Shared facts and lore — dropped under high urgency.
    # Lore and background knowledge are irrelevant during a crisis;
    # the agent should focus on the immediate situation.
    if urgency < 0.7 and fact_pool is not None:
        all_present = ["Player"] + list(present_ids)
        shared_facts = fact_pool.query_shared(all_present, limit=15)
        if shared_facts:
            group["shared_knowledge"] = [f.content for f in shared_facts]

        if urgency < 0.3:
            # Only include lore when fully calm
            lore_facts = fact_pool.query("Player", category="lore", limit=10)
            if lore_facts:
                group["lore"] = [f.content for f in lore_facts]

    # Group emotional display — always included. This IS the tactical
    # information during a crisis: who looks scared, who's ready to fight.
    group_display = _build_group_display(
        present_ids, harmonic_state, emotional_deltas,
    )
    if group_display:
        group["group_display"] = group_display

    return group


def _build_group_display(
    present_ids: list[str],
    harmonic_state: "HarmonicState | None" = None,
    emotional_deltas: "dict[str, EmotionalDelta | None] | None" = None,
) -> list[dict[str, Any]]:
    """Build the group emotional display — each NPC's observable "face".

    The fast buffer is what you'd see on someone's expression and body
    language. Compact: name + 9d demeanor vector + tension scalar.
    """
    if harmonic_state is None:
        return []

    display: list[dict[str, Any]] = []
    for npc_id in present_ids:
        fast = harmonic_state.get_semagram(npc_id)
        # Skip NPCs with no emotional state yet (just registered)
        if fast == list(ZERO_SEMAGRAM):
            continue
        entry: dict[str, Any] = {
            "name": npc_id,
            "demeanor": [round(v, 3) for v in fast],
        }
        # Add tension (1 - coherence) if available — how volatile they appear
        if emotional_deltas is not None:
            delta = emotional_deltas.get(npc_id)
            if delta is not None:
                entry["tension"] = round(1.0 - delta.coherence, 3)
        display.append(entry)

    return display


# ---------------------------------------------------------------------------
# Layer 2: Private agent blocks
# ---------------------------------------------------------------------------

def _build_agent_block(
    scheduled: ScheduledAgent,
    ctx: TurnContext,
    present_ids: list[str],
    harmonic_state: "HarmonicState | None" = None,
    emotional_deltas: "dict[str, EmotionalDelta | None] | None" = None,
    fact_pool: FactPool | None = None,
    memory_bundle: MemoryBundle | None = None,
    urgency: float = 0.0,
) -> dict[str, Any]:
    """Build a single agent block (Layer 2) scaled by tier AND urgency.

    Tier scaling (Living Doc §Agent Priority Paging):
      Tier 0 (Full): all fields — identity, full buffers, full history,
          private knowledge, emotional dynamics, state_history, task.
      Tier 1 (Abbreviated): base_vector + curvature (no buffer traces),
          recent events (last 5), dialogue_history (last 3), task.
      Tier 2 (Minimal): base_vector only, recent events (last 2).
      Tier 3+ (Stub): base_vector only. Nothing else.

    Curvature-driven truncation (urgency gradient) applies ON TOP of tier
    scaling for Tiers 0-1. Under crisis, even Tier 0 drops deep memory.
    Tiers 2-3 are already sparse enough that urgency has no further effect.
    """
    agent_id = scheduled.agent_id
    tier = scheduled.tier
    buf = ctx.agent_buffers.get(agent_id)

    # --- Tier 3+: stub block (minimal token cost) ---
    if tier >= 3:
        return {
            "agent_id": agent_id,
            "tier": tier,
            "ticks_since_last_action": scheduled.ticks_since_last_action,
            "harmonic_state": _build_harmonic_data(agent_id, harmonic_state, tier),
        }

    # --- Tier 2: minimal block ---
    if tier == 2:
        recent_events = []
        if buf:
            recent_events = [e.raw_data for e in buf.events[-2:]]
        block: dict[str, Any] = {
            "agent_id": agent_id,
            "tier": tier,
            "ticks_since_last_action": scheduled.ticks_since_last_action,
            "harmonic_state": _build_harmonic_data(agent_id, harmonic_state, tier),
        }
        if recent_events:
            block["recent_events"] = recent_events
        return block

    # --- Tier 1: abbreviated block ---
    if tier == 1:
        recent_events = []
        if buf:
            recent_events = [e.raw_data for e in buf.events[-5:]]
        memory = buf.memory if buf else TieredMemory()

        block = {
            "agent_id": agent_id,
            "tier": tier,
            "ticks_since_last_action": scheduled.ticks_since_last_action,
            "harmonic_state": _build_harmonic_data(agent_id, harmonic_state, tier),
        }
        if recent_events:
            block["recent_events"] = recent_events

        # Dialogue history — abbreviated: last 3 entries, trimmed by urgency
        if urgency < 0.7 and memory.verbatim:
            depth = max(1, min(3, int(len(memory.verbatim) * (1.0 - urgency))))
            block["dialogue_history"] = memory.verbatim[-depth:]

        # Active task survives at T1
        if buf and buf.active_task:
            block["active_task"] = buf.active_task

        # Emotional dynamics at T1 (curvature is useful for LLM calibration)
        if emotional_deltas is not None:
            delta = emotional_deltas.get(agent_id)
            if delta is not None:
                block["emotional_dynamics"] = {
                    "curvature": round(delta.curvature, 4),
                }

        return block

    # --- Tier 0: full block (with curvature-driven truncation) ---
    recent_events = []
    if buf:
        recent_events = [e.raw_data for e in buf.events[-10:]]

    memory = buf.memory if buf else TieredMemory()
    harmonic_data = _build_harmonic_data(agent_id, harmonic_state, tier)

    block = {
        "agent_id": agent_id,
        "tier": tier,
        "ticks_since_last_action": scheduled.ticks_since_last_action,
        "harmonic_state": harmonic_data,
        "recent_events": recent_events,
    }

    # --- Curvature-driven truncation gradient (Tier 0 only) ---
    if urgency < 0.7:
        history_depth = max(1, int(len(memory.verbatim) * (1.0 - urgency)))
        if memory.verbatim:
            block["dialogue_history"] = memory.verbatim[-history_depth:]

    if urgency < 0.5:
        if memory.compressed:
            block["compressed_history"] = memory.compressed
        if memory.keywords:
            block["distant_memories"] = memory.keywords

    if urgency < 0.7:
        if fact_pool is not None:
            fact_limit = TIER_FACT_LIMITS.get(tier, 2)
            all_present = ["Player"] + list(present_ids)
            private_facts = fact_pool.query_private(
                agent_id, all_present, limit=fact_limit,
            )
            if private_facts:
                block["private_knowledge"] = [f.content for f in private_facts]

    # State history from Qdrant (Tier 0 only)
    if memory_bundle is not None:
        state_history: dict[str, Any] = {}
        if memory_bundle.recent:
            state_history["recent"] = memory_bundle.recent
        if memory_bundle.summaries:
            state_history["summaries"] = memory_bundle.summaries
        if memory_bundle.expandable_refs:
            state_history["expandable_refs"] = memory_bundle.expandable_refs
        if state_history:
            block["state_history"] = state_history

    if buf and buf.active_task:
        block["active_task"] = buf.active_task

    # Emotional dynamics — full at Tier 0
    if emotional_deltas is not None:
        delta = emotional_deltas.get(agent_id)
        if delta is not None:
            block["emotional_dynamics"] = {
                "curvature": round(delta.curvature, 4),
                "snap": round(delta.snap, 4),
                "tension": round(1.0 - delta.coherence, 4),
            }

    return block


def _build_harmonic_data(
    agent_id: str,
    harmonic_state: "HarmonicState | None",
    tier: int = 0,
) -> dict[str, Any]:
    """Extract harmonic buffer data scaled by tier.

    Tier 0 (Full): base_vector + all three buffer tiers (fast/medium/slow).
        The agent's full internal emotional state.
    Tier 1 (Abbreviated): base_vector + curvature scalar. No buffer traces.
    Tier 2+ (Minimal/Stub): base_vector only.
    """
    if harmonic_state is None:
        return {"base_vector": list(ZERO_SEMAGRAM)}

    buf = harmonic_state._buffers.get(agent_id)
    if buf is None or not buf._initialized:
        return {"base_vector": list(ZERO_SEMAGRAM)}

    fast_list = [round(v, 4) for v in buf.fast.tolist()]

    # Tier 2+: base_vector only
    if tier >= 2:
        return {"base_vector": fast_list}

    # Tier 1: base_vector + curvature (lightweight dynamics indicator)
    if tier == 1:
        data: dict[str, Any] = {"base_vector": fast_list}
        if buf._last_delta is not None:
            data["curvature"] = round(buf._last_delta.curvature, 4)
        return data

    # Tier 0: full buffer traces
    return {
        "base_vector": fast_list,
        "buffers": {
            "fast": fast_list,
            "medium": [round(v, 4) for v in buf.medium.tolist()],
            "slow": [round(v, 4) for v in buf.slow.tolist()],
        },
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_location(ctx: TurnContext) -> str:
    """Extract current location from world events or default."""
    for event in reversed(ctx.world_events):
        if event.event_type == "location":
            return event.raw_data
    return "Unknown"
