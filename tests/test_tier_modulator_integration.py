"""End-to-end integration test for dynamic modulators + prompt tier-scaling.

Exercises the full chain without external dependencies (no embedding model,
no Qdrant, no LLM):

  1. Construct modulators from engine preset values
  2. Apply modulators to harmonic state for multiple agents
  3. Feed emotional events through the buffer (updates)
  4. Build a prompt with a mixed-tier roster
  5. Verify:
     - Modulator effects are visible in buffer state
     - Tier-scaled blocks have correct granularity
     - The prompt is valid JSON with the expected structure

This is the "walking through Whiterun" test from the Living Doc:
  Lydia (Tier 0, companion) — Brave, Aggressive
  Belethor (Tier 1, near-field) — Cautious shopkeeper
  Ysolda (Tier 2, mid-field) — Friendly merchant
  Heimskr (Tier 3, far-field) — Happy preacher
"""
from __future__ import annotations

import json

from shared.constants import EMOTIONAL_DIM
from shared.schemas import TypedEvent
from progeny.src.agent_scheduler import ScheduledAgent
from progeny.src.event_accumulator import AgentBuffer, TurnContext
from mindcore.harmonic_buffer import (
    HarmonicState,
    build_modulators,
)
from progeny.src.prompt_formatter import build_prompt


# ---------------------------------------------------------------------------
# Engine presets for our cast (from the Living Doc)
# ---------------------------------------------------------------------------

LYDIA_PRESETS = dict(aggression=2, confidence=3, morality=3, mood=0, assistance=2)
BELETHOR_PRESETS = dict(aggression=0, confidence=1, morality=2, mood=3, assistance=0)
YSOLDA_PRESETS = dict(aggression=0, confidence=2, morality=3, mood=3, assistance=1)
HEIMSKR_PRESETS = dict(aggression=0, confidence=2, morality=3, mood=3, assistance=0)

AMBUSH_SEMAGRAM = [0.7, 0.5, 0.0, 0.0, 0.4, 0.0, 0.0, -0.2, 0.5]
CALM_SEMAGRAM = [0.0, 0.0, 0.3, 0.0, 0.1, 0.0, 0.5, 0.6, 0.2]
NEUTRAL_SEMAGRAM = [0.0] * EMOTIONAL_DIM


def _setup_whiterun_scene() -> tuple[HarmonicState, TurnContext, list[ScheduledAgent]]:
    """Set up a Whiterun scene with four NPCs at different tiers."""
    state = HarmonicState()

    # Apply modulators from engine presets
    state.apply_modulators("Lydia", build_modulators(**LYDIA_PRESETS))
    state.apply_modulators("Belethor", build_modulators(**BELETHOR_PRESETS))
    state.apply_modulators("Ysolda", build_modulators(**YSOLDA_PRESETS))
    state.apply_modulators("Heimskr", build_modulators(**HEIMSKR_PRESETS))

    # Warm up all buffers with a neutral baseline
    for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
        state.update(name, NEUTRAL_SEMAGRAM)

    # Build turn context
    npc_ids = ["Lydia", "Belethor", "Ysolda", "Heimskr"]
    buffers = {name: AgentBuffer(agent_id=name) for name in npc_ids}

    # Add some events for Lydia (Tier 0) and Belethor (Tier 1)
    for i in range(12):
        buffers["Lydia"].events.append(
            TypedEvent(event_type="info", local_ts="ts", game_ts=float(i),
                       raw_data=f"Lydia event {i}")
        )
    for i in range(8):
        buffers["Belethor"].events.append(
            TypedEvent(event_type="info", local_ts="ts", game_ts=float(i),
                       raw_data=f"Belethor event {i}")
        )

    ctx = TurnContext(
        player_input="What do you think about this ambush?",
        agent_buffers=buffers,
        active_npc_ids=npc_ids,
        world_events=[
            TypedEvent(event_type="location", local_ts="ts", game_ts=100.0,
                       raw_data="Whiterun Market"),
        ],
        session_events=[],
    )

    roster = [
        ScheduledAgent(agent_id="Lydia", tier=0, ticks_since_last_action=0),
        ScheduledAgent(agent_id="Belethor", tier=1, ticks_since_last_action=3),
        ScheduledAgent(agent_id="Ysolda", tier=2, ticks_since_last_action=7),
        ScheduledAgent(agent_id="Heimskr", tier=3, ticks_since_last_action=47),
    ]

    return state, ctx, roster


class TestEndToEndModulatorsTierScaling:
    """Full chain: presets → modulators → emotional updates → tier-scaled prompt."""

    def test_modulator_effects_visible_in_buffer_state(self):
        """Different presets produce different emotional trajectories from same input."""
        state, ctx, roster = _setup_whiterun_scene()

        # Feed the same ambush semagram to all agents
        for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
            state.update(name, AMBUSH_SEMAGRAM)

        # Lydia (Aggressive=2, Confident=3): anger should track faster,
        # fear should be dampened
        lydia_buf = state._buffers["Lydia"]
        belethor_buf = state._buffers["Belethor"]

        # Lydia has aggression gain — anger (dim 1) should be higher on fast
        # than Belethor (who has aggression=0)
        assert lydia_buf.fast[1] > belethor_buf.fast[1]

        # Lydia has higher confidence (3 vs 1) — fear (dim 0) should be
        # more dampened
        assert lydia_buf.fast[0] < belethor_buf.fast[0]

    def test_mood_pull_visible_after_calm_period(self):
        """Happy NPCs drift toward joy axis during calm inputs."""
        state, ctx, roster = _setup_whiterun_scene()

        # Feed many calm/neutral inputs — mood pull should create drift
        for _ in range(50):
            for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
                state.update(name, NEUTRAL_SEMAGRAM)

        # Belethor, Ysolda, Heimskr all have mood=Happy (joy axis 6)
        # Lydia has mood=Neutral (no pull)
        assert state._buffers["Belethor"].fast[6] > 0.01
        assert state._buffers["Ysolda"].fast[6] > 0.01
        assert state._buffers["Heimskr"].fast[6] > 0.01
        assert abs(state._buffers["Lydia"].fast[6]) < 0.001

    def test_tier_scaled_prompt_structure(self):
        """Mixed-tier prompt has correct granularity per agent."""
        state, ctx, roster = _setup_whiterun_scene()

        # Feed some emotional data so buffers have meaningful state
        for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
            state.update(name, AMBUSH_SEMAGRAM)

        # Capture deltas for prompt building
        deltas = {
            name: state.get_delta(name)
            for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]
        }

        messages = build_prompt(
            ctx, roster,
            harmonic_state=state,
            emotional_deltas=deltas,
        )

        # Parse and verify structure
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agents = {a["agent_id"]: a for a in data["agents"]}

        # Tier 0 (Lydia): full buffers, emotional dynamics with snap/tension
        lydia = agents["Lydia"]
        assert "buffers" in lydia["harmonic_state"]
        assert "fast" in lydia["harmonic_state"]["buffers"]
        assert "medium" in lydia["harmonic_state"]["buffers"]
        assert "slow" in lydia["harmonic_state"]["buffers"]
        assert "emotional_dynamics" in lydia
        assert "snap" in lydia["emotional_dynamics"]
        assert "tension" in lydia["emotional_dynamics"]
        # Lydia gets up to 10 events
        assert len(lydia.get("recent_events", [])) <= 10

        # Tier 1 (Belethor): curvature, no buffers
        belethor = agents["Belethor"]
        assert "curvature" in belethor["harmonic_state"]
        assert "buffers" not in belethor["harmonic_state"]
        # Belethor gets up to 5 events
        assert len(belethor.get("recent_events", [])) <= 5
        # T1 emotional dynamics: curvature only
        if "emotional_dynamics" in belethor:
            assert "snap" not in belethor["emotional_dynamics"]

        # Tier 2 (Ysolda): base_vector only
        ysolda = agents["Ysolda"]
        assert "buffers" not in ysolda["harmonic_state"]
        assert "curvature" not in ysolda["harmonic_state"]
        assert "dialogue_history" not in ysolda
        assert "emotional_dynamics" not in ysolda

        # Tier 3 (Heimskr): stub
        heimskr = agents["Heimskr"]
        assert heimskr["ticks_since_last_action"] == 47
        assert "recent_events" not in heimskr
        assert "emotional_dynamics" not in heimskr

    def test_prompt_is_valid_json(self):
        """The entire prompt data payload is valid JSON."""
        state, ctx, roster = _setup_whiterun_scene()
        for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
            state.update(name, CALM_SEMAGRAM)

        messages = build_prompt(ctx, roster, harmonic_state=state)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)

        assert "group_context" in data
        assert "agents" in data
        assert "player_input" in data
        assert len(data["agents"]) == 4
        assert data["group_context"]["location"] == "Whiterun Market"
        assert data["player_input"]["text"] == "What do you think about this ambush?"

    def test_group_display_reflects_modulator_differences(self):
        """Group display shows different emotional faces per agent."""
        state, ctx, roster = _setup_whiterun_scene()

        # Same ambush, different modulators → different fast buffers
        for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
            state.update(name, AMBUSH_SEMAGRAM)

        messages = build_prompt(
            ctx, roster,
            all_active_npc_ids=["Lydia", "Belethor", "Ysolda", "Heimskr"],
            harmonic_state=state,
        )
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        display = data["group_context"].get("group_display", [])

        # All four should have non-zero emotional state → appear in display
        names_in_display = {d["name"] for d in display}
        assert "Lydia" in names_in_display
        assert "Belethor" in names_in_display

        # Find Lydia and Belethor's demeanor
        lydia_dem = next(d for d in display if d["name"] == "Lydia")
        belethor_dem = next(d for d in display if d["name"] == "Belethor")

        # Lydia's anger (dim 1) should be higher than Belethor's
        # (aggression gain amplifies anger tracking)
        assert lydia_dem["demeanor"][1] > belethor_dem["demeanor"][1]

    def test_token_budget_tier_scaling(self):
        """Tier-scaled blocks produce substantially different sizes."""
        state, ctx, roster = _setup_whiterun_scene()
        for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
            state.update(name, CALM_SEMAGRAM)

        deltas = {name: state.get_delta(name) for name in ctx.active_npc_ids}
        messages = build_prompt(ctx, roster, harmonic_state=state,
                                emotional_deltas=deltas)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agents = {a["agent_id"]: a for a in data["agents"]}

        # Compare serialized sizes — T0 should be much larger than T3
        t0_size = len(json.dumps(agents["Lydia"]))
        t1_size = len(json.dumps(agents["Belethor"]))
        t2_size = len(json.dumps(agents["Ysolda"]))
        t3_size = len(json.dumps(agents["Heimskr"]))

        assert t0_size > t1_size > t3_size
        assert t0_size > t2_size
        # T0 should be at least 2x bigger than T3
        assert t0_size > 2 * t3_size

    def test_certainty_modulates_residual_in_prompt_data(self):
        """Uncertainty feeds back into the buffer → prompt reflects attenuated residual."""
        state, ctx, roster = _setup_whiterun_scene()

        # Set Lydia to half certainty, Belethor to full certainty
        state.set_certainty("Lydia", 0.5)
        state.set_certainty("Belethor", 1.0)

        # Feed a semagram with strong residual to both
        residual_heavy = [0.0] * EMOTIONAL_DIM
        residual_heavy[8] = 0.8  # strong residual (reality content)
        residual_heavy[1] = 0.3  # some anger for context
        for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
            state.update(name, residual_heavy)

        # Build prompt — Lydia's residual should be attenuated vs Belethor's
        deltas = {name: state.get_delta(name) for name in ctx.active_npc_ids}
        messages = build_prompt(ctx, roster, harmonic_state=state,
                                emotional_deltas=deltas)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agents = {a["agent_id"]: a for a in data["agents"]}

        # Lydia (T0) has full buffers — check residual (dim 8)
        lydia_base = agents["Lydia"]["harmonic_state"]["base_vector"]
        # Belethor (T1) has base_vector
        belethor_base = agents["Belethor"]["harmonic_state"]["base_vector"]

        # Lydia's residual should be weaker than Belethor's
        # (both got same input, but Lydia's certainty was 0.5)
        assert lydia_base[8] < belethor_base[8]
