"""Tests for progeny.src.prompt_formatter."""
from __future__ import annotations

import json

from shared.schemas import TypedEvent
from progeny.src.event_accumulator import AgentBuffer, TurnContext
from progeny.src.agent_scheduler import ScheduledAgent
from progeny.src.prompt_formatter import build_prompt, SYSTEM_PROMPT, INSTRUCTION_PROMPT


def _make_context(
    player_input: str = "What do you think about the civil war?",
    active_npc_ids: list[str] | None = None,
) -> TurnContext:
    if active_npc_ids is None:
        active_npc_ids = ["Lydia"]
    buffers = {}
    for npc in active_npc_ids:
        buffers[npc] = AgentBuffer(agent_id=npc)
    return TurnContext(
        player_input=player_input,
        agent_buffers=buffers,
        active_npc_ids=active_npc_ids,
        world_events=[],
        session_events=[],
    )


def _make_roster(agent_ids: list[str]) -> list[ScheduledAgent]:
    return [
        ScheduledAgent(agent_id=aid, tier=0, ticks_since_last_action=0)
        for aid in agent_ids
    ]


class TestMessageStructure:
    def test_two_messages(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        assert len(messages) == 2

    def test_message_roles(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_system_prompt_content(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        system = messages[0]["content"]
        assert "Many-Mind Kernel" in system
        assert "ACTOR VALUES" in system
        assert "Aggression" in system
        assert "RESPONSE FORMAT" in system

    def test_instruction_in_user_message(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        assert "valid JSON" in messages[1]["content"]


class TestDataPayload:
    def test_data_is_valid_json(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert "agents" in data
        assert "player_input" in data
        assert "group_context" in data

    def test_group_context_has_present_npcs(self):
        ctx = _make_context(active_npc_ids=["Lydia", "Belethor"])
        roster = _make_roster(["Lydia", "Belethor"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert "present_npcs" in data["group_context"]
        assert set(data["group_context"]["present_npcs"]) == {"Lydia", "Belethor"}

    def test_group_context_has_location(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert "location" in data["group_context"]

    def test_player_input_included(self):
        ctx = _make_context("Tell me about dragons")
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert data["player_input"]["text"] == "Tell me about dragons"

    def test_agent_count_matches_roster(self):
        ctx = _make_context(active_npc_ids=["Lydia", "Belethor", "Ysolda"])
        roster = _make_roster(["Lydia", "Belethor", "Ysolda"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert len(data["agents"]) == 3


class TestAgentBlocks:
    def test_agent_block_has_required_fields(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert agent["agent_id"] == "Lydia"
        assert "tier" in agent
        assert "ticks_since_last_action" in agent
        assert "harmonic_state" in agent
        assert len(agent["harmonic_state"]["base_vector"]) == 9

    def test_agent_block_includes_dialogue_history(self):
        ctx = _make_context()
        ctx.agent_buffers["Lydia"].dialogue_history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "I am sworn to carry your burdens."},
        ]
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert len(agent["dialogue_history"]) == 2

    def test_agent_block_includes_recent_events(self):
        ctx = _make_context()
        ctx.agent_buffers["Lydia"].events = [
            TypedEvent(
                event_type="_speech",
                local_ts="2024-01-01",
                game_ts=100.0,
                raw_data="I heard something",
            ),
        ]
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert len(agent["recent_events"]) == 1

    def test_ticks_since_last_action_propagated(self):
        ctx = _make_context()
        roster = [ScheduledAgent(agent_id="Lydia", tier=0, ticks_since_last_action=5)]
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert data["agents"][0]["ticks_since_last_action"] == 5


class TestFactPoolIntegration:
    def test_shared_facts_in_group_context(self):
        """Facts known by ALL present NPCs appear in group_context."""
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        pool.add_fact("Dragon attacked", "event", 100.0, ["Player", "Lydia"])

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_knowledge" in gc
        assert "Dragon attacked" in gc["shared_knowledge"]

    def test_private_facts_in_agent_block(self):
        """Facts known by one NPC but not all go to the agent's private block."""
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        # Lydia knows a secret; Player doesn't
        pool.add_fact("Overheard Thalmor plot", "event", 100.0, ["Lydia"])
        # Both know this one (goes to shared, not private)
        pool.add_fact("Dragon attacked", "event", 101.0, ["Player", "Lydia"])

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert "private_knowledge" in agent
        assert "Overheard Thalmor plot" in agent["private_knowledge"]
        # Dragon attacked is shared, not private
        assert "Dragon attacked" not in agent.get("private_knowledge", [])

    def test_no_fact_pool_still_works(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)  # no fact_pool
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        # No private_knowledge key when no facts
        assert "private_knowledge" not in data["agents"][0]

    def test_lore_in_group_context(self):
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        pool.bit_index.get_or_assign("Lydia")
        pool.add_lore("Skyrim is the homeland of the Nords")

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "lore" in gc
        assert "Skyrim" in gc["lore"][0]


class TestGroupTimeline:
    def test_shared_recent_in_group_context(self):
        """Group memory verbatim entries appear as shared_recent."""
        from progeny.src.event_accumulator import TieredMemory
        ctx = _make_context()
        ctx.group_memory = TieredMemory(
            verbatim=[{"role": "Player", "content": "Watch out!"}],
        )
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_recent" in gc
        assert gc["shared_recent"][0]["content"] == "Watch out!"

    def test_shared_history_in_group_context(self):
        """Group memory compressed entries appear as shared_history."""
        from progeny.src.event_accumulator import TieredMemory
        ctx = _make_context()
        ctx.group_memory = TieredMemory(
            compressed=["Player: Watch out! [combat]"],
        )
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_history" in gc
        assert "Watch out" in gc["shared_history"][0]

    def test_shared_anchors_in_group_context(self):
        """Group memory keywords appear as shared_anchors."""
        from progeny.src.event_accumulator import TieredMemory
        ctx = _make_context()
        ctx.group_memory = TieredMemory(
            keywords=["Lydia:afraid | dragon | combat"],
        )
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_anchors" in gc
        assert "dragon" in gc["shared_anchors"][0]

    def test_empty_group_memory_omitted(self):
        """Empty group memory fields don't appear in the prompt."""
        ctx = _make_context()  # default TieredMemory is empty
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "shared_recent" not in gc
        assert "shared_history" not in gc
        assert "shared_anchors" not in gc


class TestCurvatureTruncation:
    """Verify curvature-driven prompt truncation — cognitive focus under pressure."""

    def test_calm_includes_full_group_history(self):
        """Low urgency: full shared_recent, shared_history, lore."""
        from progeny.src.event_accumulator import TieredMemory
        from progeny.src.fact_pool import FactPool

        ctx = _make_context()
        ctx.group_memory = TieredMemory(
            verbatim=[{"role": "Player", "content": "Tell me a story."}],
            compressed=["Player: discussed lore"],
            keywords=["lore | history"],
        )
        pool = FactPool()
        pool.bit_index.get_or_assign("Lydia")
        pool.add_lore("Skyrim is the homeland of the Nords")

        roster = _make_roster(["Lydia"])
        # No emotional deltas → urgency = 0.0
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]

        assert "shared_recent" in gc
        assert "shared_history" in gc
        assert "shared_anchors" in gc
        assert "lore" in gc

    def test_crisis_strips_to_anchors_only(self):
        """High urgency: only anchors + display + events survive."""
        from progeny.src.event_accumulator import TieredMemory
        from mindcore.harmonic_buffer import EmotionalDelta

        ctx = _make_context()
        ctx.group_memory = TieredMemory(
            verbatim=[{"role": "Player", "content": "Run!"}],
            compressed=["Player: ran from dragon"],
            keywords=["dragon | flee | danger"],
        )
        roster = _make_roster(["Lydia"])
        # High curvature → urgency = 1.0
        high_curv_delta = EmotionalDelta(
            semagram=[0.0]*9, delta=[0.0]*9,
            curvature=0.8, snap=0.5, coherence=0.3, lambda_t=0.9,
        )
        deltas = {"Lydia": high_curv_delta}
        messages = build_prompt(ctx, roster, emotional_deltas=deltas)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]

        # Anchors survive
        assert "shared_anchors" in gc
        # Full history stripped
        assert "shared_recent" not in gc
        assert "shared_history" not in gc
        assert "lore" not in gc

    def test_crisis_strips_agent_deep_memory(self):
        """High urgency: agent block drops dialogue_history and compressed."""
        from mindcore.harmonic_buffer import EmotionalDelta

        ctx = _make_context()
        ctx.agent_buffers["Lydia"].memory.verbatim = [
            {"role": "user", "content": "old conversation"},
        ]
        ctx.agent_buffers["Lydia"].memory.compressed = ["old summary"]
        ctx.agent_buffers["Lydia"].memory.keywords = ["old | tags"]

        roster = _make_roster(["Lydia"])
        high_curv_delta = EmotionalDelta(
            semagram=[0.0]*9, delta=[0.0]*9,
            curvature=0.8, snap=0.5, coherence=0.3, lambda_t=0.9,
        )
        deltas = {"Lydia": high_curv_delta}
        messages = build_prompt(ctx, roster, emotional_deltas=deltas)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]

        # Deep memory stripped under crisis
        assert "dialogue_history" not in agent
        assert "compressed_history" not in agent
        assert "distant_memories" not in agent
        # Essentials survive
        assert "harmonic_state" in agent
        assert "recent_events" in agent
        assert "emotional_dynamics" in agent

    def test_calm_agent_has_full_depth(self):
        """Low urgency: agent block has all memory tiers."""
        ctx = _make_context()
        ctx.agent_buffers["Lydia"].memory.verbatim = [
            {"role": "user", "content": "conversation"},
        ]
        ctx.agent_buffers["Lydia"].memory.compressed = ["summary"]
        ctx.agent_buffers["Lydia"].memory.keywords = ["tags"]

        roster = _make_roster(["Lydia"])
        # No deltas → urgency = 0.0
        messages = build_prompt(ctx, roster)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]

        assert "dialogue_history" in agent
        assert "compressed_history" in agent
        assert "distant_memories" in agent


class TestTierScaling:
    """Verify prompt tier-scaling — agent blocks scale with tier level.

    Living Doc §Agent Priority Paging:
      Tier 0 (Full): all fields, full buffer traces.
      Tier 1 (Abbreviated): base_vector + curvature, trimmed events/history.
      Tier 2 (Minimal): base_vector only, brief events.
      Tier 3+ (Stub): base_vector only, nothing else.
    """

    def _parse_agents(self, messages):
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        return {a["agent_id"]: a for a in data["agents"]}

    def test_tier0_has_full_buffers(self):
        """Tier 0 agent gets full buffer traces (fast/medium/slow)."""
        from mindcore.harmonic_buffer import HarmonicState
        state = HarmonicState()
        state.update("Lydia", [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3])

        ctx = _make_context()
        roster = [ScheduledAgent(agent_id="Lydia", tier=0, ticks_since_last_action=0)]
        messages = build_prompt(ctx, roster, harmonic_state=state)
        agents = self._parse_agents(messages)
        lydia = agents["Lydia"]

        assert "buffers" in lydia["harmonic_state"]
        assert "fast" in lydia["harmonic_state"]["buffers"]
        assert "medium" in lydia["harmonic_state"]["buffers"]
        assert "slow" in lydia["harmonic_state"]["buffers"]

    def test_tier1_has_curvature_no_buffers(self):
        """Tier 1 agent gets base_vector + curvature, no buffer traces."""
        from mindcore.harmonic_buffer import HarmonicState
        state = HarmonicState()
        state.update("Belethor", [0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.3, 0.0, 0.2])

        ctx = _make_context(active_npc_ids=["Belethor"])
        roster = [ScheduledAgent(agent_id="Belethor", tier=1, ticks_since_last_action=2)]
        messages = build_prompt(ctx, roster, harmonic_state=state)
        agents = self._parse_agents(messages)
        belethor = agents["Belethor"]

        assert len(belethor["harmonic_state"]["base_vector"]) == 9
        assert "curvature" in belethor["harmonic_state"]
        assert "buffers" not in belethor["harmonic_state"]

    def test_tier2_has_base_vector_only(self):
        """Tier 2 agent gets base_vector only."""
        from mindcore.harmonic_buffer import HarmonicState
        state = HarmonicState()
        state.update("Ysolda", [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.1])

        ctx = _make_context(active_npc_ids=["Ysolda"])
        roster = [ScheduledAgent(agent_id="Ysolda", tier=2, ticks_since_last_action=5)]
        messages = build_prompt(ctx, roster, harmonic_state=state)
        agents = self._parse_agents(messages)
        ysolda = agents["Ysolda"]

        assert len(ysolda["harmonic_state"]["base_vector"]) == 9
        assert "buffers" not in ysolda["harmonic_state"]
        assert "curvature" not in ysolda["harmonic_state"]

    def test_tier3_is_stub(self):
        """Tier 3 agent: minimal stub — just agent_id, tier, ticks, base_vector."""
        from mindcore.harmonic_buffer import HarmonicState
        state = HarmonicState()
        state.update("Heimskr", [0.0, 0.0, 0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.9])

        ctx = _make_context(active_npc_ids=["Heimskr"])
        roster = [ScheduledAgent(agent_id="Heimskr", tier=3, ticks_since_last_action=47)]
        messages = build_prompt(ctx, roster, harmonic_state=state)
        agents = self._parse_agents(messages)
        heimskr = agents["Heimskr"]

        assert heimskr["tier"] == 3
        assert heimskr["ticks_since_last_action"] == 47
        assert len(heimskr["harmonic_state"]["base_vector"]) == 9
        # Stub should NOT have these fields
        assert "recent_events" not in heimskr
        assert "dialogue_history" not in heimskr
        assert "emotional_dynamics" not in heimskr
        assert "buffers" not in heimskr["harmonic_state"]

    def test_tier2_no_dialogue_history(self):
        """Tier 2 agents don't get dialogue history."""
        ctx = _make_context(active_npc_ids=["Ysolda"])
        ctx.agent_buffers["Ysolda"] = AgentBuffer(agent_id="Ysolda")
        ctx.agent_buffers["Ysolda"].dialogue_history = [
            {"role": "user", "content": "Hello"},
        ]
        roster = [ScheduledAgent(agent_id="Ysolda", tier=2, ticks_since_last_action=5)]
        messages = build_prompt(ctx, roster)
        agents = self._parse_agents(messages)
        assert "dialogue_history" not in agents["Ysolda"]

    def test_tier1_dialogue_trimmed_to_3(self):
        """Tier 1 agents get dialogue history capped at 3 entries."""
        ctx = _make_context(active_npc_ids=["Belethor"])
        ctx.agent_buffers["Belethor"] = AgentBuffer(agent_id="Belethor")
        ctx.agent_buffers["Belethor"].dialogue_history = [
            {"role": "user", "content": f"msg{i}"} for i in range(10)
        ]
        roster = [ScheduledAgent(agent_id="Belethor", tier=1, ticks_since_last_action=2)]
        messages = build_prompt(ctx, roster)
        agents = self._parse_agents(messages)
        history = agents["Belethor"].get("dialogue_history", [])
        assert len(history) <= 3

    def test_tier2_events_limited_to_2(self):
        """Tier 2 agents get at most 2 recent events."""
        ctx = _make_context(active_npc_ids=["Ysolda"])
        ctx.agent_buffers["Ysolda"] = AgentBuffer(agent_id="Ysolda")
        for i in range(10):
            ctx.agent_buffers["Ysolda"].events.append(
                TypedEvent(event_type="info", local_ts="ts", game_ts=float(i), raw_data=f"event{i}")
            )
        roster = [ScheduledAgent(agent_id="Ysolda", tier=2, ticks_since_last_action=5)]
        messages = build_prompt(ctx, roster)
        agents = self._parse_agents(messages)
        events = agents["Ysolda"].get("recent_events", [])
        assert len(events) <= 2

    def test_tier1_events_limited_to_5(self):
        """Tier 1 agents get at most 5 recent events."""
        ctx = _make_context(active_npc_ids=["Belethor"])
        ctx.agent_buffers["Belethor"] = AgentBuffer(agent_id="Belethor")
        for i in range(20):
            ctx.agent_buffers["Belethor"].events.append(
                TypedEvent(event_type="info", local_ts="ts", game_ts=float(i), raw_data=f"event{i}")
            )
        roster = [ScheduledAgent(agent_id="Belethor", tier=1, ticks_since_last_action=2)]
        messages = build_prompt(ctx, roster)
        agents = self._parse_agents(messages)
        events = agents["Belethor"].get("recent_events", [])
        assert len(events) <= 5

    def test_mixed_tier_roster(self):
        """Mixed roster: each agent gets tier-appropriate block granularity."""
        from mindcore.harmonic_buffer import HarmonicState
        state = HarmonicState()
        for name in ["Lydia", "Belethor", "Ysolda", "Heimskr"]:
            state.update(name, [0.1] * 9)

        ctx = _make_context(active_npc_ids=["Lydia", "Belethor", "Ysolda", "Heimskr"])
        roster = [
            ScheduledAgent(agent_id="Lydia", tier=0, ticks_since_last_action=0),
            ScheduledAgent(agent_id="Belethor", tier=1, ticks_since_last_action=2),
            ScheduledAgent(agent_id="Ysolda", tier=2, ticks_since_last_action=6),
            ScheduledAgent(agent_id="Heimskr", tier=3, ticks_since_last_action=47),
        ]
        messages = build_prompt(ctx, roster, harmonic_state=state)
        agents = self._parse_agents(messages)

        # Tier 0: full buffers
        assert "buffers" in agents["Lydia"]["harmonic_state"]
        # Tier 1: curvature, no buffers
        assert "curvature" in agents["Belethor"]["harmonic_state"]
        assert "buffers" not in agents["Belethor"]["harmonic_state"]
        # Tier 2: base_vector only
        assert "buffers" not in agents["Ysolda"]["harmonic_state"]
        assert "curvature" not in agents["Ysolda"]["harmonic_state"]
        # Tier 3: stub
        assert "recent_events" not in agents["Heimskr"]

    def test_tier1_emotional_dynamics_curvature_only(self):
        """Tier 1 gets curvature in emotional_dynamics, not snap/tension."""
        from mindcore.harmonic_buffer import EmotionalDelta
        ctx = _make_context(active_npc_ids=["Belethor"])
        roster = [ScheduledAgent(agent_id="Belethor", tier=1, ticks_since_last_action=2)]
        delta = EmotionalDelta(
            semagram=[0.0]*9, delta=[0.0]*9,
            curvature=0.15, snap=0.05, coherence=0.8, lambda_t=0.4,
        )
        messages = build_prompt(ctx, roster, emotional_deltas={"Belethor": delta})
        agents = self._parse_agents(messages)
        dynamics = agents["Belethor"].get("emotional_dynamics", {})
        assert "curvature" in dynamics
        assert "snap" not in dynamics  # T1 doesn't get snap
        assert "tension" not in dynamics

    def test_tier0_no_private_knowledge_excluded(self):
        """Tier 0 agent still gets private_knowledge when fact_pool provided."""
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        pool.add_fact("Secret plan", "event", 100.0, ["Lydia"])

        ctx = _make_context()
        roster = [ScheduledAgent(agent_id="Lydia", tier=0, ticks_since_last_action=0)]
        messages = build_prompt(ctx, roster, fact_pool=pool)
        agents = self._parse_agents(messages)
        assert "private_knowledge" in agents["Lydia"]

    def test_tier1_no_private_knowledge(self):
        """Tier 1 agents don't get private_knowledge."""
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        pool.add_fact("Secret plan", "event", 100.0, ["Belethor"])

        ctx = _make_context(active_npc_ids=["Belethor"])
        roster = [ScheduledAgent(agent_id="Belethor", tier=1, ticks_since_last_action=2)]
        messages = build_prompt(ctx, roster, fact_pool=pool)
        agents = self._parse_agents(messages)
        assert "private_knowledge" not in agents["Belethor"]


class TestGroupDisplay:
    def test_group_display_shows_fast_buffer(self):
        """NPCs with emotional state show their fast buffer in group display."""
        from mindcore.harmonic_buffer import HarmonicState
        state = HarmonicState()
        angry = [0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.3]
        state.update("Lydia", angry)

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, harmonic_state=state)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        gc = data["group_context"]
        assert "group_display" in gc
        assert len(gc["group_display"]) == 1
        assert gc["group_display"][0]["name"] == "Lydia"
        assert len(gc["group_display"][0]["demeanor"]) == 9

    def test_zero_state_npcs_excluded_from_display(self):
        """NPCs with no emotional state yet don't appear in group display."""
        from mindcore.harmonic_buffer import HarmonicState
        state = HarmonicState()  # No updates — all zero

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, harmonic_state=state)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        # group_display should be empty or absent
        display = data["group_context"].get("group_display", [])
        assert display == []
