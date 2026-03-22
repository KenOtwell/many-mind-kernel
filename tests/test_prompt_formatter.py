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
        # User message contains JSON payload followed by instruction text
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert "agents" in data
        assert "player_input" in data
        assert "present_npcs" in data

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
        assert "base_vector" in agent
        assert len(agent["base_vector"]) == 9

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
    def test_known_world_in_agent_block_when_fact_pool_provided(self):
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        pool.add_fact("Dragon attacked", "event", 100.0, ["Player", "Lydia"])

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert "known_world" in agent
        assert len(agent["known_world"]) == 1
        assert agent["known_world"][0]["content"] == "Dragon attacked"

    def test_no_fact_pool_still_works(self):
        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster)  # no fact_pool
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert data["agents"][0]["known_world"] == []

    def test_lore_context_included(self):
        from progeny.src.fact_pool import FactPool
        pool = FactPool()
        pool.bit_index.get_or_assign("Lydia")
        pool.add_lore("Skyrim is the homeland of the Nords")

        ctx = _make_context()
        roster = _make_roster(["Lydia"])
        messages = build_prompt(ctx, roster, fact_pool=pool)
        json_part = messages[1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        assert "lore_context" in data
        assert "Skyrim" in data["lore_context"][0]
