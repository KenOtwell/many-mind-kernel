"""Tests for progeny.src.event_accumulator."""
from __future__ import annotations

from shared.schemas import TypedEvent, TickPackage
from progeny.src.event_accumulator import EventAccumulator, TurnContext
from tests.fixtures.factories import (
    make_turn_package,
    make_data_package,
    make_inputtext_event,
    make_info_event,
)


def _speech_event(speaker: str = "Lydia", text: str = "I am sworn to carry your burdens.") -> TypedEvent:
    """Helper: create a _speech event with parsed_data."""
    return TypedEvent(
        event_type="_speech",
        local_ts="2024-01-01T00:00:00",
        game_ts=100.0,
        raw_data=text,
        parsed_data={"speaker": speaker, "speech": text, "listener": "Player", "location": "Whiterun"},
    )


def _addnpc_event(name: str = "Lydia") -> TypedEvent:
    return TypedEvent(
        event_type="addnpc",
        local_ts="2024-01-01T00:00:00",
        game_ts=100.0,
        raw_data=f"{name}@base@female@Nord",
        parsed_data={"name": name, "race": "Nord"},
    )


def _updatestats_event(npc_name: str = "Lydia") -> TypedEvent:
    return TypedEvent(
        event_type="updatestats",
        local_ts="2024-01-01T00:00:00",
        game_ts=100.0,
        raw_data=f"{npc_name}@25@100@100@50@50@80@80@1.0",
        parsed_data={"npc_name": npc_name, "level": 25, "health": 100.0},
    )


def _location_event(location: str = "WhiterunExterior") -> TypedEvent:
    return TypedEvent(
        event_type="location",
        local_ts="2024-01-01T00:00:00",
        game_ts=100.0,
        raw_data=location,
        parsed_data=None,
    )


def _init_event() -> TypedEvent:
    return TypedEvent(
        event_type="init",
        local_ts="2024-01-01T00:00:00",
        game_ts=0.0,
        raw_data="1.0.0",
        parsed_data=None,
    )


# ---------------------------------------------------------------------------
# Turn boundary detection
# ---------------------------------------------------------------------------

class TestTurnBoundaryDetection:
    def test_turn_trigger_returns_context(self):
        acc = EventAccumulator()
        pkg = make_turn_package("Hello Lydia")
        result = acc.ingest(pkg)
        assert result is not None
        assert isinstance(result, TurnContext)
        assert result.player_input == "Hello Lydia"

    def test_data_only_returns_none(self):
        acc = EventAccumulator()
        pkg = make_data_package()
        result = acc.ingest(pkg)
        assert result is None

    def test_inputtext_s_also_detected_as_player_input(self):
        acc = EventAccumulator()
        event = TypedEvent(
            event_type="inputtext_s",
            local_ts="2024-01-01T00:00:00",
            game_ts=100.0,
            raw_data="Help me with something",
            parsed_data=None,
        )
        pkg = TickPackage(events=[event], active_npc_ids=["Lydia"])
        result = acc.ingest(pkg)
        assert result is not None
        assert result.player_input == "Help me with something"

    def test_active_npc_ids_propagated(self):
        acc = EventAccumulator()
        pkg = make_turn_package("Hi", active_npc_ids=["Lydia", "Belethor"])
        result = acc.ingest(pkg)
        assert result is not None
        assert result.active_npc_ids == ["Lydia", "Belethor"]


# ---------------------------------------------------------------------------
# Agent extraction from parsed_data
# ---------------------------------------------------------------------------

class TestAgentExtraction:
    def test_speech_routes_to_speaker(self):
        acc = EventAccumulator()
        speech = _speech_event("Belethor", "Do come back")
        pkg = TickPackage(
            events=[speech, make_inputtext_event()],
            active_npc_ids=["Belethor"],
        )
        ctx = acc.ingest(pkg)
        assert ctx is not None
        assert "Belethor" in ctx.agent_buffers
        assert len(ctx.agent_buffers["Belethor"].events) == 1

    def test_addnpc_routes_to_name(self):
        acc = EventAccumulator()
        pkg = TickPackage(
            events=[_addnpc_event("Ysolda"), make_inputtext_event()],
            active_npc_ids=["Ysolda"],
        )
        ctx = acc.ingest(pkg)
        assert "Ysolda" in ctx.agent_buffers

    def test_updatestats_routes_to_npc_name(self):
        acc = EventAccumulator()
        pkg = TickPackage(
            events=[_updatestats_event("Lydia"), make_inputtext_event()],
            active_npc_ids=["Lydia"],
        )
        ctx = acc.ingest(pkg)
        assert "Lydia" in ctx.agent_buffers

    def test_unparsed_event_goes_to_world(self):
        acc = EventAccumulator()
        info = make_info_event("Something happened in the world")
        pkg = TickPackage(
            events=[info, make_inputtext_event()],
            active_npc_ids=[],
        )
        ctx = acc.ingest(pkg)
        assert len(ctx.world_events) == 1


# ---------------------------------------------------------------------------
# Buffer management
# ---------------------------------------------------------------------------

class TestBufferManagement:
    def test_events_accumulate_across_ticks(self):
        """Events from multiple data ticks accumulate before turn flush."""
        acc = EventAccumulator()
        # First data tick
        acc.ingest(TickPackage(
            events=[_speech_event("Lydia", "First")],
            active_npc_ids=["Lydia"],
        ))
        # Second data tick
        acc.ingest(TickPackage(
            events=[_speech_event("Lydia", "Second")],
            active_npc_ids=["Lydia"],
        ))
        # Player input flushes
        pkg = TickPackage(
            events=[make_inputtext_event()],
            active_npc_ids=["Lydia"],
        )
        ctx = acc.ingest(pkg)
        assert ctx is not None
        assert len(ctx.agent_buffers["Lydia"].events) == 2

    def test_flush_clears_events_but_keeps_buffers(self):
        acc = EventAccumulator()
        pkg = TickPackage(
            events=[_speech_event("Lydia"), make_inputtext_event()],
            active_npc_ids=["Lydia"],
        )
        acc.ingest(pkg)
        # After flush, buffer exists but events are cleared
        assert "Lydia" in acc._agent_buffers
        assert len(acc._agent_buffers["Lydia"].events) == 0

    def test_dialogue_history_persists_across_turns(self):
        acc = EventAccumulator()
        acc.record_agent_output("Lydia", "First response")
        # Trigger a turn
        pkg = make_turn_package("Second question")
        acc.ingest(pkg)
        # History should still be there
        assert len(acc._agent_buffers["Lydia"].dialogue_history) == 1
        assert acc._agent_buffers["Lydia"].dialogue_history[0]["content"] == "First response"


# ---------------------------------------------------------------------------
# Location tracking
# ---------------------------------------------------------------------------

class TestLocationTracking:
    def test_location_event_updates_current_location(self):
        acc = EventAccumulator()
        pkg = TickPackage(
            events=[_location_event("Dragonsreach")],
            active_npc_ids=[],
        )
        acc.ingest(pkg)
        assert acc.current_location == "Dragonsreach"

    def test_location_appears_in_world_events(self):
        acc = EventAccumulator()
        pkg = TickPackage(
            events=[_location_event("Dragonsreach"), make_inputtext_event()],
            active_npc_ids=[],
        )
        ctx = acc.ingest(pkg)
        assert any(e.event_type == "location" for e in ctx.world_events)


# ---------------------------------------------------------------------------
# Session reset
# ---------------------------------------------------------------------------

class TestSessionReset:
    def test_init_clears_agent_buffers(self):
        acc = EventAccumulator()
        acc._get_or_create_buffer("Lydia").append(_speech_event())
        pkg = TickPackage(events=[_init_event()], active_npc_ids=[])
        acc.ingest(pkg)
        assert len(acc._agent_buffers) == 0

    def test_init_resets_location(self):
        acc = EventAccumulator()
        acc.current_location = "Dragonsreach"
        pkg = TickPackage(events=[_init_event()], active_npc_ids=[])
        acc.ingest(pkg)
        assert acc.current_location == "Unknown"


# ---------------------------------------------------------------------------
# Dialogue history recording
# ---------------------------------------------------------------------------

class TestDialogueHistory:
    def test_record_agent_output(self):
        acc = EventAccumulator()
        acc.record_agent_output("Lydia", "I am sworn to carry your burdens.")
        buf = acc._agent_buffers["Lydia"]
        assert len(buf.dialogue_history) == 1
        assert buf.dialogue_history[0]["role"] == "assistant"

    def test_record_player_input(self):
        acc = EventAccumulator()
        acc._active_npc_ids = ["Lydia", "Belethor"]
        acc.record_player_input("Hello everyone!")
        for agent_id in ["Lydia", "Belethor"]:
            buf = acc._agent_buffers[agent_id]
            assert len(buf.dialogue_history) == 1
            assert buf.dialogue_history[0]["role"] == "user"
