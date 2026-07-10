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

    def test_record_player_input_fallback_no_speech_context(self):
        """Without _speech context, falls back to all active NPCs."""
        acc = EventAccumulator()
        acc._active_npc_ids = ["Lydia", "Belethor"]
        acc.record_player_input("Hello everyone!")
        for agent_id in ["Lydia", "Belethor"]:
            buf = acc._agent_buffers[agent_id]
            assert len(buf.dialogue_history) == 1
            assert buf.dialogue_history[0]["role"] == "user"
            assert buf.dialogue_history[0]["content"] == "Hello everyone!"

    def test_record_player_input_addressee_gets_direct(self):
        """Addressee (listener) gets player input as direct speech."""
        acc = EventAccumulator()
        acc._speech_earshot = {
            "speaker": "Player", "listener": "Lydia",
            "speech": "Hello", "companions": ["Faendal"],
        }
        acc.record_player_input("Can you carry this?")
        buf = acc._agent_buffers["Lydia"]
        assert len(buf.dialogue_history) == 1
        assert buf.dialogue_history[0]["content"] == "Can you carry this?"
        assert buf.dialogue_history[0]["role"] == "user"

    def test_record_player_input_companion_gets_overheard(self):
        """Companion in earshot gets overheard attribution."""
        acc = EventAccumulator()
        acc._speech_earshot = {
            "speaker": "Player", "listener": "Lydia",
            "speech": "Hello", "companions": ["Faendal"],
        }
        acc.record_player_input("Can you carry this?")
        buf = acc._agent_buffers["Faendal"]
        assert len(buf.dialogue_history) == 1
        assert "Player [to Lydia]" in buf.dialogue_history[0]["content"]
        assert "Can you carry this?" in buf.dialogue_history[0]["content"]

    def test_record_player_input_out_of_earshot_gets_nothing(self):
        """NPCs not in earshot get nothing — the forest doesn't remember."""
        acc = EventAccumulator()
        acc._active_npc_ids = ["Lydia", "Belethor", "Faendal"]
        acc._speech_earshot = {
            "speaker": "Player", "listener": "Lydia",
            "speech": "Hello", "companions": ["Faendal"],
        }
        acc.record_player_input("Secret message")
        # Belethor is active but not in earshot — no dialogue history
        assert "Belethor" not in acc._agent_buffers

    def test_record_player_input_player_listener_skipped(self):
        """If _speech listener is Player (echo), no buffer created."""
        acc = EventAccumulator()
        acc._speech_earshot = {
            "speaker": "Player", "listener": "Player",
            "speech": "Hello", "companions": [],
        }
        acc.record_player_input("Talking to myself")
        assert "Player" not in acc._agent_buffers

    def test_record_player_input_not_in_group_timeline(self):
        """Player input is earshot-filtered into per-NPC dialogue_history only.
        group_memory (shared_recent) is NPC-speech-only for cacheability.
        """
        acc = EventAccumulator()
        acc._speech_earshot = {
            "speaker": "Player", "listener": "Lydia",
            "speech": "Hello", "companions": [],
        }
        acc._get_or_create_buffer("Lydia")
        acc.record_player_input("Secret to Lydia")
        # Group timeline must NOT contain player input
        assert len(acc._group_memory.verbatim) == 0
        # But Lydia's private history does
        assert any(
            e["content"] == "Secret to Lydia"
            for e in acc._agent_buffers["Lydia"].dialogue_history
        )


# ---------------------------------------------------------------------------
# Group memory timeline
# ---------------------------------------------------------------------------

class TestPresenceChanges:
    def test_first_tick_all_entered(self):
        """On first tick, all NPCs are newcomers (entered from empty)."""
        acc = EventAccumulator()
        pkg = make_turn_package("Hello", active_npc_ids=["Lydia", "Belethor"])
        ctx = acc.ingest(pkg)
        assert sorted(ctx.presence_changes.entered) == ["Belethor", "Lydia"]
        assert ctx.presence_changes.exited == []

    def test_stable_group_no_changes(self):
        """Same NPCs across ticks → no presence changes."""
        acc = EventAccumulator()
        acc.ingest(TickPackage(
            events=[_speech_event("Lydia")],
            active_npc_ids=["Lydia"],
        ))
        pkg = make_turn_package("Hi", active_npc_ids=["Lydia"])
        ctx = acc.ingest(pkg)
        assert ctx.presence_changes.entered == []
        assert ctx.presence_changes.exited == []

    def test_newcomer_detected(self):
        """New NPC appearing in active_npc_ids → entered."""
        acc = EventAccumulator()
        acc.ingest(TickPackage(
            events=[_speech_event("Lydia")],
            active_npc_ids=["Lydia"],
        ))
        pkg = make_turn_package("Hi", active_npc_ids=["Lydia", "Belethor"])
        ctx = acc.ingest(pkg)
        assert ctx.presence_changes.entered == ["Belethor"]
        assert ctx.presence_changes.exited == []

    def test_departure_detected(self):
        """NPC leaving active_npc_ids → exited."""
        acc = EventAccumulator()
        acc.ingest(TickPackage(
            events=[_speech_event("Lydia")],
            active_npc_ids=["Lydia", "Belethor"],
        ))
        pkg = make_turn_package("Hi", active_npc_ids=["Lydia"])
        ctx = acc.ingest(pkg)
        assert ctx.presence_changes.entered == []
        assert ctx.presence_changes.exited == ["Belethor"]

    def test_session_reset_clears_prev(self):
        """After session reset, all NPCs are newcomers again."""
        acc = EventAccumulator()
        acc.ingest(TickPackage(
            events=[_speech_event("Lydia")],
            active_npc_ids=["Lydia"],
        ))
        acc.ingest(TickPackage(events=[_init_event()], active_npc_ids=[]))
        pkg = make_turn_package("Hi", active_npc_ids=["Lydia"])
        ctx = acc.ingest(pkg)
        assert ctx.presence_changes.entered == ["Lydia"]


class TestGroupMemory:
    def test_player_input_not_in_group_memory(self):
        """Player input is in per-NPC dialogue_history only, not group_memory.
        The group timeline (shared_recent) is NPC-speech-only for cacheability.
        """
        acc = EventAccumulator()
        acc._active_npc_ids = ["Lydia"]
        acc.record_player_input("Watch out!")
        assert len(acc._group_memory.verbatim) == 0
        # But it IS in Lydia's private dialogue_history (fallback: no speech ctx)
        lydia_history = acc._agent_buffers.get("Lydia", None)
        assert lydia_history is not None
        assert any(e["content"] == "Watch out!" for e in lydia_history.dialogue_history)

    def test_agent_output_recorded_in_group_memory(self):
        acc = EventAccumulator()
        acc.record_agent_output("Lydia", "I'll handle this!")
        assert len(acc._group_memory.verbatim) == 1
        assert acc._group_memory.verbatim[0]["role"] == "Lydia"
        assert acc._group_memory.verbatim[0]["content"] == "I'll handle this!"

    def test_group_memory_accumulates_across_turns(self):
        """Group timeline persists across turn flushes (NPC speech only)."""
        acc = EventAccumulator()
        acc._active_npc_ids = ["Lydia"]
        acc.record_player_input("Hello")   # NOT in group_memory
        acc.record_agent_output("Lydia", "Greetings.")  # IS in group_memory
        pkg = make_turn_package("How are you?", active_npc_ids=["Lydia"])
        ctx = acc.ingest(pkg)
        assert len(ctx.group_memory.verbatim) >= 1
        roles = [e["role"] for e in ctx.group_memory.verbatim]
        assert "Lydia" in roles
        assert "Player" not in roles  # Player input is NOT in group timeline

    def test_group_memory_in_turn_context(self):
        """TurnContext has group_memory attribute; player input alone leaves it empty."""
        acc = EventAccumulator()
        acc._active_npc_ids = ["Lydia"]
        acc.record_player_input("Tell me about Whiterun")  # NOT in group_memory
        acc.record_agent_output("Lydia", "It is the finest city.")  # IS in group_memory
        pkg = make_turn_package("What's your name?", active_npc_ids=["Lydia"])
        ctx = acc.ingest(pkg)
        assert hasattr(ctx, "group_memory")
        assert len(ctx.group_memory.verbatim) >= 1
        assert ctx.group_memory.verbatim[0]["role"] == "Lydia"

    def test_session_reset_clears_group_memory(self):
        acc = EventAccumulator()
        acc.record_agent_output("Lydia", "Hello.")
        assert len(acc._group_memory.verbatim) == 1
        pkg = TickPackage(events=[_init_event()], active_npc_ids=[])
        acc.ingest(pkg)
        assert len(acc._group_memory.verbatim) == 0

    def test_multiple_agents_all_appear_in_group_timeline(self):
        """All NPC speakers show up in the shared timeline; player does not."""
        acc = EventAccumulator()
        acc._active_npc_ids = ["Lydia", "Belethor"]
        acc.record_player_input("Hello everyone")   # NOT in group_memory
        acc.record_agent_output("Lydia", "Greetings.")
        acc.record_agent_output("Belethor", "Do come back.")
        roles = [e["role"] for e in acc._group_memory.verbatim]
        assert roles == ["Lydia", "Belethor"]
        assert "Player" not in roles


# ---------------------------------------------------------------------------
# Chat & speech dialogue adoption
# ---------------------------------------------------------------------------

def _chat_event(speaker: str = "Lydia", listener: str = "Faendal", speech: str = "Watch yourself.") -> TypedEvent:
    """Helper: create a chat event with parsed_data."""
    return TypedEvent(
        event_type="chat",
        local_ts="2024-01-01T00:00:00",
        game_ts=100.0,
        raw_data=f"{speaker}|{listener}|{speech}",
        parsed_data={"speaker": speaker, "listener": listener, "speech": speech},
    )


class TestChatAdoption:
    """Chat and _speech events: dual-agent routing + dialogue adoption."""

    def test_chat_routes_to_speaker_buffer(self):
        acc = EventAccumulator()
        chat = _chat_event("Lydia", "Faendal", "Watch yourself.")
        pkg = TickPackage(
            events=[chat, make_inputtext_event()],
            active_npc_ids=["Lydia", "Faendal"],
        )
        ctx = acc.ingest(pkg)
        assert ctx is not None
        assert "Lydia" in ctx.agent_buffers
        assert len(ctx.agent_buffers["Lydia"].events) == 1

    def test_chat_routes_to_listener_buffer(self):
        acc = EventAccumulator()
        chat = _chat_event("Lydia", "Faendal", "Watch yourself.")
        pkg = TickPackage(
            events=[chat, make_inputtext_event()],
            active_npc_ids=["Lydia", "Faendal"],
        )
        ctx = acc.ingest(pkg)
        assert ctx is not None
        assert "Faendal" in ctx.agent_buffers
        assert len(ctx.agent_buffers["Faendal"].events) == 1

    def test_chat_adopted_as_assistant_for_speaker(self):
        acc = EventAccumulator()
        chat = _chat_event("Lydia", "Faendal", "I am sworn to carry your burdens.")
        pkg = TickPackage(
            events=[chat, make_inputtext_event()],
            active_npc_ids=["Lydia", "Faendal"],
        )
        acc.ingest(pkg)
        buf = acc._agent_buffers["Lydia"]
        assert any(
            e["role"] == "assistant" and "sworn" in e["content"]
            for e in buf.dialogue_history
        )

    def test_chat_adopted_as_user_for_addressee(self):
        acc = EventAccumulator()
        chat = _chat_event("Lydia", "Faendal", "Move along.")
        pkg = TickPackage(
            events=[chat, make_inputtext_event()],
            active_npc_ids=["Lydia", "Faendal"],
        )
        acc.ingest(pkg)
        buf = acc._agent_buffers["Faendal"]
        assert any(
            e["role"] == "user" and "Lydia [to you]" in e["content"] and "Move along" in e["content"]
            for e in buf.dialogue_history
        )

    def test_bystander_does_not_get_dialogue_history(self):
        """Active NPC not in speaker/listener role has empty dialogue_history."""
        acc = EventAccumulator()
        acc._get_or_create_buffer("Belethor")
        chat = _chat_event("Lydia", "Faendal", "Watch yourself.")
        pkg = TickPackage(
            events=[chat, make_inputtext_event()],
            active_npc_ids=["Lydia", "Faendal", "Belethor"],
        )
        acc.ingest(pkg)
        assert len(acc._agent_buffers["Belethor"].dialogue_history) == 0

    def test_speech_routes_to_listener(self):
        """_speech events now also route to the listener's buffer."""
        acc = EventAccumulator()
        speech = _speech_event("Lydia", "I am sworn to carry your burdens.")
        speech.parsed_data["listener"] = "Faendal"
        pkg = TickPackage(
            events=[speech, make_inputtext_event()],
            active_npc_ids=["Lydia", "Faendal"],
        )
        ctx = acc.ingest(pkg)
        assert "Faendal" in ctx.agent_buffers
        assert len(ctx.agent_buffers["Faendal"].events) == 1

    def test_unrecognized_speaker_falls_to_world(self):
        """Chat with empty speaker goes to world events."""
        acc = EventAccumulator()
        chat = TypedEvent(
            event_type="chat",
            local_ts="2024-01-01T00:00:00",
            game_ts=100.0,
            raw_data="just some chatter",
            parsed_data={"speaker": "", "listener": "", "speech": "just some chatter"},
        )
        pkg = TickPackage(
            events=[chat, make_inputtext_event()],
            active_npc_ids=[],
        )
        ctx = acc.ingest(pkg)
        assert len(ctx.world_events) == 1

    def test_chat_with_player_listener_skips_player_buffer(self):
        """Player is not an NPC agent — no buffer created for Player."""
        acc = EventAccumulator()
        chat = _chat_event("Lydia", "Player", "Follow me.")
        pkg = TickPackage(
            events=[chat, make_inputtext_event()],
            active_npc_ids=["Lydia"],
        )
        acc.ingest(pkg)
        assert "Player" not in acc._agent_buffers
        assert "Lydia" in acc._agent_buffers


# ---------------------------------------------------------------------------
# Earshot bitvector tracking
# ---------------------------------------------------------------------------

class TestEarshotBitvector:
    def test_speech_updates_earshot_bits(self):
        """After _speech with companions, speaker + companions have reciprocal bits."""
        acc = EventAccumulator()
        speech = TypedEvent(
            event_type="_speech",
            local_ts="2024-01-01T00:00:00",
            game_ts=100.0,
            raw_data="Hello",
            parsed_data={
                "speaker": "Lydia", "listener": "Player",
                "speech": "Hello", "location": "Whiterun",
                "companions": ["Faendal", "Belethor"], "distance": 5.0,
            },
        )
        pkg = TickPackage(
            events=[speech],
            active_npc_ids=["Lydia", "Faendal", "Belethor"],
        )
        acc.ingest(pkg)
        lydia_bits = acc._agent_buffers["Lydia"].earshot_bits
        faendal_bits = acc._agent_buffers["Faendal"].earshot_bits
        belethor_bits = acc._agent_buffers["Belethor"].earshot_bits
        lydia_bit = acc._earshot_index.mask_for("Lydia")
        faendal_bit = acc._earshot_index.mask_for("Faendal")
        assert faendal_bits & lydia_bit
        assert lydia_bits & faendal_bit
        assert belethor_bits & lydia_bit

    def test_coarse_earshot_from_active_npcs(self):
        """Active NPCs with existing buffers get coarse earshot bits."""
        acc = EventAccumulator()
        acc._get_or_create_buffer("Lydia")
        acc._get_or_create_buffer("Belethor")
        pkg = TickPackage(
            events=[],
            active_npc_ids=["Lydia", "Belethor", "Faendal"],
        )
        acc.ingest(pkg)
        lydia_bits = acc._agent_buffers["Lydia"].earshot_bits
        belethor_bit = acc._earshot_index.mask_for("Belethor")
        assert lydia_bits & belethor_bit
        assert "Faendal" not in acc._agent_buffers

    def test_earshot_cleared_on_reset(self):
        """Session reset clears earshot index and agent buffers."""
        acc = EventAccumulator()
        acc._get_or_create_buffer("Lydia").earshot_bits = 0xFF
        acc._earshot_index.get_or_assign("Lydia")
        pkg = TickPackage(events=[_init_event()], active_npc_ids=[])
        acc.ingest(pkg)
        assert len(acc._agent_buffers) == 0
        assert acc._earshot_index.get("Lydia") is None
