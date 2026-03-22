"""
Tests for falcon.src.wire_protocol — parse SKSE events, format CHIM responses.
"""
import pytest
from falcon.src.wire_protocol import (
    parse_event, format_dialogue, format_action,
    format_agent_responses, format_turn_response,
)
from tests.fixtures.factories import (
    WIRE_INPUTTEXT, WIRE_INPUTTEXT_S, WIRE_INFO, WIRE_INFONPC,
    WIRE_REQUEST, WIRE_GOODNIGHT, WIRE_CHATNF, WIRE_FUNCRET,
    WIRE_CHAT, WIRE_DEATH, WIRE_BOOK,
    WIRE_MALFORMED_SHORT, WIRE_MALFORMED_EMPTY, WIRE_MALFORMED_BAD_TS,
    WIRE_DATA_WITH_PIPES,
)


# ---------------------------------------------------------------------------
# Parsing: inbound SKSE events
# ---------------------------------------------------------------------------

class TestParseEvent:
    """Test parse_event with various SKSE wire format strings."""

    def test_inputtext_is_turn_trigger(self):
        """Player typed input should be flagged as turn trigger."""
        parsed = parse_event(WIRE_INPUTTEXT)
        assert parsed is not None
        assert parsed.event_type == "inputtext"
        assert parsed.is_turn_trigger is True
        assert parsed.is_local is False
        assert parsed.needs_forwarding is True
        assert parsed.game_ts == 54321.0
        assert "civil war" in parsed.data

    def test_inputtext_s_is_turn_trigger(self):
        """Speech input should also be a turn trigger."""
        parsed = parse_event(WIRE_INPUTTEXT_S)
        assert parsed is not None
        assert parsed.event_type == "inputtext_s"
        assert parsed.is_turn_trigger is True

    def test_info_is_data_event(self):
        """Info event should NOT be a turn trigger."""
        parsed = parse_event(WIRE_INFO)
        assert parsed is not None
        assert parsed.event_type == "info"
        assert parsed.is_turn_trigger is False
        assert parsed.needs_forwarding is True

    def test_request_is_local(self):
        """Request (polling) should be Falcon-local, not forwarded."""
        parsed = parse_event(WIRE_REQUEST)
        assert parsed is not None
        assert parsed.event_type == "request"
        assert parsed.is_local is True
        assert parsed.needs_forwarding is False

    def test_chatnf_is_local(self):
        """Chat-not-found should be Falcon-local."""
        parsed = parse_event(WIRE_CHATNF)
        assert parsed is not None
        assert parsed.is_local is True

    def test_goodnight_is_session(self):
        """Goodnight should be flagged as session event."""
        parsed = parse_event(WIRE_GOODNIGHT)
        assert parsed is not None
        assert parsed.is_session is True

    def test_funcret_forwarded(self):
        """Function return should be forwarded to Progeny."""
        parsed = parse_event(WIRE_FUNCRET)
        assert parsed is not None
        assert parsed.event_type == "funcret"
        assert parsed.needs_forwarding is True

    def test_data_with_pipes_preserved(self):
        """Data field containing pipes should be preserved intact."""
        parsed = parse_event(WIRE_DATA_WITH_PIPES)
        assert parsed is not None
        # Data is everything after the 3rd pipe
        assert "Lydia says|she wants|to help" == parsed.data

    def test_chat_event(self):
        parsed = parse_event(WIRE_CHAT)
        assert parsed is not None
        assert parsed.event_type == "chat"
        assert "Belethor" in parsed.data

    def test_death_event(self):
        parsed = parse_event(WIRE_DEATH)
        assert parsed is not None
        assert parsed.event_type == "death"

    def test_book_event(self):
        parsed = parse_event(WIRE_BOOK)
        assert parsed is not None
        assert parsed.event_type == "book"

    # --- Edge cases ---

    def test_empty_body_returns_none(self):
        assert parse_event(WIRE_MALFORMED_EMPTY) is None

    def test_too_few_fields_returns_none(self):
        assert parse_event(WIRE_MALFORMED_SHORT) is None

    def test_bad_timestamp_defaults_to_zero(self):
        """Invalid game_ts should default to 0.0, not crash."""
        parsed = parse_event(WIRE_MALFORMED_BAD_TS)
        assert parsed is not None
        assert parsed.game_ts == 0.0
        assert parsed.data == "some data"

    def test_whitespace_stripped(self):
        parsed = parse_event("  inputtext | 123 | 456.0 | hello  ")
        assert parsed is not None
        assert parsed.event_type == "inputtext"
        assert parsed.game_ts == 456.0

    def test_case_insensitive_type(self):
        parsed = parse_event("INPUTTEXT|123|456.0|hello")
        assert parsed is not None
        assert parsed.event_type == "inputtext"
        assert parsed.is_turn_trigger is True


# ---------------------------------------------------------------------------
# Formatting: outbound CHIM responses
# ---------------------------------------------------------------------------

class TestFormatDialogue:
    def test_basic_dialogue(self):
        resp = format_dialogue("Lydia", "I am sworn to carry your burdens.")
        assert resp.format() == "Lydia|dialogue|I am sworn to carry your burdens.\r\n"

    def test_empty_text(self):
        resp = format_dialogue("Lydia", "")
        assert resp.format() == "Lydia|dialogue|\r\n"


class TestFormatAction:
    def test_action_with_target(self):
        resp = format_action("Lydia", "Follow", target="Player")
        assert resp.format() == "Lydia|command|Follow@Player\r\n"

    def test_action_with_target_and_item(self):
        resp = format_action("Lydia", "GiveItemTo", target="Player", item="Health Potion")
        assert resp.format() == "Lydia|command|GiveItemTo@Player@Health Potion\r\n"

    def test_action_no_params(self):
        resp = format_action("Lydia", "SheatheWeapon")
        assert resp.format() == "Lydia|command|SheatheWeapon\r\n"

    def test_action_item_only(self):
        resp = format_action("Lydia", "CastSpell", item="Fireball")
        assert resp.format() == "Lydia|command|CastSpell@Fireball\r\n"


class TestFormatAgentResponses:
    def test_dialogue_and_action(self):
        lines = format_agent_responses(
            "Lydia",
            utterance="Stay back!",
            actions=[{"command": "Attack", "target": "Bandit"}],
        )
        assert len(lines) == 2
        assert lines[0].response_type == "dialogue"
        assert lines[1].response_type == "command"

    def test_unknown_command_dropped(self):
        """Unknown commands should be silently dropped."""
        lines = format_agent_responses(
            "Lydia",
            utterance="Hmm.",
            actions=[{"command": "FlyToMoon", "target": "Moon"}],
        )
        assert len(lines) == 1  # Only the dialogue, action dropped

    def test_no_utterance_no_actions(self):
        lines = format_agent_responses("Heimskr", utterance=None, actions=[])
        assert len(lines) == 0


class TestFormatTurnResponse:
    def test_multi_agent_response(self):
        responses = [
            {
                "agent_id": "Lydia",
                "utterance": "I am sworn to carry your burdens.",
                "actions": [{"command": "Follow", "target": "Player"}],
            },
            {
                "agent_id": "Belethor",
                "utterance": "Do come back.",
                "actions": [],
            },
        ]
        wire = format_turn_response(responses)
        assert "Lydia|dialogue|I am sworn to carry your burdens.\r\n" in wire
        assert "Lydia|command|Follow@Player\r\n" in wire
        assert "Belethor|dialogue|Do come back.\r\n" in wire

    def test_empty_responses(self):
        assert format_turn_response([]) == ""

    def test_agent_with_no_output(self):
        """Agent with no utterance and no actions produces no wire output."""
        responses = [{"agent_id": "Heimskr", "utterance": None, "actions": []}]
        assert format_turn_response(responses) == ""
