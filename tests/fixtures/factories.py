"""
Test data factories for the Many-Mind Kernel.

Each factory returns a realistic SKSE wire-format string, TypedEvent, or
TickPackage for a specific scenario. Used by wire_protocol and round-trip tests.
"""
from __future__ import annotations

from datetime import datetime, timezone
from uuid import uuid4

from shared.schemas import (
    TypedEvent, TickPackage,
    NpcMetadata, ActorValues,
    TurnResponse, AgentResponse, ActorValueDeltas, ActionCommand,
    ExtractionLevel,
)


# ---------------------------------------------------------------------------
# Raw SKSE wire strings (what AIAgent.dll actually sends)
# ---------------------------------------------------------------------------

# Calm conversation — player talks to Lydia in Whiterun
WIRE_INPUTTEXT = "inputtext|1710624000|54321.0|What do you think about the civil war?"

# Speech input variant
WIRE_INPUTTEXT_S = "inputtext_s|1710624001|54321.5|I need your help with something"

# Game state events (NPC info, environment, location)
WIRE_INFO = "info|1710624002|54322.0|Lydia drew her weapon"
WIRE_INFONPC = "infonpc|1710624003|54323.0|Lydia|health:100|stamina:80"
WIRE_INFOLOC = "infoloc|1710624004|54324.0|WhiterunExterior|clear|afternoon"
WIRE_LOCATION = "location|1710624005|54325.0|WhiterunExterior"

# Narrative events
WIRE_CHAT = "chat|1710624006|54326.0|Belethor|Lydia|Do come back"
WIRE_DEATH = "death|1710624007|54327.0|Bandit was killed by Lydia"
WIRE_QUEST = "quest|1710624008|54328.0|BQ01|Started"
WIRE_BOOK = "book|1710624009|54329.0|The Book of the Dragonborn|Ancient tales..."

# Control events
WIRE_REQUEST = "request|1710624010|54330.0|"
WIRE_REQUEST_EMPTY = "request|1710624011|54331.0|"

# Session events
WIRE_GOODNIGHT = "goodnight|1710624012|54332.0|"

# Function return
WIRE_FUNCRET = "funcret|1710624013|54333.0|Follow|success"

# Error case
WIRE_CHATNF = "chatnf|1710624014|54334.0|NonExistentNPC"

# Edge cases
WIRE_MALFORMED_SHORT = "info|123"
WIRE_MALFORMED_EMPTY = ""
WIRE_MALFORMED_BAD_TS = "info|123|not_a_number|some data"
WIRE_DATA_WITH_PIPES = "info|1710624015|54335.0|Lydia says|she wants|to help"


# ---------------------------------------------------------------------------
# TypedEvent factories
# ---------------------------------------------------------------------------

def make_inputtext_event(
    text: str = "What do you think about the civil war?",
    game_ts: float = 54321.0,
) -> TypedEvent:
    """Create an inputtext TypedEvent (player input)."""
    return TypedEvent(
        event_type="inputtext",
        local_ts=datetime.now(timezone.utc).isoformat(),
        game_ts=game_ts,
        raw_data=text,
        parsed_data=None,
    )


def make_info_event(
    data: str = "Lydia drew her weapon",
    game_ts: float = 54322.0,
) -> TypedEvent:
    """Create an info TypedEvent (data event)."""
    return TypedEvent(
        event_type="info",
        local_ts=datetime.now(timezone.utc).isoformat(),
        game_ts=game_ts,
        raw_data=data,
        parsed_data=None,
    )


# ---------------------------------------------------------------------------
# TickPackage factories
# ---------------------------------------------------------------------------

def make_turn_package(
    input_text: str = "What do you think about the civil war?",
    active_npc_ids: list[str] | None = None,
    game_ts: float = 54321.0,
) -> TickPackage:
    """Create a TickPackage with player input."""
    if active_npc_ids is None:
        active_npc_ids = ["Lydia"]
    event = make_inputtext_event(input_text, game_ts)
    return TickPackage(
        events=[event],
        active_npc_ids=active_npc_ids,
        tick_interval_ms=2000,
    )


def make_data_package(
    event_type: str = "info",
    data: str = "Lydia drew her weapon",
    game_ts: float = 54322.0,
) -> TickPackage:
    """Create a data-only TickPackage (no player input)."""
    event = TypedEvent(
        event_type=event_type,
        local_ts=datetime.now(timezone.utc).isoformat(),
        game_ts=game_ts,
        raw_data=data,
        parsed_data=None,
    )
    return TickPackage(
        events=[event],
        active_npc_ids=[],
        tick_interval_ms=2000,
    )


def make_multi_npc_package() -> TickPackage:
    """Create a turn TickPackage with multiple NPCs (Whiterun market walk)."""
    return make_turn_package(
        input_text="Hello everyone!",
        active_npc_ids=["Lydia", "Belethor", "Ysolda", "Heimskr"],
    )


# Backward-compat aliases
make_turn_payload = make_turn_package
make_data_payload = make_data_package
make_multi_npc_payload = make_multi_npc_package


# ---------------------------------------------------------------------------
# Expected TurnResponse factories
# ---------------------------------------------------------------------------

def make_lydia_response() -> TurnResponse:
    """Expected stub response for Lydia."""
    return TurnResponse(
        tick_id=uuid4(),
        responses=[
            AgentResponse(
                agent_id="Lydia",
                utterance="I am sworn to carry your burdens.",
                actor_value_deltas=ActorValueDeltas(Confidence=3, Mood=0, Assistance=2),
                actions=[ActionCommand(command="Follow", target="Player")],
                extraction_level=ExtractionLevel.STRICT,
            ),
        ],
        processing_time_ms=0,
        model_used="stub-canned",
    )


# ---------------------------------------------------------------------------
# Expected CHIM wire output
# ---------------------------------------------------------------------------

# What Falcon should return to SKSE for Lydia's stub response
EXPECTED_WIRE_LYDIA = (
    "Lydia|dialogue|I am sworn to carry your burdens.\r\n"
    "Lydia|command|Follow@Player\r\n"
)

EXPECTED_WIRE_MULTI_NPC = (
    "Lydia|dialogue|I am sworn to carry your burdens.\r\n"
    "Lydia|command|Follow@Player\r\n"
    "Belethor|dialogue|Do come back.\r\n"
    "Ysolda|dialogue|I've been thinking about trading with the Khajiit caravans.\r\n"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _lydia_metadata() -> NpcMetadata:
    """NpcMetadata for Lydia — kept for potential direct use in tests."""
    return NpcMetadata(
        position=[100.5, 200.3, 50.0],
        cell="WhiterunExterior",
        level=25,
        hp=100.0, mp=50.0, sp=80.0,
        equipment=["Steel Sword", "Iron Shield", "Steel Armor"],
        actor_values=ActorValues(Aggression=1, Confidence=2, Morality=3, Mood=0, Assistance=2),
        is_follower=True,
    )
