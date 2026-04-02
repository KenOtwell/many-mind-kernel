"""
SKSE wire protocol translation.

Parse inbound: type|localts|gamets|data → ParsedEvent
Format outbound: AgentResponse → NPCName|DialogueType|Text\r\n
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from shared.constants import (
    FALCON_LOCAL_TYPES, SESSION_TYPES,
    COMMAND_VOCABULARY, WIRE_LINE_ENDING, WIRE_ACTION_PARAM_SEPARATOR,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Inbound: SKSE → Falcon
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ParsedEvent:
    """A parsed SKSE inbound event."""
    event_type: str
    local_ts: str
    game_ts: float
    data: str
    is_local: bool
    is_session: bool

    @property
    def needs_forwarding(self) -> bool:
        """Whether this event should be forwarded to Progeny."""
        return not self.is_local


def parse_event(raw_body: str) -> Optional[ParsedEvent]:
    """
    Parse an SKSE inbound event from pipe-delimited wire format.

    Format: type|localts|gamets|data
    Returns None if malformed (never raises).
    """
    stripped = raw_body.strip()
    if not stripped:
        logger.warning("Empty event body received")
        return None

    # Split on first 3 pipes — data field may contain pipes
    parts = stripped.split("|", 3)
    if len(parts) < 3:
        logger.warning("Malformed event (need ≥3 pipe fields): %.100s", stripped)
        return None

    event_type = parts[0].lower().strip()
    local_ts = parts[1].strip()
    game_ts_str = parts[2].strip()
    data = parts[3] if len(parts) > 3 else ""

    try:
        game_ts = float(game_ts_str)
    except ValueError:
        logger.warning("Invalid game_ts '%s', defaulting to 0.0", game_ts_str)
        game_ts = 0.0

    return ParsedEvent(
        event_type=event_type,
        local_ts=local_ts,
        game_ts=game_ts,
        data=data,
        is_local=event_type in FALCON_LOCAL_TYPES,
        is_session=event_type in SESSION_TYPES,
    )


# ---------------------------------------------------------------------------
# Outbound: Falcon → SKSE
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class WireResponse:
    """A single CHIM wire protocol response line."""
    npc_name: str
    response_type: str  # "dialogue" or "command"
    content: str

    def format(self) -> str:
        """Render as CHIM wire line: NPCName|type|content\\r\\n"""
        return f"{self.npc_name}|{self.response_type}|{self.content}{WIRE_LINE_ENDING}"


def format_dialogue(npc_name: str, text: str) -> WireResponse:
    """Format a dialogue line for SKSE."""
    return WireResponse(npc_name=npc_name, response_type="dialogue", content=text)


def format_action(npc_name: str, command: str,
                   target: str = "", item: str = "") -> WireResponse:
    """Format an action command line for SKSE."""
    # Build ActionName@Params string
    params_parts = [p for p in (target, item) if p]
    params = WIRE_ACTION_PARAM_SEPARATOR.join(params_parts)
    action_str = f"{command}{WIRE_ACTION_PARAM_SEPARATOR}{params}" if params else command

    return WireResponse(npc_name=npc_name, response_type="command", content=action_str)


def format_agent_responses(
    agent_id: str,
    utterance: Optional[str],
    actions: list[dict],
    actor_value_deltas: Optional[dict] = None,
) -> list[WireResponse]:
    """
    Convert one agent's response into CHIM wire lines.

    Order: dialogue, then explicit actions, then actor value deltas.
    Actor value deltas are emitted as SetBehavior commands:
        NPCName|command|SetBehavior@Aggression@2
    These are handled in-game by MMKSetBehavior.psc via CHIM_CommandReceived.
    Unknown commands are silently dropped.
    """
    lines: list[WireResponse] = []

    if utterance:
        lines.append(format_dialogue(agent_id, utterance))

    for action in actions:
        cmd = action.get("command", "")
        if cmd not in COMMAND_VOCABULARY:
            logger.warning("Dropping unknown command '%s' for %s", cmd, agent_id)
            continue
        lines.append(format_action(
            agent_id, cmd,
            target=action.get("target", "") or "",
            item=action.get("item", "") or "",
        ))

    # Actor value deltas — emit as SetBehavior@ValueName@NewValue
    # The DLL forwards these via CHIM_CommandReceived to MMKSetBehavior.psc
    if actor_value_deltas:
        for value_name, new_value in actor_value_deltas.items():
            if new_value is not None:
                lines.append(format_action(
                    agent_id, "SetBehavior",
                    target=value_name,
                    item=str(new_value),
                ))

    return lines


def format_turn_response(responses: list[dict]) -> str:
    """
    Format a complete TurnResponse into a CHIM wire protocol string.

    Input: list of AgentResponse dicts (from TurnResponse.responses).
    Output: multi-line string ready to return to SKSE plugin.
    Includes actor_value_deltas as SetBehavior commands.
    """
    all_lines: list[WireResponse] = []
    for resp in responses:
        agent_id = resp.get("agent_id", "Unknown")
        utterance = resp.get("utterance")
        actions = resp.get("actions", [])
        # actor_value_deltas: {"Aggression": 2, "Confidence": 3, ...} (None values skipped)
        avd_raw = resp.get("actor_value_deltas") or {}
        avd = {k: v for k, v in avd_raw.items() if v is not None}
        all_lines.extend(format_agent_responses(agent_id, utterance, actions, avd))

    return "".join(line.format() for line in all_lines)
