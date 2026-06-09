"""
Event accumulator for Progeny.

Ingests TypedEvents from Falcon's TickPackages, maintains per-agent event
buffers across turns, detects player input (inputtext/inputtext_s),
and flushes accumulated context for prompt building.

Player input detection is Progeny's autonomous cognitive concern —
Falcon ships all events as pure data with no turn-coupling flags.
Progeny decides when to respond based on accumulated state.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional

from shared.constants import PLAYER_INPUT_TYPES, SESSION_TYPES
from shared.schemas import TickPackage, TypedEvent

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from progeny.src.fact_pool import FactPool

logger = logging.getLogger(__name__)


@dataclass
class TieredMemory:
    """Three-tier sliding window for agent memory.

    Verbatim  (newest): full dialogue entries, perfect recall.
    Compressed (middle): one-line structural summaries.
    Keywords   (oldest): pipe-delimited semantic tags, retrieval anchors.
    """
    verbatim: list[dict] = field(default_factory=list)    # max 8
    compressed: list[str] = field(default_factory=list)    # max 10
    keywords: list[str] = field(default_factory=list)      # max 10


@dataclass
class AgentBuffer:
    """Per-agent event buffer across turns."""
    agent_id: str
    events: list[TypedEvent] = field(default_factory=list)
    memory: TieredMemory = field(default_factory=TieredMemory)
    active_task: str = ""

    @property
    def dialogue_history(self) -> list[dict]:
        """Backward-compatible access — reads from memory.verbatim."""
        return self.memory.verbatim

    @dialogue_history.setter
    def dialogue_history(self, value: list[dict]) -> None:
        """Backward-compatible setter — writes to memory.verbatim."""
        self.memory.verbatim = value

    def append(self, event: TypedEvent) -> None:
        self.events.append(event)

    def clear(self) -> None:
        self.events.clear()


@dataclass
class PresenceChanges:
    """NPCs who entered or exited the scene since the last tick.

    Used by the recognition bootstrap: when an NPC enters, existing
    agents fire referent-filtered retrieval against the newcomer's ID.
    Results queue as one-tick-delayed remindings (private, Layer 2).
    """
    entered: list[str] = field(default_factory=list)
    exited: list[str] = field(default_factory=list)


@dataclass
class TurnContext:
    """Accumulated context for one turn, ready for prompt building."""
    player_input: str
    agent_buffers: dict[str, AgentBuffer]
    active_npc_ids: list[str]
    world_events: list[TypedEvent]
    session_events: list[TypedEvent]
    group_memory: TieredMemory = field(default_factory=TieredMemory)
    presence_changes: PresenceChanges = field(default_factory=PresenceChanges)


class EventAccumulator:
    """
    Accumulates typed events from Falcon's TickPackages.

    Maintains per-agent buffers, tracks world state, and detects player
    input. When player input arrives, flush_turn() returns a TurnContext
    with everything the prompt builder needs.
    """

    def __init__(self, fact_pool: "FactPool | None" = None) -> None:
        # Per-agent event buffers — persist across ticks until flushed on turn
        self._agent_buffers: dict[str, AgentBuffer] = {}
        # World/info events accumulated between turns
        self._world_events: list[TypedEvent] = []
        # Session lifecycle events
        self._session_events: list[TypedEvent] = []
        # Current location (updated from location events)
        self.current_location: str = "Unknown"
        # Last player input (from most recent player input event)
        self._pending_player_input: Optional[str] = None
        # Active NPC IDs from latest tick
        self._active_npc_ids: list[str] = []
        # Previous tick's active NPCs — for presence-change detection
        self._prev_active_npc_ids: set[str] = set()
        # ATMS fact pool — bitvector-tagged world knowledge
        self._fact_pool = fact_pool
        # Group-level shared timeline — the canonical record of "what happened"
        # that all present participants share. Condenses through the same
        # TieredMemory pipeline as personal history. Individual NPCs carry
        # their own emotional signatures; the group memory carries the facts.
        self._group_memory = TieredMemory()

    def ingest(self, package: TickPackage) -> Optional[TurnContext]:
        """
        Ingest a TickPackage from Falcon.

        Routes each event to the appropriate buffer based on type.
        Tracks presence changes (entered/exited NPCs) for the recognition
        bootstrap. Returns a TurnContext if player input was detected.
        """
        # Detect presence changes before updating active IDs
        current_set = set(package.active_npc_ids)
        self._presence_entered = sorted(current_set - self._prev_active_npc_ids)
        self._presence_exited = sorted(self._prev_active_npc_ids - current_set)
        self._prev_active_npc_ids = current_set

        self._active_npc_ids = package.active_npc_ids
        has_player_input = False

        # Present NPCs for fact propagation (player + all active)
        present_ids = ["Player"] + list(package.active_npc_ids)

        # Register new NPCs in fact pool and give them lore
        if self._fact_pool is not None:
            for npc_id in package.active_npc_ids:
                if self._fact_pool.bit_index.get(npc_id) is None:
                    self._fact_pool.bit_index.get_or_assign(npc_id)
                    self._fact_pool.ensure_lore_bits(npc_id)

        for event in package.events:
            event_type = event.event_type

            # Player input detection (Progeny's autonomous decision)
            if event_type in PLAYER_INPUT_TYPES:
                has_player_input = True
                self._pending_player_input = event.raw_data
                continue

            # Session lifecycle
            if event_type in SESSION_TYPES:
                self._session_events.append(event)
                if event_type in ("init", "wipe"):
                    self._handle_reset()
                continue

            # Location tracking
            if event_type == "location":
                self.current_location = event.raw_data
                self._world_events.append(event)
                self._record_fact(event, present_ids)
                continue

            # Route to agent buffer based on event type
            agent_id = self._extract_agent_id(event)
            if agent_id:
                self._get_or_create_buffer(agent_id).append(event)
            else:
                # World/info events without a clear agent owner
                self._world_events.append(event)

            # Behavior adoption: canned NPC _speech is recorded into the
            # speaker's dialogue history the same way LLM-generated output
            # is, so on the next turn the agent sees "I said this" instead
            # of having a silent gap where vanilla dialogue happened. Only
            # fires when the speaker is an active NPC (player speech goes
            # through record_player_input from inputtext events).
            if (event.event_type == "_speech"
                    and event.parsed_data
                    and agent_id
                    and agent_id in self._active_npc_ids):
                speech_text = (event.parsed_data.get("speech") or "").strip()
                if speech_text:
                    self.record_npc_speech(agent_id, speech_text)

            # Record fact for all significant events
            self._record_fact(event, present_ids)

        # If player input detected, flush and return context
        if has_player_input and self._pending_player_input is not None:
            return self.flush_turn()
        return None

    def flush_turn(self) -> TurnContext:
        """
        Flush accumulated state and return a TurnContext for prompt building.

        Snapshots per-agent event buffers (copies event lists) so the returned
        TurnContext is independent of live state. Dialogue history and group
        memory persist across turns (they're the cross-turn timeline).
        """
        player_input = self._pending_player_input or ""
        # Snapshot: copy each buffer's events so clearing doesn't affect the context
        snapshot_buffers: dict[str, AgentBuffer] = {}
        for agent_id, buf in self._agent_buffers.items():
            snap = AgentBuffer(agent_id=agent_id)
            snap.events = list(buf.events)
            snap.dialogue_history = buf.dialogue_history  # Shared ref is OK — persists
            snapshot_buffers[agent_id] = snap

        context = TurnContext(
            player_input=player_input,
            agent_buffers=snapshot_buffers,
            active_npc_ids=list(self._active_npc_ids),
            world_events=list(self._world_events),
            session_events=list(self._session_events),
            group_memory=self._group_memory,  # Shared ref — persists across turns
            presence_changes=PresenceChanges(
                entered=self._presence_entered,
                exited=self._presence_exited,
            ),
        )
        # Clear tick-level buffers; agent buffers persist structure but clear events
        for buf in self._agent_buffers.values():
            buf.clear()
        self._world_events.clear()
        self._session_events.clear()
        self._pending_player_input = None
        return context

    def record_agent_output(self, agent_id: str, utterance: str) -> None:
        """
        Record LLM-generated output into agent's dialogue history and
        the shared group timeline.

        Behavior adoption: adopted as the agent's own output (role=assistant).
        On the next turn, the agent sees this as something it said.
        Group timeline: recorded as shared event (everyone present heard it).
        """
        buf = self._get_or_create_buffer(agent_id)
        buf.dialogue_history.append({"role": "assistant", "content": utterance})
        # Group timeline — everyone present heard this NPC speak
        self._group_memory.verbatim.append(
            {"role": agent_id, "content": utterance}
        )

    def record_npc_speech(self, agent_id: str, text: str) -> None:
        """
        Record canned NPC speech (from _speech events) into dialogue history.

        Per the Behavior Adoption design (Living Doc §Event Accumulator):
        externally-generated NPC dialogue — CHIM canned lines, RDO replies,
        vanilla ambient — is adopted as the agent's own output. The agent
        cannot distinguish "I said this" from "the game made me say this";
        on the next turn it sees the line in its history and rationalises
        continuity from there.

        Same shape as record_agent_output: role=assistant in the speaker's
        private history, role=<agent_id> in the group timeline. Text is
        the clean speech field from SpeechData, not the JSON envelope.
        """
        buf = self._get_or_create_buffer(agent_id)
        buf.dialogue_history.append({"role": "assistant", "content": text})
        self._group_memory.verbatim.append(
            {"role": agent_id, "content": text}
        )

    def record_player_input(self, text: str) -> None:
        """Record player input into dialogue history and group timeline."""
        # Player input goes into all active agent buffers
        for agent_id in self._active_npc_ids:
            buf = self._get_or_create_buffer(agent_id)
            buf.dialogue_history.append({"role": "user", "content": text})
        # Group timeline — everyone present heard the player
        self._group_memory.verbatim.append(
            {"role": "Player", "content": text}
        )

    def _extract_agent_id(self, event: TypedEvent) -> Optional[str]:
        """
        Extract the agent (NPC) this event is about from parsed_data.

        Uses structural data from Falcon's parsers — no semantic interpretation.
        """
        parsed = event.parsed_data
        if parsed is None:
            return None

        event_type = event.event_type

        # Speech events — speaker is the agent
        if event_type == "_speech" and "speaker" in parsed:
            return parsed["speaker"]

        # NPC registration (prefix match — DLL may send addnpc variants)
        if event_type.startswith("addnpc") and "name" in parsed:
            return parsed["name"]

        # Stats update
        if event_type == "updatestats" and "npc_name" in parsed:
            return parsed["npc_name"]

        # Item transfer — source is the acting agent
        if event_type == "itemtransfer" and "source" in parsed:
            return parsed["source"]

        return None

    def _get_or_create_buffer(self, agent_id: str) -> AgentBuffer:
        """Get or create an agent buffer."""
        if agent_id not in self._agent_buffers:
            self._agent_buffers[agent_id] = AgentBuffer(agent_id=agent_id)
        return self._agent_buffers[agent_id]

    def _record_fact(self, event: TypedEvent, present_ids: list[str]) -> None:
        """Create a fact from an event and set knowledge bits for present NPCs.

        Speech events additionally propagate to companions in earshot.
        Location events supersede the previous location fact.
        """
        if self._fact_pool is None:
            return

        # For _speech, prefer the parsed speech text — storing the raw
        # JSON envelope (audios path, companions list, etc.) as a fact
        # would pollute retrieval with non-utterance noise.
        if event.event_type == "_speech" and event.parsed_data:
            content = (event.parsed_data.get("speech") or "").strip()
            if not content:
                content = event.raw_data
        else:
            content = event.raw_data

        if not content:
            return

        category = "event"
        if event.event_type == "location":
            category = "location"
        elif event.event_type == "_speech":
            category = "speech"
        elif event.event_type in ("_quest", "_uquest", "quest"):
            category = "quest"

        fact = self._fact_pool.add_fact(
            content=content,
            category=category,
            game_ts=event.game_ts,
            knower_ids=present_ids,
        )

        # Speech: also propagate to companions in earshot
        if event.event_type == "_speech" and event.parsed_data:
            companions = event.parsed_data.get("companions", [])
            if companions:
                self._fact_pool.propagate_earshot(fact.fact_id, companions)

    def _handle_reset(self) -> None:
        """Handle init/wipe — clear all agent buffers, world state, and group memory."""
        logger.info("Session reset — clearing all agent buffers and group memory")
        self._agent_buffers.clear()
        self._world_events.clear()
        self._group_memory = TieredMemory()
        self._prev_active_npc_ids = set()
        self.current_location = "Unknown"
