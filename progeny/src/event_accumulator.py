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
from progeny.src.fact_pool import NpcBitIndex

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from progeny.src.fact_pool import FactPool
    from mindcore.harmonic_buffer import HarmonicState

# Dialogue event types eligible for behaviour adoption
_DIALOGUE_TYPES: frozenset[str] = frozenset({"chat", "_speech"})

# IDs that are never NPC agents — skip buffer creation and adoption
_NON_AGENT_IDS: frozenset[str] = frozenset({"Player", ""})

logger = logging.getLogger(__name__)


def _felt_vector(
    harmonic_state: "HarmonicState",
    agent_id: str,
) -> list[float] | None:
    """Extract the agent's current fast-buffer as a compact felt vector.

    Returns None if the agent has no state yet or is at zero (avoids
    polluting entries with uninformative all-zero vectors).
    """
    fast = harmonic_state.get_semagram(agent_id)
    if not any(abs(v) > 1e-6 for v in fast):
        return None
    return [round(v, 3) for v in fast]


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
    earshot_bits: int = 0  # Bitvector: which other agents are within hearing range

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
        # Earshot context from the most recent _speech event — who can hear
        # the player when inputtext arrives. None = no speech context yet.
        self._speech_earshot: Optional[dict] = None
        # ATMS fact pool — bitvector-tagged world knowledge
        self._fact_pool = fact_pool
        # Group-level shared timeline — the canonical record of "what happened"
        # that all present participants share. Condenses through the same
        # TieredMemory pipeline as personal history. Individual NPCs carry
        # their own emotional signatures; the group memory carries the facts.
        self._group_memory = TieredMemory()
        # Earshot proximity tracking — per-session, cleared on init/wipe
        self._earshot_index = NpcBitIndex()

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

        # Coarse earshot: all active NPCs in loaded cells can hear each other
        self._update_coarse_earshot(package.active_npc_ids)

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

            # Stash _speech earshot context for gating record_player_input
            if event_type == "_speech" and event.parsed_data:
                self._speech_earshot = event.parsed_data

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

            # Route to agent buffer(s) based on event type
            agent_roles = self._extract_agent_ids(event)
            routed = False
            for agent_id, role in agent_roles:
                if agent_id not in _NON_AGENT_IDS:
                    self._get_or_create_buffer(agent_id).append(event)
                    self._adopt_dialogue(event, agent_id, role)
                    routed = True
            if not routed:
                # World/info events without a clear agent owner
                self._world_events.append(event)

            # Update earshot bitvectors from _speech companion data
            self._update_earshot(event)

            # Behavior adoption: canned NPC _speech is recorded into the
            # speaker's dialogue history the same way LLM-generated output
            # is, so on the next turn the agent sees "I said this" instead
            # of having a silent gap where vanilla dialogue happened. Only
            # fires when the speaker is an active NPC (player speech goes
            # through record_player_input from inputtext events).
            if event.event_type == "_speech" and event.parsed_data:
                speaker = event.parsed_data.get("speaker", "")
                if speaker and speaker not in _NON_AGENT_IDS and speaker in self._active_npc_ids:
                    speech_text = (event.parsed_data.get("speech") or "").strip()
                    if speech_text:
                        self.record_npc_speech(speaker, speech_text)

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

    def record_agent_output(
        self,
        agent_id: str,
        utterance: str,
        harmonic_state: "HarmonicState | None" = None,
    ) -> None:
        """Record LLM-generated output into agent's dialogue history and
        the shared group timeline.

        Behavior adoption: adopted as the agent's own output (role=assistant).
        felt_at_speaking carries the speaker's 9d fast buffer at output time,
        encoding the emotional state from which the utterance emerged. This
        is the self-perspective: how it felt to say this.
        """
        buf = self._get_or_create_buffer(agent_id)
        entry: dict = {"role": "assistant", "content": utterance}
        if harmonic_state is not None:
            felt = _felt_vector(harmonic_state, agent_id)
            if felt:
                entry["felt_at_speaking"] = felt
        buf.dialogue_history.append(entry)
        # Group timeline — everyone present heard this NPC speak.
        # No felt state here: this is the objective record, not the
        # speaker's subjective experience.
        self._group_memory.verbatim.append(
            {"role": agent_id, "content": utterance}
        )

    def record_npc_speech(
        self,
        agent_id: str,
        text: str,
        harmonic_state: "HarmonicState | None" = None,
    ) -> None:
        """Record canned NPC speech (from _speech events) into dialogue history.

        Behavior adoption: externally-generated NPC dialogue is adopted as
        the agent's own output. felt_at_speaking is included when harmonic_state
        is available, giving the canned line an emotional fingerprint.
        """
        buf = self._get_or_create_buffer(agent_id)
        entry: dict = {"role": "assistant", "content": text}
        if harmonic_state is not None:
            felt = _felt_vector(harmonic_state, agent_id)
            if felt:
                entry["felt_at_speaking"] = felt
        buf.dialogue_history.append(entry)
        self._group_memory.verbatim.append(
            {"role": agent_id, "content": text}
        )

    def record_player_input(
        self,
        text: str,
        harmonic_state: "HarmonicState | None" = None,
    ) -> None:
        """Record player input into dialogue history for NPCs in earshot.

        Each NPC that heard the player gets a private entry with felt_at_receiving
        — how it landed for THEM specifically (the other-perspective). NPCs outside
        earshot get nothing.

        Player input is NOT recorded in _group_memory.verbatim: it already appears
        verbatim in the payload's player_input field, and per-NPC framing belongs
        in private Layer 2, not the shared objective record. Keeping group_memory
        as NPC-speech-only makes shared_recent grow more slowly and more cacheable.
        """

        # Per-NPC dialogue history — earshot-filtered (experience layer).
        # Each entry carries felt_at_receiving: how the player's words landed
        # for that specific NPC (their harmonic state at reception time).
        ctx = self._speech_earshot
        if ctx is not None:
            addressee = ctx.get("listener", "")
            companions = ctx.get("companions", [])

            # Addressee hears the player directly.
            if addressee and addressee not in _NON_AGENT_IDS:
                buf = self._get_or_create_buffer(addressee)
                entry: dict = {"role": "user", "content": text}
                if harmonic_state is not None:
                    felt = _felt_vector(harmonic_state, addressee)
                    if felt:
                        entry["felt_at_receiving"] = felt
                buf.dialogue_history.append(entry)

            # Companions in earshot overheard it.
            for comp in companions:
                if comp and comp not in _NON_AGENT_IDS and comp != addressee:
                    buf = self._get_or_create_buffer(comp)
                    target = addressee or "someone"
                    entry = {"role": "user",
                             "content": f"Player [to {target}]: {text}"}
                    if harmonic_state is not None:
                        felt = _felt_vector(harmonic_state, comp)
                        if felt:
                            entry["felt_at_receiving"] = felt
                    buf.dialogue_history.append(entry)
        else:
            # No speech context — fallback to all active NPCs (typed input).
            for agent_id in self._active_npc_ids:
                if agent_id not in _NON_AGENT_IDS:
                    buf = self._get_or_create_buffer(agent_id)
                    entry = {"role": "user", "content": text}
                    if harmonic_state is not None:
                        felt = _felt_vector(harmonic_state, agent_id)
                        if felt:
                            entry["felt_at_receiving"] = felt
                    buf.dialogue_history.append(entry)

    def _extract_agent_ids(self, event: TypedEvent) -> list[tuple[str, str]]:
        """
        Extract agents involved in this event with their perspective role.

        Returns list of (agent_id, role) tuples. Role is one of:
          "speaker"  — agent produced this dialogue (adoption: role=assistant)
          "listener" — dialogue was directed at this agent (adoption: role=user)
          "owner"    — non-dialogue event about this agent (no adoption)

        Uses structural data from Falcon's parsers — no semantic interpretation.
        """
        parsed = event.parsed_data
        if parsed is None:
            return []

        event_type = event.event_type
        results: list[tuple[str, str]] = []

        # Speech events — speaker + listener
        if event_type == "_speech":
            speaker = parsed.get("speaker", "")
            if speaker:
                results.append((speaker, "speaker"))
            listener = parsed.get("listener", "")
            if listener:
                results.append((listener, "listener"))
            return results

        # Chat events — speaker + listener
        if event_type == "chat":
            speaker = parsed.get("speaker", "")
            if speaker:
                results.append((speaker, "speaker"))
            listener = parsed.get("listener", "")
            if listener:
                results.append((listener, "listener"))
            return results

        # NPC registration (prefix match — DLL may send addnpc variants)
        if event_type.startswith("addnpc") and "name" in parsed:
            return [(parsed["name"], "owner")]

        # Stats update
        if event_type == "updatestats" and "npc_name" in parsed:
            return [(parsed["npc_name"], "owner")]

        # Item transfer — source is the acting agent
        if event_type == "itemtransfer" and "source" in parsed:
            return [(parsed["source"], "owner")]

        return []

    def _adopt_dialogue(self, event: TypedEvent, agent_id: str, role: str) -> None:
        """Adopt dialogue into an agent's history based on perspective.

        Speaker: role=assistant — the agent said this (behaviour adoption).
        Listener: role=user with speaker attribution — directed speech received.
        Owner: no adoption (non-dialogue events).

        _speech events are handled separately by record_npc_speech() which
        uses the clean parsed speech text instead of the raw JSON envelope.
        This method only handles chat events for speaker/listener adoption.
        """
        if role == "owner" or event.event_type not in _DIALOGUE_TYPES:
            return
        # _speech adoption is handled by record_npc_speech() — skip here
        # to avoid double-recording and to use clean parsed text.
        if event.event_type == "_speech":
            return
        parsed = event.parsed_data
        if parsed is None:
            return
        speech = parsed.get("speech", "")
        if not speech:
            return

        buf = self._get_or_create_buffer(agent_id)
        if role == "speaker":
            buf.dialogue_history.append({"role": "assistant", "content": speech})
        elif role == "listener":
            speaker_name = parsed.get("speaker", "Unknown")
            buf.dialogue_history.append(
                {"role": "user", "content": f"{speaker_name} [to you]: {speech}"}
            )

    # ------------------------------------------------------------------
    # Earshot proximity tracking
    # ------------------------------------------------------------------

    def _update_earshot(self, event: TypedEvent) -> None:
        """Update earshot bitvectors from _speech companion data.

        The SKSE _speech event carries companions[] — the NPCs within
        hearing distance. Set reciprocal earshot bits for the speaker
        and all companions so each knows who else was present.
        """
        if event.event_type != "_speech" or event.parsed_data is None:
            return
        parsed = event.parsed_data
        speaker = parsed.get("speaker", "")
        companions = parsed.get("companions", [])

        # Build earshot group: speaker + companions, excluding non-agents
        earshot_group: list[str] = []
        if speaker and speaker not in _NON_AGENT_IDS:
            earshot_group.append(speaker)
        for comp in companions:
            if comp and comp not in _NON_AGENT_IDS:
                earshot_group.append(comp)
        if not earshot_group:
            return

        group_mask = self._earshot_index.mask_for_all(earshot_group)
        for name in earshot_group:
            buf = self._get_or_create_buffer(name)
            buf.earshot_bits |= group_mask

    def _update_coarse_earshot(self, active_npc_ids: list[str]) -> None:
        """Set coarse earshot bits from loaded-cell proximity.

        All active NPCs in the same loaded cells are potential earshot
        candidates. Only updates agents that already have buffers to
        avoid creating empty ones from bulk NPC lists.
        """
        npc_ids = [nid for nid in active_npc_ids if nid not in _NON_AGENT_IDS]
        if not npc_ids:
            return
        group_mask = self._earshot_index.mask_for_all(npc_ids)
        for name in npc_ids:
            if name in self._agent_buffers:
                self._agent_buffers[name].earshot_bits |= group_mask

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
        self._earshot_index = NpcBitIndex()
        self._speech_earshot = None
