"""
Canonical data schemas for the Many-Mind Kernel.

THE CONTRACT — shared types between Falcon and Progeny.
Both services import from here. If it's not in this file,
it's not part of the interface.

Wire flow:
  SKSE → Falcon → Qdrant wrapper (write + auto-embed) → signal Progeny
  Progeny → Qdrant wrapper (write + auto-embed) → keys via HTTP → Falcon
  Falcon → Qdrant (key lookup for response text) → SKSE wire format

Both services write through the same Qdrant enrichment wrapper
(text in → auto-embed 384d semantic + 9d emotional → store → return key).
Progeny owns cognitive work: emotional deltas, retrieval, scheduling,
prompting, LLM interaction. Shared emotional projection in shared/emotional.py.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """
    SKSE event types from the wire protocol.

    See living doc §SKSE Wire Protocol for the full taxonomy with per-type
    data field formats. Several types are prefix-matched in HerikaServer
    (e.g. any type starting with "info" or "addnpc"); Falcon preserves the
    lowercased type verbatim and Progeny handles prefix semantics.
    """
    # --- Turn triggers (Progeny detects these in incoming TickPackage) ---
    INPUTTEXT = "inputtext"
    INPUTTEXT_S = "inputtext_s"

    # --- Response polling (Falcon-local, never reaches Progeny) ---
    REQUEST = "request"

    # --- Dialogue / speech context ---
    SPEECH = "_speech"
    CHAT = "chat"
    CHATNF = "chatnf"
    JUST_SAY = "just_say"
    FUNCRET = "funcret"

    # --- Session lifecycle ---
    INIT = "init"
    WIPE = "wipe"
    PLAYERDIED = "playerdied"
    GOODNIGHT = "goodnight"
    WAITSTART = "waitstart"
    WAITSTOP = "waitstop"

    # --- NPC registration and live stats ---
    ADDNPC = "addnpc"
    UPDATESTATS = "updatestats"
    ITEMTRANSFER = "itemtransfer"
    ENABLE_BG = "enable_bg"
    SWITCHRACE = "switchrace"

    # --- Quest events ---
    QUEST = "quest"
    QUEST_JSON = "_quest"
    UQUEST = "_uquest"
    QUESTDATA = "_questdata"
    QUESTRESET = "_questreset"

    # --- Info / world events (info* prefix in HerikaServer) ---
    INFO = "info"
    INFONPC = "infonpc"
    INFOLOC = "infoloc"
    INFOSAVE = "infosave"
    LOCATION = "location"
    DEATH = "death"
    BOOK = "book"
    CONTENTBOOK = "contentbook"

    # --- Diary ---
    DIARY = "diary"
    DIARY_NEARBY = "diary_nearby"

    # --- World data utilities (bulk /‑delimited topology loads) ---
    UTIL_LOCATION_NAME = "util_location_name"
    UTIL_FACTION_NAME = "util_faction_name"
    UTIL_LOCATION_NPC = "util_location_npc"
    NAMED_CELL = "named_cell"
    NAMED_CELL_STATIC = "named_cell_static"

    # --- Task management ---
    FORCE_CURRENT_TASK = "force_current_task"
    RECOVER_LAST_TASK = "recover_last_task"

    # --- Configuration / admin ---
    SETCONF = "setconf"
    TOGGLEMODEL = "togglemodel"

    # --- Dynamic profiles ---
    UPDATEPROFILE = "updateprofile"
    UPDATEPROFILES_BATCH_ASYNC = "updateprofiles_batch_async"
    CORE_PROFILE_ASSIGN = "core_profile_assign"

    # --- SNQE (Synthetic Narrative Quest Engine) ---
    SNQE = "snqe"

    # --- Deprecated (log + ignore) ---
    UPDATEEQUIPMENT = "updateequipment"
    UPDATEINVENTORY = "updateinventory"
    UPDATESKILLS = "updateskills"


class CompressionTier(str, Enum):
    """Memory compression tiers in Qdrant."""
    RAW = "RAW"
    MOD = "MOD"
    MAX = "MAX"


class ExtractionLevel(str, Enum):
    """How cleanly the LLM response parsed."""
    STRICT = "strict"
    REPAIRED = "repaired"
    REGEX = "regex"
    PLAINTEXT = "plaintext"


class PrivacyLevel(str, Enum):
    """Memory access control levels."""
    PRIVATE = "PRIVATE"
    SEMI_PRIVATE = "SEMI_PRIVATE"
    COLLECTIVE = "COLLECTIVE"
    ANONYMOUS = "ANONYMOUS"


# ---------------------------------------------------------------------------
# Structurally decoded event data models — output of event_parsers.py
# ---------------------------------------------------------------------------

class SpeechData(BaseModel):
    """
    Decoded _speech event data. JSON payload from SKSE.

    companions[] contains names of nearby NPCs — used by Progeny for
    Many-Mind scheduling (Chorus tier candidates).
    """
    listener: str
    speaker: str
    speech: str
    location: str
    companions: list[str] = Field(default_factory=list)
    distance: float = 0.0


class NpcRegistration(BaseModel):
    """
    Decoded addnpc event data. @-delimited, 43+ fields.

    Sent when an NPC enters loaded cells. Contains skills (18), equipment
    (10 slots), stats (8), mods, factions, and class info.
    """
    name: str
    base: str = ""
    gender: str = ""
    race: str = ""
    refid: str = ""
    skills: dict[str, str] = Field(default_factory=dict)
    equipment: dict[str, str] = Field(default_factory=dict)
    stats: dict[str, float] = Field(default_factory=dict)
    mods: list[str] = Field(default_factory=list)
    factions: list[dict] = Field(default_factory=list)
    class_info: Optional[dict] = None


class NpcStats(BaseModel):
    """
    Decoded updatestats event data. @-delimited.

    Sent every ~3 seconds in combat or on hit.
    """
    npc_name: str
    level: int = 1
    health: float = 0.0
    health_max: float = 0.0
    magicka: float = 0.0
    magicka_max: float = 0.0
    stamina: float = 0.0
    stamina_max: float = 0.0
    scale: float = 1.0


class QuestData(BaseModel):
    """Decoded _quest event data. JSON payload."""
    form_id: str
    name: str = ""
    brief: str = ""
    stage: str = ""
    giver: str = ""
    status: str = ""


class QuestUpdate(BaseModel):
    """Decoded _uquest / _questdata event data. @-delimited."""
    form_id: str
    briefing: str = ""
    stage: str = ""


class ItemTransfer(BaseModel):
    """Decoded itemtransfer event data. Natural-language, regex-parsed."""
    source: str
    dest: str
    item_name: str
    count: int = 1


# ---------------------------------------------------------------------------
# Falcon → Progeny wire contract: TypedEvent and TickPackage
# ---------------------------------------------------------------------------

class TypedEvent(BaseModel):
    """
    A single SKSE event, structurally decoded by Falcon.

    raw_data is the verbatim data field from the wire. parsed_data is the
    structurally decoded form based on event_type (populated by
    event_parsers.py for known types; None for unrecognised types).
    Progeny does all semantic interpretation — Falcon only decodes structure.
    """
    event_type: str          # Lowercased verbatim from wire
    local_ts: str
    game_ts: float
    raw_data: str
    parsed_data: Optional[dict] = None   # Type-specific structural decode
    is_turn_trigger: bool = False


class TickPackage(BaseModel):
    """
    Falcon → Progeny: one tick window of typed SKSE events.

    Falcon ships this on its tick cadence (~1-3 seconds, configurable).
    Contains every SKSE event accumulated since the last tick, in arrival
    order. Progeny detects turn boundaries by scanning for inputtext /
    inputtext_s events. has_turn_trigger is a convenience boolean —
    Progeny should verify by scanning events.

    active_npc_ids is populated from Falcon's addnpc-derived NPC registry
    (the set of NPCs currently in loaded cells).
    """
    tick_id: UUID = Field(default_factory=uuid4)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    events: list[TypedEvent] = Field(default_factory=list)
    has_turn_trigger: bool = False
    tick_interval_ms: int = 0
    active_npc_ids: list[str] = Field(
        default_factory=list,
        description="NPC IDs currently in loaded cells, from accumulated addnpc events.",
    )


# ---------------------------------------------------------------------------
# Cognitive / context sub-models — used internally by Progeny
# ---------------------------------------------------------------------------

class EmotionalState(BaseModel):
    """Per-agent emotional state snapshot, computed by Progeny."""
    base_vector: list[float] = Field(..., min_length=9, max_length=9,
                                     description="Current 9d semagram")
    delta: list[float] = Field(..., min_length=9, max_length=9,
                               description="Change this tick")
    curvature: float = Field(0.0, description="1st derivative — priority gradient")
    snap: float = Field(0.0, description="2nd derivative — event boundary detector")


class MemorySummary(BaseModel):
    """A memory summary from retrieval."""
    text: str
    tier: CompressionTier
    arc_id: Optional[str] = None


class LoreHit(BaseModel):
    """A lore entry from Oghma retrieval."""
    topic: str
    content: str


class AgentMemoryContext(BaseModel):
    """Retrieved memory context for one agent."""
    retrieved_keys: list[str] = Field(default_factory=list)
    summaries: list[MemorySummary] = Field(default_factory=list)
    lore_hits: list[LoreHit] = Field(default_factory=list)


class ActorValues(BaseModel):
    """Current Creation Engine actor values for an NPC."""
    Aggression: int = Field(0, ge=0, le=3)
    Confidence: int = Field(2, ge=0, le=4)
    Morality: int = Field(3, ge=0, le=3)
    Mood: int = Field(0, ge=0, le=7)
    Assistance: int = Field(0, ge=0, le=2)


class NpcMetadata(BaseModel):
    """Per-NPC metadata from SKSE."""
    position: list[float] = Field(..., min_length=3, max_length=3)
    cell: str
    in_scene: bool = False
    level: int = 1
    hp: float = 100.0
    mp: float = 100.0
    sp: float = 100.0
    equipment: list[str] = Field(default_factory=list)
    actor_values: ActorValues = Field(default_factory=ActorValues)
    is_follower: bool = False
    active_task: Optional[str] = None


class PlayerState(BaseModel):
    """Current player state."""
    position: list[float] = Field(default_factory=lambda: [0.0, 0.0, 0.0],
                                  min_length=3, max_length=3)
    cell: str = "Unknown"
    input_text: Optional[str] = None


class WorldState(BaseModel):
    """World state delta."""
    weather: str = "clear"
    time_of_day: float = 12.0
    cell_transition: bool = False
    reset: bool = False


# ---------------------------------------------------------------------------
# LLM response models — Progeny → Falcon contract
# ---------------------------------------------------------------------------

class ActionCommand(BaseModel):
    """A single action from the 43-command vocabulary."""
    command: str
    target: Optional[str] = None
    item: Optional[str] = None


class ActorValueDeltas(BaseModel):
    """LLM-proposed changes to Creation Engine actor values."""
    Aggression: Optional[int] = Field(None, ge=0, le=3)
    Confidence: Optional[int] = Field(None, ge=0, le=4)
    Morality: Optional[int] = Field(None, ge=0, le=3)
    Mood: Optional[int] = Field(None, ge=0, le=7)
    Assistance: Optional[int] = Field(None, ge=0, le=2)


class UpdatedHarmonics(BaseModel):
    """LLM-proposed emotional state update (proposed by LLM, validated by Progeny)."""
    base_vector: list[float] = Field(..., min_length=9, max_length=9)


class NewMemory(BaseModel):
    """A memory the LLM wants to store."""
    text: str


class AgentResponse(BaseModel):
    """Response for a single agent from the LLM turn.

    utterance_key: if set, the utterance text was written to Qdrant via
    the enrichment wrapper and this is the point ID. Falcon reads the
    text by key for wire formatting. Falls back to inline utterance if
    utterance_key is None (backward compat, tests, stub mode).
    """
    agent_id: str
    utterance: Optional[str] = None
    utterance_key: Optional[str] = None
    actor_value_deltas: Optional[ActorValueDeltas] = None
    actions: list[ActionCommand] = Field(default_factory=list)
    updated_harmonics: Optional[UpdatedHarmonics] = None
    new_memories: list[NewMemory] = Field(default_factory=list)
    extraction_level: ExtractionLevel = ExtractionLevel.STRICT


class LLMTimings(BaseModel):
    """Token-level timing breakdown from the LLM backend."""
    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    prompt_tokens_per_sec: float = 0.0
    generated_tokens: int = 0
    generation_ms: float = 0.0
    generation_tokens_per_sec: float = 0.0
    cache_tokens: int = 0  # Tokens served from KV cache (skipped prompt eval)


class TurnResponse(BaseModel):
    """
    Progeny → Falcon: per-turn response bundle.

    Returned as the HTTP response body when the TickPackage contained
    a turn trigger (inputtext / inputtext_s). tick_id echoes the
    TickPackage.tick_id for correlation.
    """
    tick_id: UUID
    turn_id: UUID = Field(default_factory=uuid4)
    responses: list[AgentResponse] = Field(default_factory=list)
    processing_time_ms: int = 0
    model_used: str = "stub"
    llm_timings: Optional[LLMTimings] = None


class AckResponse(BaseModel):
    """
    Progeny → Falcon: ack for data-only tick packages.

    Returned immediately — Progeny accumulated the events.
    tick_id echoes the TickPackage.tick_id for correlation.
    """
    tick_id: UUID
    status: str = "accumulated"
