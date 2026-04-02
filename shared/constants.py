"""
Constants for the Many-Mind Kernel.

Emotional axes, actor values, command vocabulary, collection names.
"""

# ---------------------------------------------------------------------------
# 9d Emotional Semagram
# ---------------------------------------------------------------------------

# Gram-Schmidt priority order — earlier axes preserved more faithfully
EMOTIONAL_AXES: list[str] = [
    "fear",        # dim 0 — drift 1.000
    "anger",       # dim 1 — drift 0.985
    "love",        # dim 2 — drift 0.986
    "disgust",     # dim 3 — drift 0.846
    "excitement",  # dim 4 — drift 0.941
    "sadness",     # dim 5 — drift 0.900
    "joy",         # dim 6 — drift 0.780
    "safety",      # dim 7 — drift 0.803
    "residual",    # dim 8 — orthogonal complement magnitude
]

EMOTIONAL_DIM: int = 9
SEMANTIC_DIM: int = 384

# Zero semagram — default emotional state
ZERO_SEMAGRAM: list[float] = [0.0] * EMOTIONAL_DIM


# ---------------------------------------------------------------------------
# Creation Engine Actor Values
# ---------------------------------------------------------------------------

# { name: (min, max, description) }
ACTOR_VALUE_RANGES: dict[str, tuple[int, int, str]] = {
    "Aggression": (0, 3, "0=Unaggressive 1=Aggressive 2=Very Aggressive 3=Frenzied"),
    "Confidence": (0, 4, "0=Cowardly 1=Cautious 2=Average 3=Brave 4=Foolhardy"),
    "Morality":   (0, 3, "0=Any crime 1=Violence against enemies 2=Property crime 3=No crime"),
    "Mood":       (0, 7, "0=Neutral 1=Anger 2=Fear 3=Happy 4=Sad 5=Surprised 6=Puzzled 7=Disgusted"),
    "Assistance": (0, 2, "0=Nobody 1=Allies 2=Friends and allies"),
}


# ---------------------------------------------------------------------------
# SKSE Event Types
# ---------------------------------------------------------------------------

# Player input event types — Progeny uses these to identify player speech
# among accumulated events and decide when to respond. This is a semantic
# label for event classification, not a flow-control gate.
PLAYER_INPUT_TYPES: frozenset[str] = frozenset({"inputtext", "inputtext_s"})

# Events handled locally by Falcon
# request  : SKSE polls for queued responses — dequeue from local response queue.
# chatnf   : Chat target not found — log and return empty.
# just_say : Direct output passthrough — queue data text directly for SKSE.
FALCON_LOCAL_TYPES: frozenset[str] = frozenset({"request", "chatnf", "just_say"})

# Session lifecycle events (carry session semantics; forwarded to Progeny in tick).
# init / wipe also trigger an NPC registry clear on Falcon.
SESSION_TYPES: frozenset[str] = frozenset({
    "init", "wipe", "playerdied", "goodnight", "waitstart", "waitstop",
})

# NPC registration and live stats events.
NPC_DATA_TYPES: frozenset[str] = frozenset({
    "addnpc", "updatestats", "itemtransfer", "enable_bg", "switchrace",
})

# Quest management events.
QUEST_TYPES: frozenset[str] = frozenset({
    "_quest", "_uquest", "_questdata", "_questreset", "quest",
})

# World data utility events — bulk /‑delimited topology/location loads.
WORLD_DATA_TYPES: frozenset[str] = frozenset({
    "util_location_name", "util_faction_name", "util_location_npc",
    "named_cell", "named_cell_static",
})

# NOTE: Several event types in HerikaServer use prefix matching (strpos),
# e.g. any type starting with "info" or "addnpc". Falcon lowercases the type
# and ships it verbatim in TypedEvent.event_type; Progeny handles prefix
# semantics if needed.


# ---------------------------------------------------------------------------
# HerikaServer Command Vocabulary (43 commands)
# ---------------------------------------------------------------------------

COMMAND_VOCABULARY: frozenset[str] = frozenset({
    # Combat
    "Attack", "AttackHunt", "Brawl", "SheatheWeapon", "CastSpell", "Surrender",
    # Movement
    "MoveTo", "TravelTo", "Follow", "FollowPlayer", "ComeCloser",
    "ReturnBackHome", "StopWalk", "IncreaseWalkSpeed", "DecreaseWalkSpeed",
    # Items/Economy
    "GiveItemTo", "GiveItemToPlayer", "GiveGoldTo", "PickupItem",
    "OpenInventory", "OpenInventory2", "CheckInventory",
    # Intelligence
    "Inspect", "LookAt", "InspectSurroundings", "SearchMemory",
    "SearchDiary", "ReadDiaryPage", "ReadQuestJournal", "GetDateTime",
    # Social/State
    "Talk", "SetCurrentTask", "MakeFollower", "EndConversation",
    "Relax", "TakeASeat", "GoToSleep", "WaitHere",
    # Ceremonial/Special
    "Toast", "Drink", "StartRitualCeremony", "EndRitualCeremony",
    "Training", "UseSoulGaze",
    # MMK extension — actor value dial-tuning
    # Wire format: NPCName|command|SetBehavior@Aggression@2
    # Handled by MMKSetBehavior.psc (CHIM_CommandReceived ModEvent)
    "SetBehavior",
})


# ---------------------------------------------------------------------------
# Qdrant Collection Names
# ---------------------------------------------------------------------------

COLLECTION_NPC_MEMORIES: str = "skyrim_npc_memories"
COLLECTION_WORLD_EVENTS: str = "skyrim_world_events"
COLLECTION_SESSION_CONTEXT: str = "skyrim_session_context"
COLLECTION_AGENT_STATE: str = "skyrim_agent_state"
COLLECTION_LORE: str = "skyrim_lore"
COLLECTION_NPC_PROFILES: str = "skyrim_npc_profiles"


# ---------------------------------------------------------------------------
# CHIM Wire Protocol Constants
# ---------------------------------------------------------------------------

WIRE_DELIMITER: str = "|"
WIRE_LINE_ENDING: str = "\r\n"
WIRE_ACTION_PARAM_SEPARATOR: str = "@"
