"""
Structural event data parsers for Falcon.

These parsers decode the `data` field of each SKSE event type into typed
Python dicts (matching the shared schema models). Mechanical, deterministic,
no semantic interpretation. Falcon decodes structure and ships; Progeny
interprets meaning.

Contract for every parser:
  - Accept the raw `data` string from the wire event.
  - Return a dict (serialisable as the corresponding schema model) or None.
  - Never raise. Log warnings on unexpected input; silently skip bad sub-fields.
  - Missing optional fields get their model defaults — never cause a crash.

Dispatcher:
  parse_typed_data(event_type, raw_data) → dict | None
"""
from __future__ import annotations

import json
import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# _speech → SpeechData
# ---------------------------------------------------------------------------

def parse_speech(data: str) -> Optional[dict]:
    """_speech → SpeechData dict. JSON payload."""
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        logger.warning("_speech: JSON decode failed: %s", exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("_speech: data is not a JSON object")
        return None

    companions_raw = parsed.get("companions", [])
    companions = [str(c) for c in companions_raw] if isinstance(companions_raw, list) else []

    try:
        distance = float(parsed.get("distance", 0.0))
    except (TypeError, ValueError):
        distance = 0.0

    return {
        "listener": str(parsed.get("listener", "")),
        "speaker":  str(parsed.get("speaker",  "")),
        "speech":   str(parsed.get("speech",   "")),
        "location": str(parsed.get("location", "")),
        "companions": companions,
        "distance": distance,
    }


# ---------------------------------------------------------------------------
# addnpc → NpcRegistration
# ---------------------------------------------------------------------------

# NPC skill names in @-delimited addnpc order, indices 5-22 (18 skills)
_NPC_SKILLS: list[str] = [
    "archery", "block", "onehanded", "twohanded", "conjuration",
    "destruction", "restoration", "alteration", "illusion",
    "heavyarmor", "lightarmor", "lockpicking", "pickpocket",
    "sneak", "speech", "smithing", "alchemy", "enchanting",
]

# NPC equipment slot names, indices 23-32 (10 slots, each "name^baseid")
_EQUIPMENT_SLOTS: list[str] = [
    "helmet", "armor", "boots", "gloves", "amulet",
    "ring", "cape", "backpack", "left_hand", "right_hand",
]

# NPC stat field names and their defaults, indices 33-40
_NPC_STAT_FIELDS: list[tuple[str, object]] = [
    ("level",       1),
    ("health",      0.0),
    ("health_max",  0.0),
    ("magicka",     0.0),
    ("magicka_max", 0.0),
    ("stamina",     0.0),
    ("stamina_max", 0.0),
    ("scale",       1.0),
]


def parse_addnpc(data: str) -> Optional[dict]:
    """addnpc → NpcRegistration dict. @-delimited, 43+ fields."""
    parts = data.split("@")
    name = parts[0].strip() if parts else ""
    if not name:
        logger.warning("addnpc: missing NPC name in data: %.60s", data)
        return None

    def _get(idx: int) -> str:
        return parts[idx] if len(parts) > idx else ""

    result: dict = {
        "name":   name,
        "base":   _get(1),
        "gender": _get(2),
        "race":   _get(3),
        "refid":  _get(4),
        "skills":     {},
        "equipment":  {},
        "stats":      {},
        "mods":       [],
        "factions":   [],
        "class_info": None,
    }

    # Skills: indices 5-22
    for i, skill_name in enumerate(_NPC_SKILLS):
        result["skills"][skill_name] = _get(5 + i)

    # Equipment: indices 23-32 (each "name^baseid")
    for i, slot_name in enumerate(_EQUIPMENT_SLOTS):
        raw_slot = _get(23 + i)
        slot_parts = raw_slot.split("^", 1)
        result["equipment"][slot_name] = slot_parts[0]
        result["equipment"][slot_name + "_baseid"] = slot_parts[1] if len(slot_parts) > 1 else ""

    # Stats: indices 33-40
    for i, (field_name, default) in enumerate(_NPC_STAT_FIELDS):
        raw = _get(33 + i)
        try:
            result["stats"][field_name] = int(raw) if isinstance(default, int) else float(raw)
        except (ValueError, TypeError):
            result["stats"][field_name] = default

    # Mods: index 41, #-delimited
    raw_mods = _get(41)
    if raw_mods:
        result["mods"] = [m for m in raw_mods.split("#") if m]

    # Factions: index 42, "formID:rank#formID:rank#..."
    raw_factions = _get(42)
    for pair in raw_factions.split("#"):
        faction_parts = pair.split(":", 1)
        if len(faction_parts) == 2:
            try:
                result["factions"].append({
                    "formid": faction_parts[0],
                    "rank":   int(faction_parts[1]),
                })
            except ValueError:
                pass

    # Class: index 43, "className:formID[:trainSkill:trainLevel]"
    raw_class = _get(43)
    if raw_class:
        class_parts = raw_class.split(":", 3)
        if len(class_parts) >= 2:
            class_info: dict = {"name": class_parts[0], "formid": class_parts[1]}
            if len(class_parts) == 4 and class_parts[2]:
                class_info["teaches"] = class_parts[2]
                try:
                    class_info["max_training_level"] = int(class_parts[3])
                except ValueError:
                    pass
            result["class_info"] = class_info

    return result


# ---------------------------------------------------------------------------
# updatestats → NpcStats
# ---------------------------------------------------------------------------

def parse_updatestats(data: str) -> Optional[dict]:
    """updatestats → NpcStats dict. @-delimited: npc@level@hp@hp_max@mp@mp_max@sp@sp_max@scale"""
    parts = data.split("@")
    npc_name = parts[0].strip() if parts else ""
    if not npc_name:
        logger.warning("updatestats: missing NPC name in data: %.60s", data)
        return None

    def _f(idx: int, default: float = 0.0) -> float:
        try:
            return float(parts[idx]) if len(parts) > idx else default
        except (ValueError, TypeError):
            return default

    def _i(idx: int, default: int = 0) -> int:
        try:
            return int(parts[idx]) if len(parts) > idx else default
        except (ValueError, TypeError):
            return default

    return {
        "npc_name":    npc_name,
        "level":       _i(1, 1),
        "health":      _f(2),
        "health_max":  _f(3),
        "magicka":     _f(4),
        "magicka_max": _f(5),
        "stamina":     _f(6),
        "stamina_max": _f(7),
        "scale":       _f(8, 1.0),
    }


# ---------------------------------------------------------------------------
# _quest → QuestData
# ---------------------------------------------------------------------------

def parse_quest_json(data: str) -> Optional[dict]:
    """_quest → QuestData dict. JSON payload."""
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError as exc:
        logger.warning("_quest: JSON decode failed: %s", exc)
        return None
    if not isinstance(parsed, dict):
        logger.warning("_quest: data is not a JSON object")
        return None

    quest_data = parsed.get("data", {})
    giver = str(quest_data.get("questgiver", "")) if isinstance(quest_data, dict) else ""

    return {
        "form_id": str(parsed.get("formId",       "")),
        "name":    str(parsed.get("name",          "")),
        "brief":   str(parsed.get("currentbrief",  "")),
        "stage":   str(parsed.get("stage",         "")),
        "giver":   giver,
        "status":  str(parsed.get("status",        "")),
    }


# ---------------------------------------------------------------------------
# _uquest / _questdata → QuestUpdate
# ---------------------------------------------------------------------------

def parse_quest_update(data: str) -> Optional[dict]:
    """_uquest / _questdata → QuestUpdate dict. @-delimited: formId@unknown@briefing@stage"""
    parts = data.split("@")
    form_id = parts[0].strip() if parts else ""
    if not form_id:
        return None
    return {
        "form_id":  form_id,
        "briefing": parts[2] if len(parts) > 2 else "",
        "stage":    parts[3] if len(parts) > 3 else "",
    }


# ---------------------------------------------------------------------------
# itemtransfer → ItemTransfer
# ---------------------------------------------------------------------------

_ITEM_TRANSFER_RE = re.compile(
    r"^(.+?)\s+gave\s+(\d+)\s+(.+?)\s+to\s+(.+)$",
    re.IGNORECASE,
)


def parse_itemtransfer(data: str) -> Optional[dict]:
    """itemtransfer → ItemTransfer dict. Natural language: 'Lydia gave 2 Health Potion to Faendal'"""
    m = _ITEM_TRANSFER_RE.match(data.strip())
    if not m:
        logger.warning("itemtransfer: unrecognised format: %.80s", data)
        return None
    return {
        "source":    m.group(1).strip(),
        "count":     int(m.group(2)),
        "item_name": m.group(3).strip(),
        "dest":      m.group(4).strip(),
    }


# ---------------------------------------------------------------------------
# util_location_name → LocationData
# ---------------------------------------------------------------------------

def parse_location_name(data: str) -> Optional[dict]:
    """util_location_name → dict. /-delimited: name/formid/region/hold/tags/is_interior/factions/x/y"""
    if data.strip() == "__CLEAR_ALL__":
        return {"clear_all": True}
    parts = data.split("/")
    if not parts or not parts[0].strip():
        return None

    def _get(idx: int) -> str:
        return parts[idx].strip() if len(parts) > idx else ""

    def _float(idx: int) -> float:
        try:
            return float(parts[idx]) if len(parts) > idx else 0.0
        except (ValueError, TypeError):
            return 0.0

    return {
        "name":        _get(0),
        "formid":      _get(1),
        "region":      _get(2),
        "hold":        _get(3),
        "tags":        _get(4),
        "is_interior": _get(5) == "1",
        "factions":    _get(6),
        "x":           _float(7),
        "y":           _float(8),
    }


# ---------------------------------------------------------------------------
# util_faction_name → FactionData
# ---------------------------------------------------------------------------

def parse_faction_name(data: str) -> Optional[dict]:
    """util_faction_name → dict. /-delimited: formid/name"""
    parts = data.split("/", 1)
    if len(parts) < 2:
        return None
    return {"formid": parts[0].strip(), "name": parts[1].strip()}


# ---------------------------------------------------------------------------
# util_location_npc → NpcPosition
# ---------------------------------------------------------------------------

def parse_location_npc(data: str) -> Optional[dict]:
    """util_location_npc → dict. /-delimited: npcName/x/y/z/tag"""
    parts = data.split("/")
    npc_name = parts[0].strip() if parts else ""
    if not npc_name:
        return None

    def _float(idx: int) -> float:
        try:
            return float(parts[idx]) if len(parts) > idx else 0.0
        except (ValueError, TypeError):
            return 0.0

    return {
        "npc_name": npc_name,
        "x":        _float(1),
        "y":        _float(2),
        "z":        _float(3),
        "tag":      parts[4].strip() if len(parts) > 4 else "",
    }


# ---------------------------------------------------------------------------
# named_cell → CellData
# ---------------------------------------------------------------------------

def parse_named_cell(data: str) -> Optional[dict]:
    """named_cell → dict. /-delimited, 12 fields: cell topology."""
    parts = data.split("/")
    if not parts or not parts[0].strip():
        return None

    def _get(idx: int) -> str:
        return parts[idx].strip() if len(parts) > idx else ""

    def _float(idx: int) -> float:
        try:
            return float(parts[idx]) if len(parts) > idx else 0.0
        except (ValueError, TypeError):
            return 0.0

    return {
        "cell_name":             _get(0),
        "id":                    _get(1),
        "location_id":           _get(2),
        "interior":              _get(3) == "1",
        "dest_door_cell_id":     _get(4),
        "dest_door_exterior":    _get(5),
        "door_id":               _get(6),
        "worldspace":            _get(7),
        "closed":                _get(8) == "1",
        "door_name":             _get(9),
        "door_x":                _float(10),
        "door_y":                _float(11),
    }


# ---------------------------------------------------------------------------
# named_cell_static → CellStaticItems
# ---------------------------------------------------------------------------

def parse_named_cell_static(data: str) -> Optional[dict]:
    """named_cell_static → dict. /-delimited: cellId/comma-separated name@refid pairs"""
    parts = data.split("/", 1)
    if not parts or not parts[0].strip():
        return None
    cell_id = parts[0].strip()
    items = []
    if len(parts) > 1 and parts[1].strip():
        for pair in parts[1].split(","):
            item_parts = pair.strip().split("@", 1)
            if item_parts and item_parts[0]:
                items.append({
                    "name":  item_parts[0].strip(),
                    "refid": item_parts[1].strip() if len(item_parts) > 1 else "",
                })
    return {"cell_id": cell_id, "items": items}


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_PARSERS: dict[str, object] = {
    "_speech":            parse_speech,
    "addnpc":             parse_addnpc,
    "updatestats":        parse_updatestats,
    "_quest":             parse_quest_json,
    "_uquest":            parse_quest_update,
    "_questdata":         parse_quest_update,
    "itemtransfer":       parse_itemtransfer,
    "util_location_name": parse_location_name,
    "util_faction_name":  parse_faction_name,
    "util_location_npc":  parse_location_npc,
    "named_cell":         parse_named_cell,
    "named_cell_static":  parse_named_cell_static,
}


def parse_typed_data(event_type: str, raw_data: str) -> Optional[dict]:
    """
    Dispatch to the appropriate structural parser for event_type.

    Returns a serialisable dict or None if:
      - No parser is registered for this event type (most types have none —
        their raw_data is preserved verbatim in TypedEvent).
      - The parser encounters unrecoverable input.

    Never raises.
    """
    parser = _PARSERS.get(event_type)
    if parser is None:
        return None
    try:
        return parser(raw_data)  # type: ignore[operator]
    except Exception:
        logger.exception("Unexpected error in parser for event type '%s'", event_type)
        return None
