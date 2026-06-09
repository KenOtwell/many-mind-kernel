"""End-to-end tests for `_speech` event handling across Falcon and Progeny.

Locks in four related behaviours that were introduced together:

  1. `parse_speech` extracts a typed `audios` field (Bethesda voice file
     path SKSE just read) into SpeechData.
  2. `EventAccumulator.ingest` adopts canned NPC `_speech` events into the
     speaker's `dialogue_history` per the Behavior Adoption design \u2014
     mirroring the LLM output path so the agent sees \"I said this\" on the
     next turn.
  3. Fact pool entries for `_speech` events store the clean `speech` text,
     not the JSON envelope (no audios path leakage into retrieval).
  4. `emotional_delta.process_inbound` embeds the clean speech text, not
     `raw_data`, so the 9d semagram isn't polluted by JSON syntax.

The canned payload mirrors the exact wire shape captured in production
gameplay logs (Bryst's ferry response) so the tests exercise realistic
input structure, including the backslash-escaped Bethesda path.
"""
from __future__ import annotations

import json
from unittest.mock import patch

import pytest

from falcon.src.event_parsers import parse_speech
from progeny.src.event_accumulator import AgentBuffer, EventAccumulator, TurnContext
from progeny.src.fact_pool import FactPool
from progeny.src.harmonic_buffer import HarmonicState
from progeny.src import emotional_delta as ed_mod
from shared.schemas import SpeechData, TickPackage, TypedEvent


# ---------------------------------------------------------------------------
# Test data \u2014 matches the real wire shape from gameplay logs
# ---------------------------------------------------------------------------

BRYST_SPEECH = "You think I do this for my health? Come back when you've got the coin."
BRYST_AUDIOS = (
    "Data\\Sound\\Voice\\Dawnguard.esm\\MaleEvenToned"
    "\\DLC1DialogueFerrySystem__00016833_1.wav"
)


def _canned_bryst_payload() -> str:
    """Canonical _speech JSON payload for an NPC canned line."""
    return json.dumps({
        "audios": BRYST_AUDIOS,
        "debug": "true",
        "listener": "Ken",
        "location": " outdoors ,Hold: Falkreath",
        "speaker": "Bryst",
        "speech": BRYST_SPEECH,
    })


def _player_payload() -> str:
    """_speech JSON where the speaker is the player (no audios field)."""
    return json.dumps({
        "listener": "Bryst",
        "speaker": "Ken",
        "speech": "Ken: I'd like to hire your boat.",
        "location": " outdoors ,Hold: Falkreath",
        "companions": ["Bryst", "Deem-Ja"],
    })


def _typed(raw: str, parsed: dict) -> TypedEvent:
    """Wrap a parsed _speech in a TypedEvent for ingest()."""
    return TypedEvent(
        event_type="_speech",
        local_ts="2026-06-09T18:00:00",
        game_ts=4696801.0,
        raw_data=raw,
        parsed_data=parsed,
    )


# ---------------------------------------------------------------------------
# 1. parse_speech: audios is a typed field
# ---------------------------------------------------------------------------

class TestSpeechParseAudios:
    def test_audios_extracted_for_canned_line(self):
        parsed = parse_speech(_canned_bryst_payload())
        assert parsed is not None
        assert parsed["audios"] == BRYST_AUDIOS

    def test_audios_is_none_when_absent(self):
        """Player input and LLM-generated speech have no audios field."""
        parsed = parse_speech(_player_payload())
        assert parsed is not None
        assert parsed["audios"] is None

    def test_speech_field_is_clean_text(self):
        parsed = parse_speech(_canned_bryst_payload())
        assert parsed["speech"] == BRYST_SPEECH
        assert "{" not in parsed["speech"]
        assert "audios" not in parsed["speech"]

    def test_speech_data_pydantic_round_trip(self):
        parsed = parse_speech(_canned_bryst_payload())
        sd = SpeechData(**parsed)
        assert sd.speaker == "Bryst"
        assert sd.speech == BRYST_SPEECH
        assert sd.audios == BRYST_AUDIOS

    def test_speech_data_accepts_missing_audios(self):
        """SpeechData.audios defaults to None for backward compatibility."""
        sd = SpeechData(
            listener="Player", speaker="Lydia", speech="Hello",
            location="Whiterun",
        )
        assert sd.audios is None


# ---------------------------------------------------------------------------
# 2. Behavior adoption: NPC _speech lands in dialogue_history
# ---------------------------------------------------------------------------

class TestSpeechAdoption:
    def test_npc_speech_adopted_into_dialogue_history(self):
        acc = EventAccumulator()
        parsed = parse_speech(_canned_bryst_payload())
        pkg = TickPackage(
            events=[_typed(_canned_bryst_payload(), parsed)],
            active_npc_ids=["Bryst", "Deem-Ja"],
        )
        acc.ingest(pkg)

        buf = acc._agent_buffers["Bryst"]
        assert len(buf.dialogue_history) == 1
        entry = buf.dialogue_history[0]
        assert entry["role"] == "assistant"
        assert entry["content"] == BRYST_SPEECH
        # No JSON noise leaked in
        assert "{" not in entry["content"]
        assert "audios" not in entry["content"]

    def test_npc_speech_also_lands_in_group_memory(self):
        acc = EventAccumulator()
        parsed = parse_speech(_canned_bryst_payload())
        pkg = TickPackage(
            events=[_typed(_canned_bryst_payload(), parsed)],
            active_npc_ids=["Bryst", "Deem-Ja"],
        )
        acc.ingest(pkg)

        group = acc._group_memory.verbatim
        assert len(group) == 1
        assert group[0]["role"] == "Bryst"
        assert group[0]["content"] == BRYST_SPEECH

    def test_player_speaker_not_adopted_into_dialogue_history(self):
        """Player input flows through inputtext -> record_player_input,
        not through _speech adoption. A _speech event where speaker==Player
        must not pollute any NPC's assistant history."""
        acc = EventAccumulator()
        parsed = parse_speech(_player_payload())
        pkg = TickPackage(
            events=[_typed(_player_payload(), parsed)],
            active_npc_ids=["Bryst", "Deem-Ja"],
        )
        acc.ingest(pkg)

        # Ken still gets a buffer via _extract_agent_id routing, but no adoption
        assert "Ken" in acc._agent_buffers
        assert len(acc._agent_buffers["Ken"].dialogue_history) == 0
        # And no spurious group entries
        assert acc._group_memory.verbatim == []

    def test_npc_speaker_not_in_active_ids_is_not_adopted(self):
        """Defensive: if _extract_agent_id returns a name that isn't in
        active_npc_ids (stray event from a despawned NPC), don't adopt."""
        acc = EventAccumulator()
        parsed = parse_speech(_canned_bryst_payload())
        pkg = TickPackage(
            events=[_typed(_canned_bryst_payload(), parsed)],
            active_npc_ids=["Deem-Ja"],  # Bryst NOT in active list
        )
        acc.ingest(pkg)

        # Buffer is created (event routing) but no adoption
        assert "Bryst" in acc._agent_buffers
        assert len(acc._agent_buffers["Bryst"].dialogue_history) == 0


# ---------------------------------------------------------------------------
# 3. Fact pool stores clean text, not JSON envelope
# ---------------------------------------------------------------------------

class TestSpeechFactContent:
    def test_speech_fact_content_is_clean_text(self):
        pool = FactPool()
        acc = EventAccumulator(fact_pool=pool)
        parsed = parse_speech(_canned_bryst_payload())
        pkg = TickPackage(
            events=[_typed(_canned_bryst_payload(), parsed)],
            active_npc_ids=["Bryst", "Deem-Ja"],
        )
        acc.ingest(pkg)

        speech_facts = [f for f in pool._facts.values() if f.category == "speech"]
        assert len(speech_facts) >= 1
        fact = speech_facts[0]
        assert fact.content == BRYST_SPEECH
        assert "{" not in fact.content
        assert "audios" not in fact.content


# ---------------------------------------------------------------------------
# 4. emotional_delta embeds clean speech, not JSON
# ---------------------------------------------------------------------------

class TestSpeechEmotionalEmbed:
    def test_process_inbound_uses_parsed_speech(self):
        """process_inbound should hand process_texts() the clean speech
        string, not the raw JSON envelope."""
        parsed = parse_speech(_canned_bryst_payload())
        event = _typed(_canned_bryst_payload(), parsed)

        buf = AgentBuffer(agent_id="Bryst")
        buf.events = [event]
        ctx = TurnContext(
            player_input="",
            agent_buffers={"Bryst": buf},
            active_npc_ids=["Bryst"],
            world_events=[],
            session_events=[],
        )

        captured: list[tuple[str, str]] = []

        def fake_process_texts(pairs, harmonic_state):
            captured.extend(pairs)
            return {agent_id: None for agent_id, _ in pairs}

        with patch.object(ed_mod, "process_texts", fake_process_texts):
            ed_mod.process_inbound(ctx, HarmonicState())

        assert len(captured) == 1
        agent_id, text = captured[0]
        assert agent_id == "Bryst"
        assert text == BRYST_SPEECH
        # The whole point: no JSON or audios path in the embed input
        assert "{" not in text
        assert "audios" not in text
        assert BRYST_AUDIOS not in text

    def test_process_inbound_falls_back_to_raw_if_no_parsed_speech(self):
        """If a _speech event somehow lacks parsed_data['speech'] but has
        raw_data, fall back to raw rather than dropping the event."""
        event = TypedEvent(
            event_type="_speech",
            local_ts="2026-06-09T18:00:00",
            game_ts=4696801.0,
            raw_data="bare fallback text",
            parsed_data={"speaker": "Bryst"},  # no speech field
        )
        buf = AgentBuffer(agent_id="Bryst")
        buf.events = [event]
        ctx = TurnContext(
            player_input="",
            agent_buffers={"Bryst": buf},
            active_npc_ids=["Bryst"],
            world_events=[],
            session_events=[],
        )

        captured: list[tuple[str, str]] = []

        def fake_process_texts(pairs, harmonic_state):
            captured.extend(pairs)
            return {}

        with patch.object(ed_mod, "process_texts", fake_process_texts):
            ed_mod.process_inbound(ctx, HarmonicState())

        assert captured == [("Bryst", "bare fallback text")]
