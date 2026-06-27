"""
End-to-end verification of _speech event handling.

Constructs an _speech event matching the exact wire payload observed
in production logs (Bryst's canned ferry response), pushes it through:

  Falcon parse_speech  -->  TypedEvent  -->  Progeny EventAccumulator
                                                |
                                                +--> dialogue_history adoption
                                                +--> group_memory recording
                                                +--> fact pool (clean text)
                                                +--> emotional_delta (clean text)

Inspects each downstream artifact and reports pass/fail. Run from repo root:
    python scripts/verify_speech_event.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Repo root on path so `shared`/`falcon`/`progeny` import cleanly
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from falcon.src.event_parsers import parse_speech
from progeny.src.event_accumulator import EventAccumulator
from progeny.src.fact_pool import FactPool
from shared.schemas import SpeechData, TickPackage, TypedEvent


# ASCII markers — Windows PowerShell's default console (CP1252) chokes on
# the Unicode checkmark/cross when called from a stock environment.
PASS = "[OK]  "
FAIL = "[FAIL]"
ok_count = 0
fail_count = 0


def check(label: str, condition: bool, detail: str = "") -> None:
    global ok_count, fail_count
    if condition:
        print(f"  {PASS} {label}" + (f"  ({detail})" if detail else ""))
        ok_count += 1
    else:
        print(f"  {FAIL} {label}" + (f"  ({detail})" if detail else ""))
        fail_count += 1


def main() -> int:
    # ------------------------------------------------------------------
    # 1. Realistic _speech wire payload \u2014 Bryst's canned ferry response.
    # ------------------------------------------------------------------
    print("=" * 70)
    print("1. parse_speech: extract typed `audios` field from canned NPC line")
    print("=" * 70)
    canned_payload = json.dumps({
        "audios": "Data\\Sound\\Voice\\Dawnguard.esm\\MaleEvenToned\\DLC1DialogueFerrySystem__00016833_1.wav",
        "debug": "true",
        "listener": "Ken",
        "location": " outdoors ,Hold: Falkreath",
        "speaker": "Bryst",
        "speech": "You think I do this for my health? Come back when you've got the coin.",
    })

    parsed = parse_speech(canned_payload)
    print(f"  parsed dict: {json.dumps(parsed, indent=2)}")
    check("parsed is not None", parsed is not None)
    check("speaker == 'Bryst'", parsed["speaker"] == "Bryst")
    check("speech is clean (no JSON braces)",
          parsed["speech"].startswith("You think I do this") and "{" not in parsed["speech"])
    check("audios is typed field, not None",
          parsed["audios"] is not None,
          detail=f"got {parsed['audios']!r}")
    check("audios path matches input",
          parsed["audios"].endswith("DLC1DialogueFerrySystem__00016833_1.wav"))

    # Validate the Pydantic model accepts the new shape
    sd = SpeechData(**parsed)
    check("SpeechData(**parsed) succeeds", isinstance(sd, SpeechData))
    check("SpeechData.audios round-trip", sd.audios == parsed["audios"])

    # ------------------------------------------------------------------
    # 2. EventAccumulator ingest: NPC _speech should adopt into history
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("2. EventAccumulator.ingest: NPC _speech adopted into dialogue_history")
    print("=" * 70)

    fact_pool = FactPool()
    accumulator = EventAccumulator(fact_pool=fact_pool)

    typed_event = TypedEvent(
        event_type="_speech",
        local_ts="2026-06-09T18:00:00",
        game_ts=4696801.0,
        raw_data=canned_payload,
        parsed_data=parsed,
    )
    package = TickPackage(
        events=[typed_event],
        tick_interval_ms=2000,
        active_npc_ids=["Bryst", "Deem-Ja"],
    )

    accumulator.ingest(package)

    bryst_buf = accumulator._agent_buffers.get("Bryst")
    check("Bryst buffer created", bryst_buf is not None)
    check("Bryst.events has 1 entry",
          bryst_buf is not None and len(bryst_buf.events) == 1)
    check("Bryst.dialogue_history has 1 entry",
          bryst_buf is not None and len(bryst_buf.dialogue_history) == 1,
          detail=f"got {len(bryst_buf.dialogue_history) if bryst_buf else 0}")

    if bryst_buf and bryst_buf.dialogue_history:
        entry = bryst_buf.dialogue_history[0]
        check("dialogue_history entry role == 'assistant'",
              entry["role"] == "assistant",
              detail=f"got {entry['role']!r}")
        check("dialogue_history content is clean speech (no JSON braces)",
              entry["content"] == "You think I do this for my health? Come back when you've got the coin.",
              detail=f"got {entry['content'][:60]!r}...")

    group = accumulator._group_memory.verbatim
    check("group_memory.verbatim has 1 entry", len(group) == 1)
    if group:
        check("group entry role == 'Bryst'",
              group[0]["role"] == "Bryst",
              detail=f"got {group[0]['role']!r}")
        check("group entry content is clean speech",
              group[0]["content"] == "You think I do this for my health? Come back when you've got the coin.")

    # ------------------------------------------------------------------
    # 3. Fact pool: stored fact content should be clean text, not JSON
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("3. FactPool: _speech fact stores clean text, not JSON envelope")
    print("=" * 70)

    all_facts = list(fact_pool._facts.values())
    speech_facts = [f for f in all_facts if f.category == "speech"]
    check("at least one speech fact exists", len(speech_facts) >= 1)
    if speech_facts:
        f = speech_facts[0]
        print(f"  fact content: {f.content!r}")
        check("fact content has no JSON braces",
              "{" not in f.content,
              detail="raw_data leaked into fact" if "{" in f.content else "clean")
        check("fact content matches speech field",
              f.content == "You think I do this for my health? Come back when you've got the coin.")

    # ------------------------------------------------------------------
    # 4. Negative case: PLAYER speech should NOT adopt into NPC history
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("4. Negative: player as `speaker` in _speech does NOT pollute history")
    print("=" * 70)

    fact_pool2 = FactPool()
    accumulator2 = EventAccumulator(fact_pool=fact_pool2)
    player_payload = json.dumps({
        "listener": "Bryst",
        "speaker": "Ken",
        "speech": "Ken: I'd like to hire your boat.",
        "location": " outdoors ,Hold: Falkreath",
        "companions": ["Bryst", "Deem-Ja"],
    })
    player_parsed = parse_speech(player_payload)
    typed_player = TypedEvent(
        event_type="_speech",
        local_ts="2026-06-09T18:00:00",
        game_ts=4696802.0,
        raw_data=player_payload,
        parsed_data=player_parsed,
    )
    accumulator2.ingest(TickPackage(
        events=[typed_player],
        tick_interval_ms=2000,
        active_npc_ids=["Bryst", "Deem-Ja"],
    ))

    ken_buf = accumulator2._agent_buffers.get("Ken")
    check("Ken buffer (created via _extract_agent_id)",
          ken_buf is not None,
          detail="speaker routing still works for non-NPC speakers")
    check("Ken (player) NOT adopted into dialogue_history",
          ken_buf is not None and len(ken_buf.dialogue_history) == 0,
          detail="player speech goes through inputtext path instead")

    # ------------------------------------------------------------------
    # 5. emotional_delta: process_inbound uses parsed_data['speech'], not JSON
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print("5. emotional_delta.process_inbound: embeds clean speech text")
    print("=" * 70)

    # Build a TurnContext-shaped object (we only need the bits process_inbound reads)
    from progeny.src.event_accumulator import AgentBuffer, TurnContext
    from mindcore.harmonic_buffer import HarmonicState

    # Just verify the branch the code takes \u2014 we don't need to actually run the
    # full embed (which requires model load). Mock by patching process_texts.
    captured_pairs: list[tuple[str, str]] = []

    from progeny.src import emotional_delta as ed_mod
    original_process_texts = ed_mod.process_texts

    def fake_process_texts(pairs, harmonic_state):
        captured_pairs.extend(pairs)
        return {agent_id: None for agent_id, _ in pairs}

    ed_mod.process_texts = fake_process_texts
    try:
        buf = AgentBuffer(agent_id="Bryst")
        buf.events = [typed_event]  # the canned Bryst event from step 1
        ctx = TurnContext(
            player_input="",
            agent_buffers={"Bryst": buf},
            active_npc_ids=["Bryst"],
            world_events=[],
            session_events=[],
        )
        ed_mod.process_inbound(ctx, HarmonicState())
    finally:
        ed_mod.process_texts = original_process_texts

    print(f"  captured embed pairs: {captured_pairs}")
    check("exactly one (Bryst, text) pair captured", len(captured_pairs) == 1)
    if captured_pairs:
        agent_id, text = captured_pairs[0]
        check("agent_id == 'Bryst'", agent_id == "Bryst")
        check("embedded text is clean speech (no JSON, no audios path)",
              text == "You think I do this for my health? Come back when you've got the coin.",
              detail=f"text preview: {text[:60]!r}")
        check("'audios' substring NOT in embedded text",
              "audios" not in text,
              detail="JSON keys would otherwise leak into the semagram")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"RESULT: {ok_count} passed, {fail_count} failed")
    print("=" * 70)
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
