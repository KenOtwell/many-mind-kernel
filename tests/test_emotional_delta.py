"""Tests for progeny.src.emotional_delta.

Strategy: monkeypatch embedding to return deterministic unit vectors so
tests are fast and model-free. Real projection bases are loaded — the 9d
math runs for real. HarmonicState is used directly (no mocking needed).
"""
from __future__ import annotations

import numpy as np
import pytest

from shared.constants import EMOTIONAL_DIM
from progeny.src.harmonic_buffer import EmotionalDelta, HarmonicState
from shared.schemas import AgentResponse


# ---------------------------------------------------------------------------
# Module-level fake embedding — a unit vector in R^384
# ---------------------------------------------------------------------------

_FAKE_384 = np.ones(384, dtype=np.float32) / np.sqrt(384)


@pytest.fixture()
def fake_embed(monkeypatch):
    """Patch embedding layer to return deterministic unit vectors.

    All texts map to the same _FAKE_384 vector — direction is stable,
    magnitude is 1. Lets tests focus on pipeline structure and state
    transitions without loading the sentence transformer.
    """
    monkeypatch.setattr(
        "progeny.src.emotional_delta.embedding.embed_one",
        lambda text: _FAKE_384.copy(),
    )
    monkeypatch.setattr(
        "progeny.src.emotional_delta.embedding.embed",
        lambda texts: np.stack([_FAKE_384.copy() for _ in texts]),
    )


@pytest.fixture(autouse=True)
def _load_projection_bases():
    """Ensure the 9d projection bases are loaded for every test in this file."""
    from progeny.src.emotional_projection import load_bases
    load_bases()


# ---------------------------------------------------------------------------
# process_text — the fundamental operation
# ---------------------------------------------------------------------------

class TestProcessText:
    def test_returns_emotional_delta(self, fake_embed):
        from progeny.src.emotional_delta import process_text
        state = HarmonicState()
        delta = process_text("Lydia", "I am sworn to carry your burdens.", state)
        assert isinstance(delta, EmotionalDelta)
        assert len(delta.semagram) == EMOTIONAL_DIM

    def test_creates_agent_buffer_on_first_call(self, fake_embed):
        from progeny.src.emotional_delta import process_text
        state = HarmonicState()
        assert "Lydia" not in state.agent_ids
        process_text("Lydia", "Something happened.", state)
        assert "Lydia" in state.agent_ids

    def test_semagram_nonzero_after_update(self, fake_embed):
        from progeny.src.emotional_delta import process_text
        state = HarmonicState()
        delta = process_text("Lydia", "Something happened.", state)
        assert any(v != 0.0 for v in delta.semagram)

    def test_independent_agents_independent_state(self, fake_embed):
        from progeny.src.emotional_delta import process_text
        state = HarmonicState()
        process_text("Lydia", "Battle cry!", state)
        # Belethor gets no update yet
        assert "Belethor" not in state.agent_ids


# ---------------------------------------------------------------------------
# process_texts — batch path
# ---------------------------------------------------------------------------

class TestProcessTexts:
    def test_empty_pairs_returns_empty_dict(self):
        from progeny.src.emotional_delta import process_texts
        result = process_texts([], HarmonicState())
        assert result == {}

    def test_single_pair_returns_one_delta(self, fake_embed):
        from progeny.src.emotional_delta import process_texts
        state = HarmonicState()
        result = process_texts([("Lydia", "I am ready.")], state)
        assert list(result.keys()) == ["Lydia"]
        assert isinstance(result["Lydia"], EmotionalDelta)

    def test_multiple_agents_each_get_delta(self, fake_embed):
        from progeny.src.emotional_delta import process_texts
        state = HarmonicState()
        pairs = [("Lydia", "I fight!"), ("Belethor", "Do come back.")]
        result = process_texts(pairs, state)
        assert "Lydia" in result
        assert "Belethor" in result

    def test_same_agent_multiple_texts_one_update(self, fake_embed):
        """Multiple texts for the same agent average into one harmonic update."""
        from progeny.src.emotional_delta import process_texts
        state = HarmonicState()
        pairs = [("Lydia", "First text."), ("Lydia", "Second text.")]
        result = process_texts(pairs, state)
        # One result per agent, not per pair
        assert list(result.keys()) == ["Lydia"]

    def test_all_texts_embedded_in_one_batch_call(self, monkeypatch):
        """Efficiency: embed() is called exactly once for all texts."""
        embed_call_sizes: list[int] = []

        def _embed(texts):
            embed_call_sizes.append(len(texts))
            return np.stack([_FAKE_384.copy() for _ in texts])

        monkeypatch.setattr("progeny.src.emotional_delta.embedding.embed", _embed)

        from progeny.src.emotional_delta import process_texts
        pairs = [("Lydia", "text1"), ("Belethor", "text2"), ("Ysolda", "text3")]
        process_texts(pairs, HarmonicState())

        assert len(embed_call_sizes) == 1   # One call
        assert embed_call_sizes[0] == 3     # All 3 texts together

    def test_harmonic_state_updated_in_place(self, fake_embed):
        from progeny.src.emotional_delta import process_texts
        state = HarmonicState()
        process_texts([("Lydia", "text"), ("Belethor", "text")], state)
        assert "Lydia" in state.agent_ids
        assert "Belethor" in state.agent_ids


# ---------------------------------------------------------------------------
# process_inbound — player input + NPC speech events
# ---------------------------------------------------------------------------

def _make_speech_event(speaker: str, text: str):
    """Helper: create a _speech TypedEvent."""
    from shared.schemas import TypedEvent
    return TypedEvent(
        event_type="_speech",
        local_ts="2024-01-01T00:00:00",
        game_ts=100.0,
        raw_data=text,
        parsed_data={"speaker": speaker, "speech": text},
    )


def _make_turn_context(
    player_input: str,
    active_npc_ids: list[str],
    speech_by: dict[str, list[str]] | None = None,
):
    """Build a TurnContext directly without going through the accumulator."""
    from progeny.src.event_accumulator import AgentBuffer, TurnContext

    agent_buffers: dict[str, AgentBuffer] = {}
    for agent_id in active_npc_ids:
        agent_buffers[agent_id] = AgentBuffer(agent_id=agent_id)

    # Inject speech events into the relevant agent's buffer
    for speaker, texts in (speech_by or {}).items():
        if speaker not in agent_buffers:
            agent_buffers[speaker] = AgentBuffer(agent_id=speaker)
        for text in texts:
            agent_buffers[speaker].append(_make_speech_event(speaker, text))

    return TurnContext(
        player_input=player_input,
        agent_buffers=agent_buffers,
        active_npc_ids=active_npc_ids,
        world_events=[],
        session_events=[],
    )


class TestProcessInbound:
    def test_player_input_updates_all_active_agents(self, fake_embed):
        from progeny.src.emotional_delta import process_inbound
        ctx = _make_turn_context("Hello!", ["Lydia", "Belethor", "Ysolda"])
        result = process_inbound(ctx, HarmonicState())
        assert "Lydia" in result
        assert "Belethor" in result
        assert "Ysolda" in result

    def test_player_input_does_not_affect_inactive_agents(self, fake_embed):
        """Agents not in active_npc_ids don't receive the player's words."""
        from progeny.src.emotional_delta import process_inbound
        ctx = _make_turn_context("Hello!", active_npc_ids=["Lydia"])
        state = HarmonicState()
        process_inbound(ctx, state)
        assert "Belethor" not in state.agent_ids

    def test_empty_active_npc_ids_no_player_pairs(self, fake_embed):
        from progeny.src.emotional_delta import process_inbound
        ctx = _make_turn_context("Hello!", active_npc_ids=[])
        result = process_inbound(ctx, HarmonicState())
        assert result == {}

    def test_npc_speech_affects_speaker(self, fake_embed):
        """NPC speech shifts the speaking agent, not bystanders."""
        from progeny.src.emotional_delta import process_inbound
        # Only Lydia speaks; active_npc_ids is empty (player input won't fire)
        ctx = _make_turn_context(
            player_input="",
            active_npc_ids=[],
            speech_by={"Lydia": ["I am sworn to carry your burdens."]},
        )
        state = HarmonicState()
        result = process_inbound(ctx, state)
        assert "Lydia" in result
        assert "Belethor" not in state.agent_ids

    def test_npc_speech_and_player_input_both_processed(self, fake_embed):
        """Lydia hears the player AND has her own speech — both contribute."""
        from progeny.src.emotional_delta import process_inbound
        ctx = _make_turn_context(
            player_input="Watch out!",
            active_npc_ids=["Lydia"],
            speech_by={"Lydia": ["I'll handle this!"]},
        )
        result = process_inbound(ctx, HarmonicState())
        assert "Lydia" in result

    def test_empty_player_input_still_processes_speech(self, fake_embed):
        """Empty player input is falsy — only NPC speech contributes."""
        from progeny.src.emotional_delta import process_inbound
        ctx = _make_turn_context(
            player_input="",
            active_npc_ids=[],
            speech_by={"Belethor": ["Have you seen the new vendor?"]},
        )
        result = process_inbound(ctx, HarmonicState())
        assert "Belethor" in result

    def test_non_speech_events_in_buffer_ignored(self, fake_embed):
        """Info events in agent buffer don't contribute to the pipeline."""
        from progeny.src.emotional_delta import process_inbound
        from progeny.src.event_accumulator import AgentBuffer, TurnContext
        from shared.schemas import TypedEvent

        info_event = TypedEvent(
            event_type="info",
            local_ts="2024-01-01",
            game_ts=100.0,
            raw_data="Lydia drew her weapon",
            parsed_data=None,
        )
        lydia_buf = AgentBuffer(agent_id="Lydia")
        lydia_buf.append(info_event)

        ctx = TurnContext(
            player_input="",
            agent_buffers={"Lydia": lydia_buf},
            active_npc_ids=[],
            world_events=[],
            session_events=[],
        )
        result = process_inbound(ctx, HarmonicState())
        # No _speech event and no player input → nothing processed
        assert result == {}


# ---------------------------------------------------------------------------
# process_outbound — LLM-generated utterances
# ---------------------------------------------------------------------------

class TestProcessOutbound:
    def test_empty_responses_returns_empty(self):
        from progeny.src.emotional_delta import process_outbound
        result = process_outbound([], HarmonicState())
        assert result == {}

    def test_response_without_utterance_skipped(self, fake_embed):
        from progeny.src.emotional_delta import process_outbound
        responses = [AgentResponse(agent_id="Lydia", utterance=None)]
        result = process_outbound(responses, HarmonicState())
        assert result == {}

    def test_utterance_updates_speaking_agent(self, fake_embed):
        from progeny.src.emotional_delta import process_outbound
        state = HarmonicState()
        responses = [AgentResponse(
            agent_id="Lydia",
            utterance="I am sworn to carry your burdens.",
        )]
        result = process_outbound(responses, state)
        assert "Lydia" in result
        assert isinstance(result["Lydia"], EmotionalDelta)
        assert "Lydia" in state.agent_ids

    def test_multiple_agents_all_updated(self, fake_embed):
        from progeny.src.emotional_delta import process_outbound
        responses = [
            AgentResponse(agent_id="Lydia", utterance="I'll protect you."),
            AgentResponse(agent_id="Belethor", utterance="Do come back."),
        ]
        result = process_outbound(responses, HarmonicState())
        assert "Lydia" in result
        assert "Belethor" in result

    def test_mix_of_utterance_and_silent_agents(self, fake_embed):
        """Silent agents (no utterance) are skipped; speakers are updated."""
        from progeny.src.emotional_delta import process_outbound
        state = HarmonicState()
        responses = [
            AgentResponse(agent_id="Lydia", utterance="Stand your ground!"),
            AgentResponse(agent_id="Ysolda", utterance=None),   # silent
        ]
        result = process_outbound(responses, state)
        assert "Lydia" in result
        assert "Ysolda" not in result
        assert "Ysolda" not in state.agent_ids


# ---------------------------------------------------------------------------
# Bidirectionality — the core architectural property
# ---------------------------------------------------------------------------

class TestBidirectionality:
    def test_same_text_same_semagram_in_both_directions(self, monkeypatch):
        """The same text produces the same semagram whether inbound or outbound.

        This is the bidirectionality guarantee: the pipeline doesn't care
        about direction. Text is text. Feeling is feeling.
        """
        monkeypatch.setattr(
            "progeny.src.emotional_delta.embedding.embed_one",
            lambda text: _FAKE_384.copy(),
        )
        monkeypatch.setattr(
            "progeny.src.emotional_delta.embedding.embed",
            lambda texts: np.stack([_FAKE_384.copy() for _ in texts]),
        )

        from progeny.src.emotional_delta import process_text, process_outbound

        TEXT = "I am sworn to carry your burdens."

        # Inbound path: direct process_text call
        state_in = HarmonicState()
        delta_in = process_text("Lydia", TEXT, state_in)

        # Outbound path: LLM response utterance
        state_out = HarmonicState()
        delta_out = process_outbound(
            [AgentResponse(agent_id="Lydia", utterance=TEXT)],
            state_out,
        )

        np.testing.assert_allclose(
            delta_in.semagram, delta_out["Lydia"].semagram, atol=1e-5,
            err_msg="Same text must produce same semagram regardless of pipeline direction",
        )

    def test_outbound_shifts_state_as_inbound_does(self, fake_embed):
        """After outbound update, the agent's semagram should be non-zero."""
        from progeny.src.emotional_delta import process_outbound
        state = HarmonicState()
        process_outbound(
            [AgentResponse(agent_id="Lydia", utterance="Something important.")],
            state,
        )
        sem = state.get_semagram("Lydia")
        assert any(v != 0.0 for v in sem), "Outbound utterance should shift semagram"
