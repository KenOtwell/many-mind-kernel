"""Tests for the valence-conditioned approach (Phase 6c).

Layers:
  - Pure unit tests for the warmth/wariness projection, valence extraction,
    the hysteretic referent-precedence blend, and approachability.
  - Retrieval tests against in-memory Qdrant (hand-made vectors, no model)
    for percept_cued_valence.
  - A routes-level integration test (embedding monkeypatched) asserting that
    warmth promotes and wariness suppresses the approach nudge.
"""
from __future__ import annotations

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient

import progeny.src.qdrant_client as client_mod
from progeny.src.qdrant_client import ensure_collections
from progeny.src.memory_writer import MemoryWriter
from progeny.src import valence
from progeny.src.goal_priming import EXCITEMENT_AXIS
from shared.constants import EMOTIONAL_DIM, SEMANTIC_DIM

FEAR_AXIS = 0
SAFETY_AXIS = 7


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

@pytest.fixture
async def qdrant():
    mem = AsyncQdrantClient(location=":memory:")
    client_mod.configure(mem)
    await ensure_collections()
    yield mem
    client_mod.configure(None)


@pytest.fixture
def writer():
    return MemoryWriter()


@pytest.fixture(autouse=True)
def _clear_valence_ledger():
    """Each test starts with a clean consolidated-prior ledger."""
    valence.reset()
    yield
    valence.reset()


def _sem(val: float = 0.1) -> list[float]:
    v = [val] * SEMANTIC_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo(val: float = 0.3) -> list[float]:
    return [val] * EMOTIONAL_DIM


def _emo_warm(scale: float = 0.6) -> list[float]:
    """Warm reaction: love + joy + safety."""
    v = [0.0] * EMOTIONAL_DIM
    v[2] = scale   # love
    v[6] = scale   # joy
    v[SAFETY_AXIS] = scale
    return v


def _emo_wary(scale: float = 0.6) -> list[float]:
    """Wary reaction: fear + anger + disgust."""
    v = [0.0] * EMOTIONAL_DIM
    v[FEAR_AXIS] = scale
    v[1] = scale   # anger
    v[3] = scale   # disgust
    return v


def _reading(**kwargs) -> valence.ValenceReading:
    base = dict(observer="Mara", subject="Bjorn")
    base.update(kwargs)
    return valence.ValenceReading(**base)


# ---------------------------------------------------------------------------
# Pure: warmth projection
# ---------------------------------------------------------------------------

class TestWarmth:
    def test_positive_axes_read_warm(self):
        assert valence.warmth(_emo_warm()) > 0

    def test_negative_axes_read_wary(self):
        assert valence.warmth(_emo_wary()) < 0

    def test_neutral_is_zero(self):
        assert valence.warmth([0.0] * EMOTIONAL_DIM) == 0.0

    def test_negative_safety_erodes_warmth(self):
        v = [0.0] * EMOTIONAL_DIM
        v[SAFETY_AXIS] = -0.5  # feeling unsafe
        assert valence.warmth(v) < 0

    def test_components_split_matches_documented_convention(self):
        v = [0.0] * EMOTIONAL_DIM
        v[6] = 0.4   # joy -> positive
        v[0] = 0.1   # fear -> negative
        pos, neg, res = valence.valence_components(v)
        assert pos == pytest.approx(0.4)
        assert neg == pytest.approx(0.1)
        assert res == 0.0

    def test_build_percept_text_sharpens_then_falls_back(self):
        assert valence.build_percept_text("Bjorn", "soldier, nord") == "Bjorn, soldier, nord"
        assert valence.build_percept_text("Bjorn", "") == "Bjorn"
        assert valence.build_percept_text("", "soldier") == "soldier"


# ---------------------------------------------------------------------------
# Pure: extraction
# ---------------------------------------------------------------------------

class TestExtractValence:
    def test_warm_individual_memories_yield_positive_individual(self):
        contribs = [
            valence.MemoryAffect("a", weight=1.0, warmth=1.2, is_individual=True),
            valence.MemoryAffect("b", weight=1.0, warmth=0.8, is_individual=True),
        ]
        reading = valence.extract_valence("Mara", "Bjorn", contribs)
        assert reading.individual > 0
        assert reading.has_individual
        assert reading.individual_support == 2
        assert reading.general_support == 0

    def test_individual_and_general_are_separated(self):
        contribs = [
            valence.MemoryAffect("a", weight=1.0, warmth=1.0, is_individual=True),
            valence.MemoryAffect("b", weight=1.0, warmth=-1.0, is_individual=False),
        ]
        reading = valence.extract_valence("Mara", "Bjorn", contribs)
        assert reading.individual > 0
        assert reading.general < 0

    def test_weight_dominates_sign(self):
        contribs = [
            valence.MemoryAffect("a", weight=3.0, warmth=-1.0, is_individual=False),
            valence.MemoryAffect("b", weight=0.1, warmth=1.0, is_individual=False),
        ]
        reading = valence.extract_valence("Mara", "Bjorn", contribs)
        assert reading.general < 0  # the heavily-weighted wary memory wins

    def test_empty_is_neutral_no_signal(self):
        reading = valence.extract_valence("Mara", "Bjorn", [])
        assert not reading.has_signal
        assert reading.individual == 0.0 and reading.general == 0.0


# ---------------------------------------------------------------------------
# Pure: hysteretic referent-precedence blend
# ---------------------------------------------------------------------------

class TestHysteresisBlend:
    def test_first_meeting_is_pure_prior(self):
        reading = _reading(general=-0.4, general_support=2)
        blended = valence.blend_valence(reading)
        assert blended.effective == pytest.approx(-0.4)
        assert blended.individual_weight == 0.0
        assert blended.encounters == 0

    def test_one_kind_encounter_does_not_flip_wary_prior(self):
        reading = _reading(general=-1.0, individual=1.0, general_support=3, individual_support=1)
        blended = valence.blend_valence(reading)
        assert blended.is_wary  # still wary after a single kind memory
        assert blended.individual_weight == pytest.approx(0.25)

    def test_individual_emerges_from_class_shadow_over_encounters(self):
        def kind():
            return _reading(general=-1.0, individual=1.0, general_support=3, individual_support=1)

        results = [valence.blend_valence(kind()) for _ in range(6)]
        assert results[0].is_wary           # n=1 wary
        assert results[-1].is_warm          # by n=6 warm
        # Monotonic non-decreasing override weight as evidence accumulates.
        weights = [r.individual_weight for r in results]
        assert weights == sorted(weights)

    def test_weight_is_saturating(self):
        assert valence.individual_weight(0) == 0.0
        assert valence.individual_weight(3) == pytest.approx(0.5)  # k=3
        assert valence.individual_weight(1000) < 1.0
        assert valence.individual_weight(1000) > 0.99

    def test_no_update_leaves_ledger_untouched(self):
        reading = _reading(general=0.0, individual=1.0, individual_support=1)
        valence.blend_valence(reading, update=False)
        assert valence.peek_prior("Mara", "Bjorn") == (0.0, 0)


# ---------------------------------------------------------------------------
# Pure: approachability
# ---------------------------------------------------------------------------

class TestApproachability:
    def test_default_is_neutral(self):
        assert valence.approachability(None) == 1.0
        assert valence.approachability([]) == 1.0

    def test_warm_traits_raise_and_wary_lower(self):
        assert valence.approachability(["friendly", "outgoing"]) > 1.0
        assert valence.approachability(["suspicious", "gruff"]) < 1.0

    def test_tone_is_considered(self):
        assert valence.approachability(None, tone="warm and welcoming") > 1.0

    def test_clamped(self):
        many = ["friendly"] * 50
        assert valence.approachability(many) <= valence.APPROACHABILITY_MAX


# ---------------------------------------------------------------------------
# Retrieval: percept_cued_valence (in-memory Qdrant, no model)
# ---------------------------------------------------------------------------

class TestPerceptCuedValence:
    async def test_warm_individual_memories_yield_warmth(self, qdrant, writer):
        obs, subj = "Mara", "Bjorn"
        for i, content in enumerate(("Bjorn shared his bread.", "Bjorn mended my fence.")):
            await writer.write_raw_event(
                agent_id=obs, content=content, semantic_vector=_sem(),
                emotional_vector=_emo_warm(), game_ts=float(i), event_type="event",
                referents=[subj],
            )
        reading = await valence.percept_cued_valence(obs, subj, _sem(), _emo_warm())
        assert reading.has_individual
        assert reading.individual > 0
        assert reading.recall  # the warm episodes are surfaced as recall

    async def test_wary_individual_memories_yield_wariness(self, qdrant, writer):
        obs, subj = "Mara", "Bjorn"
        await writer.write_raw_event(
            agent_id=obs, content="Bjorn threatened me at the gate.", semantic_vector=_sem(),
            emotional_vector=_emo_wary(), game_ts=1.0, event_type="event", referents=[subj],
        )
        reading = await valence.percept_cued_valence(obs, subj, _sem(), _emo_wary())
        assert reading.has_individual
        assert reading.individual < 0

    async def test_general_class_memory_is_prior_not_individual(self, qdrant, writer):
        obs = "Mara"
        # A wary memory about a DIFFERENT soldier — generalizes to a new one.
        await writer.write_raw_event(
            agent_id=obs, content="A soldier named Ralof robbed me.", semantic_vector=_sem(),
            emotional_vector=_emo_wary(), game_ts=1.0, event_type="event", referents=["Ralof"],
        )
        reading = await valence.percept_cued_valence(obs, "Bjorn", _sem(), _emo_wary())
        assert not reading.has_individual
        assert reading.general_support >= 1
        assert reading.general < 0

    async def test_no_memories_returns_empty_reading(self, qdrant):
        reading = await valence.percept_cued_valence("Mara", "Bjorn", _sem(), _emo())
        assert not reading.has_signal
        assert reading.individual == 0.0 and reading.general == 0.0
        assert reading.recall == []


# ---------------------------------------------------------------------------
# Routes integration: warmth promotes, wariness suppresses (embedding stubbed)
# ---------------------------------------------------------------------------

class TestApproachConditioning:
    @staticmethod
    def _stub_embedding(monkeypatch, routes):
        monkeypatch.setattr(routes.shared_embedding, "is_loaded", lambda: True)
        monkeypatch.setattr(
            routes.shared_embedding, "embed_one",
            lambda text: np.asarray(_sem(), dtype=float),
        )
        monkeypatch.setattr(routes.shared_emotional, "is_loaded", lambda: True)

    @staticmethod
    def _init_observer(routes, obs: str) -> None:
        # Two updates so fast != slow → a non-zero deviation query.
        routes._harmonic_state.reset()
        routes._reminding_queue.clear()
        routes._harmonic_state.update(obs, [0.1] * EMOTIONAL_DIM)
        routes._harmonic_state.update(obs, _emo_warm(0.2))

    async def test_warmth_promotes_curiosity(self, qdrant, writer, monkeypatch):
        import progeny.api.routes as routes
        self._stub_embedding(monkeypatch, routes)
        obs, subj = "Mara", "Bjorn"
        self._init_observer(routes, obs)
        await writer.write_raw_event(
            agent_id=obs, content="Bjorn shared his bread.", semantic_vector=_sem(),
            emotional_vector=_emo_warm(), game_ts=1.0, event_type="event", referents=[subj],
        )
        before = routes._harmonic_state.get_semagram(obs)
        await routes._condition_approach_by_valence([subj], [obs, subj])
        after = routes._harmonic_state.get_semagram(obs)
        assert after[EXCITEMENT_AXIS] > before[EXCITEMENT_AXIS]

    async def test_wariness_suppresses_with_guarded_affect(self, qdrant, writer, monkeypatch):
        import progeny.api.routes as routes
        self._stub_embedding(monkeypatch, routes)
        obs, subj = "Mara", "Bjorn"
        self._init_observer(routes, obs)
        await writer.write_raw_event(
            agent_id=obs, content="Bjorn threatened me at the gate.", semantic_vector=_sem(),
            emotional_vector=_emo_wary(), game_ts=1.0, event_type="event", referents=[subj],
        )
        before = routes._harmonic_state.get_semagram(obs)
        await routes._condition_approach_by_valence([subj], [obs, subj])
        after = routes._harmonic_state.get_semagram(obs)
        # Guarded: fear rises rather than curiosity.
        assert after[FEAR_AXIS] > before[FEAR_AXIS]
        routes._harmonic_state.reset()
        routes._reminding_queue.clear()
