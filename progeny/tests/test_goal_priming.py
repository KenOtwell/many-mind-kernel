"""Tests for goal resonance priming (Phase 1).

Hermetic: in-memory Qdrant + hand-made vectors. Does NOT load the
sentence-transformer model — prime_goals takes pre-embedded queries, so the
priming/blend/nudge logic is exercised without a model download.
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

import progeny.src.qdrant_client as client_mod
from progeny.src.qdrant_client import ensure_collections, write_goal
from progeny.src import goal_priming
from progeny.src.goal_priming import EXCITEMENT_AXIS, RESIDUAL_AXIS
from mindcore.harmonic_buffer import HarmonicState
from shared.constants import EMOTIONAL_DIM, SEMANTIC_DIM


@pytest.fixture
async def qdrant():
    mem = AsyncQdrantClient(location=":memory:")
    client_mod.configure(mem)
    await ensure_collections()
    yield mem
    client_mod.configure(None)


def _sem(val: float = 0.1) -> list[float]:
    v = [val] * SEMANTIC_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo(val: float = 0.5) -> list[float]:
    v = [val] * EMOTIONAL_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


async def _seed_hunt_goal(
    owner: str = "all",
    weight: float = 0.6,
    goal_id: str = "00000000-0000-0000-0000-000000000001",
) -> str:
    return await write_goal(
        goal_id=goal_id,
        statement="I set out to hunt game for dinner.",
        semantic_vec=_sem(0.1),
        emotional_vec=_emo(0.5),
        payload={
            "name": "hunt_for_game",
            "role": "goal",
            "owner": owner,
            "state": "primed",
            "base_weight": weight,
        },
    )


class TestPrimeGoals:
    async def test_resonant_percept_activates_goal(self, qdrant):
        await _seed_hunt_goal()
        res = await goal_priming.prime_goals(
            "Lydia", semantic_query=_sem(0.1), emotional_query=_emo(0.5), lambda_t=0.5,
        )
        assert res.top is not None
        assert res.top.name == "hunt_for_game"
        assert res.top.activation > 0.0

    async def test_strong_match_surfaces_recall_and_curiosity_nudge(self, qdrant):
        await _seed_hunt_goal()
        res = await goal_priming.prime_goals(
            "Lydia", semantic_query=_sem(0.1), emotional_query=_emo(0.5), lambda_t=0.5,
        )
        # Recalled content present (unlabeled goal statement).
        assert res.recall and "hunt" in res.recall[0].lower()
        # Curiosity nudge points along excitement + residual only.
        assert res.nudge[EXCITEMENT_AXIS] > 0.0
        assert res.nudge[RESIDUAL_AXIS] > 0.0
        for i, v in enumerate(res.nudge):
            if i not in (EXCITEMENT_AXIS, RESIDUAL_AXIS):
                assert v == 0.0

    async def test_priming_emits_no_actions(self, qdrant):
        """Resonance is a trigger, not an action — the result carries no commands."""
        await _seed_hunt_goal()
        res = await goal_priming.prime_goals(
            "Lydia", semantic_query=_sem(0.1), emotional_query=_emo(0.5),
        )
        assert not hasattr(res, "actions")

    async def test_owner_scoping_excludes_non_owner(self, qdrant):
        await _seed_hunt_goal(owner="Lydia", goal_id="00000000-0000-0000-0000-000000000002")
        # Belethor neither owns the private goal nor is there a shared one.
        belethor = await goal_priming.prime_goals(
            "Belethor", semantic_query=_sem(0.1), emotional_query=_emo(0.5),
        )
        assert belethor.top is None
        # Lydia sees her own goal.
        lydia = await goal_priming.prime_goals(
            "Lydia", semantic_query=_sem(0.1), emotional_query=_emo(0.5),
        )
        assert lydia.top is not None

    async def test_empty_collection_yields_no_activation(self, qdrant):
        res = await goal_priming.prime_goals(
            "Lydia", semantic_query=_sem(0.1), emotional_query=_emo(0.5),
        )
        assert res.activations == []
        assert res.recall == []
        assert not any(res.nudge)


class TestCuriosityNudge:
    async def test_nudge_raises_excitement_and_residual_and_curvature(self):
        hs = HarmonicState()
        # Initialize the agent's felt state with a neutral-ish base.
        hs.update("Lydia", [0.1] * EMOTIONAL_DIM)
        before = hs.get_semagram("Lydia")

        nudge = [0.0] * EMOTIONAL_DIM
        nudge[EXCITEMENT_AXIS] = 0.4
        nudge[RESIDUAL_AXIS] = 0.4
        delta = hs.apply_nudge("Lydia", nudge)
        after = hs.get_semagram("Lydia")

        assert delta is not None
        assert after[EXCITEMENT_AXIS] > before[EXCITEMENT_AXIS]
        assert after[RESIDUAL_AXIS] > before[RESIDUAL_AXIS]
        # Curvature rose, so scheduling would see the salience and could promote.
        assert delta.curvature > 0.0

    async def test_nudge_is_noop_for_unknown_agent(self):
        hs = HarmonicState()
        assert hs.apply_nudge("Ghost", [0.1] * EMOTIONAL_DIM) is None
