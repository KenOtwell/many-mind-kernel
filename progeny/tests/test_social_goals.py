"""Tests for the get-acquainted social goal (Phase 6d).

Mostly pure: the node, the valence-gated social activation, the prior-vs-
individual affect gap, the social predicates, the candidate->committed->
satisfied lifecycle driven through the real Phase 1-2 machinery, and the
dissonance affect-gap term.
"""
from __future__ import annotations

import pytest

from progeny.src import acquaintance
from progeny.src import social_goals
from progeny.src import valence
from progeny.src.goal_lifecycle import (
    LifecycleStore,
    PerceptView,
    compute_dissonance,
    evaluate_predicate,
    update_lifecycle,
)
from progeny.src.goal_pool import OWNER_ALL, GoalState


@pytest.fixture(autouse=True)
def _clean_state():
    """Each test starts with empty stranger ledger + valence snapshots."""
    acquaintance.clear()
    valence.reset()
    yield
    acquaintance.clear()
    valence.reset()


# ---------------------------------------------------------------------------
# The node
# ---------------------------------------------------------------------------

class TestGetAcquaintedNode:
    def test_node_fields(self):
        node = social_goals.get_acquainted_node()
        assert node.name == social_goals.GET_ACQUAINTED_NAME
        assert node.owner == OWNER_ALL
        assert node.role == "goal"
        assert node.success_predicate == "acquainted()"
        assert node.statement  # a first-person recall line

    def test_social_goal_id_is_deterministic(self):
        assert social_goals.social_goal_id() == social_goals.get_acquainted_node().goal_id


# ---------------------------------------------------------------------------
# present_strangers — intersect the ledger with who is here
# ---------------------------------------------------------------------------

class TestPresentStrangers:
    def test_intersects_ledger_with_present(self):
        acquaintance.record_stranger("Mara", "Bjorn")
        present = {"Mara", "Bjorn", "Ralof"}
        assert social_goals.present_strangers("Mara", present) == {"Bjorn"}

    def test_absent_stranger_excluded(self):
        acquaintance.record_stranger("Mara", "Bjorn")
        assert social_goals.present_strangers("Mara", {"Mara", "Ralof"}) == set()

    def test_none_when_no_strangers(self):
        assert social_goals.present_strangers("Mara", {"Mara", "Bjorn"}) == set()


# ---------------------------------------------------------------------------
# social_activation — valence gates approach
# ---------------------------------------------------------------------------

class TestSocialActivation:
    def test_no_strangers_is_zero(self):
        assert social_goals.social_activation("Mara", set()) == 0.0

    def test_neutral_stranger_yields_base_curiosity(self):
        # No valence snapshot -> neutral -> base activation (above candidacy).
        act = social_goals.social_activation("Mara", {"Bjorn"})
        assert act == pytest.approx(social_goals.BASE_SOCIAL_ACTIVATION)

    def test_warmth_promotes(self):
        valence.record_social("Mara", "Bjorn", effective=0.5)
        assert social_goals.social_activation("Mara", {"Bjorn"}) > social_goals.BASE_SOCIAL_ACTIVATION

    def test_wariness_suppresses_to_zero(self):
        valence.record_social("Mara", "Bjorn", effective=-1.0)
        assert social_goals.social_activation("Mara", {"Bjorn"}) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# affect_gap — expectation violation as salience
# ---------------------------------------------------------------------------

class TestAffectGap:
    def test_zero_without_snapshots(self):
        assert social_goals.affect_gap("Mara", {"Bjorn"}) == 0.0

    def test_returns_max_gap(self):
        valence.record_social("Mara", "Bjorn", effective=0.0, affect_gap=0.7)
        valence.record_social("Mara", "Ralof", effective=0.0, affect_gap=0.2)
        assert social_goals.affect_gap("Mara", {"Bjorn", "Ralof"}) == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Social predicates on PerceptView
# ---------------------------------------------------------------------------

class TestSocialPredicates:
    def test_has_stranger_and_acquainted(self):
        view = PerceptView(strangers={"Mara": {"Bjorn"}})
        assert view.has_stranger("Mara") is True
        assert evaluate_predicate("has_stranger()", view, "Mara") is True
        assert evaluate_predicate("acquainted()", view, "Mara") is False

    def test_acquainted_when_no_strangers(self):
        view = PerceptView()
        assert view.has_stranger("Mara") is False
        assert evaluate_predicate("acquainted()", view, "Mara") is True


# ---------------------------------------------------------------------------
# Lifecycle: candidate -> committed -> satisfied via the real machinery
# ---------------------------------------------------------------------------

class TestSocialGoalLifecycle:
    def test_stranger_drives_candidate_then_committed_then_satisfied(self):
        node = social_goals.get_acquainted_node()
        owned = [node]
        store = LifecycleStore()
        with_stranger = PerceptView(strangers={"Mara": {"Bjorn"}})

        # Sustained co-presence with a stranger: candidate, then committed.
        update_lifecycle("Mara", owned, store, {node.goal_id: 0.30}, with_stranger)
        assert store.state_of("Mara", node) == GoalState.CANDIDATE

        update_lifecycle("Mara", owned, store, {node.goal_id: 0.45}, with_stranger)
        update_lifecycle("Mara", owned, store, {node.goal_id: 0.45}, with_stranger)
        assert store.state_of("Mara", node) == GoalState.COMMITTED

        # Acquaintance established (no co-present strangers) -> satisfied.
        acquainted = PerceptView()
        update_lifecycle("Mara", owned, store, {node.goal_id: 0.45}, acquainted)
        assert store.state_of("Mara", node) == GoalState.SATISFIED

    def test_wariness_keeps_goal_below_candidacy(self):
        # A wary disposition suppresses the social activation below the
        # candidate threshold, so the introduction never fires.
        valence.record_social("Mara", "Bjorn", effective=-1.0)
        act = social_goals.social_activation("Mara", {"Bjorn"})
        node = social_goals.get_acquainted_node()
        store = LifecycleStore()
        update_lifecycle(
            "Mara", [node], store, {node.goal_id: act},
            PerceptView(strangers={"Mara": {"Bjorn"}}),
        )
        assert store.state_of("Mara", node) == GoalState.PRIMED


# ---------------------------------------------------------------------------
# Dissonance affect-gap term
# ---------------------------------------------------------------------------

class TestDissonanceAffectGap:
    def test_affect_gap_raises_dissonance(self):
        store = LifecycleStore()
        view = PerceptView()
        base = compute_dissonance("Mara", [], store, view)
        gapped = compute_dissonance("Mara", [], store, view, affect_gap=0.8)
        assert gapped > base

    def test_default_affect_gap_is_backward_compatible(self):
        store = LifecycleStore()
        view = PerceptView()
        assert compute_dissonance("Mara", [], store, view) == 0.0


# ---------------------------------------------------------------------------
# Seeding (no embedding model: registers in-memory, skips Qdrant write)
# ---------------------------------------------------------------------------

class TestSeeding:
    async def test_seed_registers_node_in_pool(self):
        from progeny.src.goal_pool import GoalPool

        pool = GoalPool()
        await social_goals.seed_social_goals(pool)
        assert pool.get(social_goals.social_goal_id()) is not None
