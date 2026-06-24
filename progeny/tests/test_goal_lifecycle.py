"""Tests for the defeasible goal lifecycle + dissonance (Phase 2).

Pure unit tests — no Qdrant, no model. GoalNodes are built directly and
PerceptView is constructed from primitives.
"""
from __future__ import annotations

from progeny.src.goal_pool import GoalNode, GoalState
from progeny.src.goal_lifecycle import (
    COMMIT_TICKS,
    LifecycleStore,
    PerceptView,
    compute_dissonance,
    evaluate_predicate,
    update_lifecycle,
)


def _node(name: str, **kw) -> GoalNode:
    return GoalNode(
        name=name,
        statement=kw.pop("statement", name),
        trigger_cues=kw.pop("trigger_cues", name),
        affect_signature=kw.pop("affect_signature", name),
        **kw,
    )


# ---------------------------------------------------------------------------
# Predicate evaluation
# ---------------------------------------------------------------------------

class TestEvaluatePredicate:
    def setup_method(self):
        self.view = PerceptView(
            percept_text="there is a rabbit nearby in the brush",
            equipment={"Lydia": {"Hunting Bow"}},
        )

    def test_perceived_true_and_false(self):
        assert evaluate_predicate("perceived('rabbit')", self.view, "Lydia") is True
        assert evaluate_predicate("perceived('quail')", self.view, "Lydia") is False

    def test_or_and(self):
        assert evaluate_predicate("perceived('rabbit') or perceived('quail')", self.view, "Lydia") is True
        assert evaluate_predicate("perceived('rabbit') and perceived('quail')", self.view, "Lydia") is False

    def test_has_equipped_weapon(self):
        assert evaluate_predicate("has_equipped(category='weapon')", self.view, "Lydia") is True
        assert evaluate_predicate("has_equipped(category='weapon')", self.view, "Belethor") is False

    def test_inventory_has_unobservable_is_false(self):
        # No inventory feed -> conservative False (goal stays open).
        assert evaluate_predicate("inventory_has(item='arrow')", self.view, "Lydia") is False

    def test_empty_and_garbage_are_false(self):
        assert evaluate_predicate("", self.view, "Lydia") is False
        assert evaluate_predicate("nonsense", self.view, "Lydia") is False


# ---------------------------------------------------------------------------
# Defeasible lifecycle transitions
# ---------------------------------------------------------------------------

class TestLifecycleTransitions:
    def test_success_satisfies_then_reverts_when_predicate_retracts(self):
        """Non-monotonic: perceived -> SATISFIED; cue gone -> reopens to PRIMED."""
        node = _node("prey_rabbit", role="candidate", parent="locate_prey",
                     success_predicate="perceived('rabbit')")
        store = LifecycleStore()

        update_lifecycle("Lydia", [node], store, {node.goal_id: 0.0},
                         PerceptView(percept_text="a rabbit darts past"))
        assert store.state_of("Lydia", node) == GoalState.SATISFIED

        update_lifecycle("Lydia", [node], store, {node.goal_id: 0.0},
                         PerceptView(percept_text="an empty meadow"))
        assert store.state_of("Lydia", node) == GoalState.PRIMED

    def test_candidate_then_committed_ramp(self):
        node = _node("hunt", success_predicate="")  # never auto-satisfies
        store = LifecycleStore()
        view = PerceptView()

        update_lifecycle("Lydia", [node], store, {node.goal_id: 0.4}, view)
        assert store.state_of("Lydia", node) == GoalState.CANDIDATE  # lead_ticks=1

        for _ in range(COMMIT_TICKS):
            update_lifecycle("Lydia", [node], store, {node.goal_id: 0.4}, view)
        assert store.state_of("Lydia", node) == GoalState.COMMITTED

    def test_defeasible_drop_to_primed(self):
        node = _node("hunt", success_predicate="")
        store = LifecycleStore()
        view = PerceptView()
        for _ in range(COMMIT_TICKS + 1):
            update_lifecycle("Lydia", [node], store, {node.goal_id: 0.5}, view)
        assert store.state_of("Lydia", node) == GoalState.COMMITTED
        update_lifecycle("Lydia", [node], store, {node.goal_id: 0.0}, view)
        assert store.state_of("Lydia", node) == GoalState.PRIMED

    def test_candidate_threshold_band(self):
        node = _node("hunt", success_predicate="")
        store = LifecycleStore()
        update_lifecycle("Lydia", [node], store, {node.goal_id: 0.25}, PerceptView())
        assert store.state_of("Lydia", node) == GoalState.CANDIDATE


# ---------------------------------------------------------------------------
# Disjunctive sibling lateral inhibition
# ---------------------------------------------------------------------------

class TestLateralInhibition:
    def test_leader_wins_loser_stays_warm_and_present(self):
        rabbit = _node("prey_rabbit", role="candidate", parent="locate_prey",
                       success_predicate="perceived('rabbit')")
        quail = _node("prey_quail", role="candidate", parent="locate_prey",
                      success_predicate="perceived('quail')")
        store = LifecycleStore()
        view = PerceptView(percept_text="an open field")  # neither perceived

        update_lifecycle("Lydia", [rabbit, quail], store,
                         {rabbit.goal_id: 0.4, quail.goal_id: 0.3}, view)

        # Rabbit leads; quail is inhibited back to PRIMED (warm, not deleted).
        assert store.state_of("Lydia", rabbit) in (GoalState.CANDIDATE, GoalState.COMMITTED)
        assert store.state_of("Lydia", quail) == GoalState.PRIMED
        assert store.get("Lydia", rabbit.goal_id) is not None
        assert store.get("Lydia", quail.goal_id) is not None


# ---------------------------------------------------------------------------
# Dissonance
# ---------------------------------------------------------------------------

class TestDissonance:
    def test_unmet_enabler_raises_dissonance(self):
        bow = _node("have_bow", role="enabler",
                    enabler_predicate="has_equipped(category='weapon')")
        store = LifecycleStore()
        store.get_or_create("Lydia", bow)  # active (PRIMED) by default

        unmet = compute_dissonance("Lydia", [bow], store, PerceptView(equipment={}))
        met = compute_dissonance("Lydia", [bow], store,
                                 PerceptView(equipment={"Lydia": {"Iron Sword"}}))
        assert unmet > met

    def test_volatility_raises_dissonance(self):
        store = LifecycleStore()
        calm = compute_dissonance("Lydia", [], store, PerceptView())
        volatile = compute_dissonance(
            "Lydia", [], store, PerceptView(),
            curvature=1.0, snap=1.0, coherence=0.0,
        )
        assert volatile > calm
