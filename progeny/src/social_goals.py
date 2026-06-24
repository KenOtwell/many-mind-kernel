"""
Social goals — the "get acquainted" attractor (Phase 6d, Social Bootstrapping).

6a-6c gave each NPC an identity, a notion of who is a stranger, and a valence
(warmth vs wariness) toward the people in its space. 6d turns that into a
*motive*: a seed social goal, "get acquainted", that reuses the Phase 1-2 goal
lifecycle (candidate -> committed -> satisfied) rather than a scripted greeting.

The goal node is an ordinary resonance attractor (trigger cues + a mild-
curiosity affect signature + a first-person recall line), but its activation and
resolution are driven by the social state, not by semantic resonance alone:

  * activation: a co-present stranger (acquaintance.strangers_of) raises the
    goal, gated by valence — warmth promotes the approach, wariness suppresses
    it (the NPC does not dare introduce itself to someone it distrusts). This is
    the lateral competition with avoidance, expressed as activation magnitude.
  * success: ``acquainted()`` — once the agent holds beliefs about the formerly
    unknown person (6e propagates them) the stranger leaves the ledger and the
    goal is satisfied; the tension-to-resolution curve closes.

The prior-versus-individual affect gap (class says wary, this individual's own
memories say safe, or vice-versa) is surfaced for the dissonance term so an
expectation-violating person becomes the most cognitively interesting one in the
room. Everything here is server-side; the goal reaches the prompt only as affect
(the curiosity nudge applied by priming) plus the recalled goal statement —
never an imperative to introduce.
"""
from __future__ import annotations

import logging

from progeny.src import qdrant_client as progeny_qdrant
from progeny.src import valence
from progeny.src.acquaintance import strangers_of
from progeny.src.goal_pool import OWNER_ALL, GoalNode, GoalState, GoalPool, vectorize_node

logger = logging.getLogger(__name__)

GET_ACQUAINTED_NAME = "get_acquainted"

# Base activation contributed by a co-present stranger before valence gating.
# Above CANDIDATE_THRESHOLD (0.20) so a neutral stranger becomes a candidate;
# warmth can lift it past COMMIT_THRESHOLD (0.35) to commit, while wariness
# drops it below candidacy (the approach is suppressed).
BASE_SOCIAL_ACTIVATION: float = 0.30
# Valence gate bounds: wariness can fully suppress, warmth can lift by half.
GATE_MIN: float = 0.0
GATE_MAX: float = 1.5


def get_acquainted_node(owner: str = OWNER_ALL) -> GoalNode:
    """The seed 'get acquainted' social goal (a resonance attractor).

    success_predicate ``acquainted()`` resolves the goal once the agent no
    longer regards anyone co-present as a stranger (6e closes the loop).
    """
    return GoalNode(
        name=GET_ACQUAINTED_NAME,
        statement="There's someone here I don't know. Perhaps I should introduce myself.",
        trigger_cues=(
            "a stranger, someone I have not met, an unfamiliar face, "
            "meeting someone new, an introduction"
        ),
        affect_signature="mild curiosity and uncertainty about meeting someone new",
        role="goal",
        owner=owner,
        state=GoalState.PRIMED,
        base_weight=0.45,
        success_predicate="acquainted()",
    )


def social_goal_id(owner: str = OWNER_ALL) -> str:
    """Deterministic goal id of the get-acquainted node (for activation injection)."""
    return get_acquainted_node(owner).goal_id


def present_strangers(observer: str, present_ids: set[str]) -> set[str]:
    """Co-present NPCs the observer currently regards as strangers (6b ledger)."""
    return {s for s in strangers_of(observer) if s in present_ids}


def _warmth_gate(warmth: float) -> float:
    """Map effective valence to an activation gate: wary->0, neutral->1, warm->up to 1.5."""
    return max(GATE_MIN, min(GATE_MAX, 1.0 + warmth))


def social_activation(
    observer: str,
    strangers: set[str],
    base: float = BASE_SOCIAL_ACTIVATION,
) -> float:
    """Activation of the get-acquainted goal from co-present strangers.

    Zero when no stranger is present. Otherwise ``base`` gated by the mean
    effective valence toward the present strangers: warmth (>0) promotes,
    wariness (<0) suppresses (down to zero), neutral leaves the base curiosity.
    """
    if not strangers:
        return 0.0
    snaps = [valence.social_toward(observer, s) for s in strangers]
    warmths = [s.effective for s in snaps if s is not None]
    warmth = sum(warmths) / len(warmths) if warmths else 0.0
    return base * _warmth_gate(warmth)


def affect_gap(observer: str, present_subjects: set[str]) -> float:
    """Largest prior-vs-individual affect gap toward any co-present person.

    The gap (class prior disagreeing with this individual's own memories) is the
    salience signal fed into dissonance — expectation violation deserves
    attention. Zero when no gap has been measured.
    """
    snaps = [valence.social_toward(observer, s) for s in present_subjects]
    gaps = [s.affect_gap for s in snaps if s is not None]
    return max(gaps) if gaps else 0.0


async def seed_social_goals(pool: GoalPool, owner: str = OWNER_ALL) -> int:
    """Register + persist the social goal(s). Mirrors goal_pool.seed_goals.

    Idempotent (deterministic goal IDs). Registers in-memory regardless; skips
    the Qdrant write when vectors could not be computed (cold start) so the
    lifecycle still works from the injected social activation.
    """
    nodes = [get_acquainted_node(owner)]
    written = 0
    for node in nodes:
        vectorize_node(node)
        pool.register(node)
        if node.semantic_vec and node.emotional_vec:
            await progeny_qdrant.write_goal(
                goal_id=node.goal_id,
                statement=node.statement,
                semantic_vec=node.semantic_vec,
                emotional_vec=node.emotional_vec,
                payload=node.to_payload(),
            )
            written += 1
    logger.info("Seeded %d social goal(s) (owner=%s)", written, owner)
    return written
