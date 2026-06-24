"""
Goal priming by resonance.

Each turn, live percepts query the skyrim_goals collection on both axes
(semantic trigger + emotional affect) and the blended score becomes a goal's
activation. Priming does NOT surface a structured opportunity list. Its output
is consumed server-side: an emotional nudge (curiosity, scaled by activation)
applied to the agent's harmonic buffer, plus a recall hint naming which goal to
bring to mind as ordinary recalled content. Recognition and percept-to-goal
binding then emerge inside the LLM.

See plan: Goal Resonance for Progeny — Phase 1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client.models import FieldCondition, Filter, MatchValue

from shared.constants import COLLECTION_GOALS, EMOTIONAL_DIM
from progeny.src.goal_pool import OWNER_ALL
from progeny.src.qdrant_client import search_vector

logger = logging.getLogger(__name__)

# Curiosity direction in the 9d semagram: excitement (dim 4) + residual
# (dim 8) — the axes Skyrim's Surprised and Puzzled moods already map to. A
# resonance hit pushes the felt state along this direction, which both colours
# the prompt's emotional state and raises curvature so the scheduler pays the
# agent more attention.
EXCITEMENT_AXIS = 4
RESIDUAL_AXIS = 8

# Conversion gains — kept small; a nudge is a lean, not a shove.
CURIOSITY_GAIN = 0.6        # transient spike per unit of leading activation
STANDING_PULL_GAIN = 0.05   # per-tick pull from each active unsatisfied goal

DEFAULT_BROAD_LIMIT = 20
DEFAULT_ACTIVATION_FLOOR = 0.15   # below this, a goal is not considered primed
DEFAULT_RECALL_TOP_K = 1


@dataclass
class GoalActivation:
    """One goal's blended resonance with the current percept frame."""
    goal_id: str
    name: str
    statement: str
    activation: float
    emotional_score: float = 0.0
    semantic_score: float = 0.0


@dataclass
class GoalPrimingResult:
    """Server-side priming output. NOT injected into the prompt as-is.

    nudge is applied to the harmonic buffer; recall is surfaced as ordinary
    recalled content. Deliberately carries no actions — resonance is a trigger,
    not an action.
    """
    agent_id: str
    activations: list[GoalActivation] = field(default_factory=list)
    nudge: list[float] = field(default_factory=lambda: [0.0] * EMOTIONAL_DIM)
    recall: list[str] = field(default_factory=list)

    @property
    def top(self) -> GoalActivation | None:
        return self.activations[0] if self.activations else None


def curiosity_direction() -> list[float]:
    """Unit-ish 9d direction representing curiosity (excitement + residual)."""
    d = [0.0] * EMOTIONAL_DIM
    d[EXCITEMENT_AXIS] = 1.0
    d[RESIDUAL_AXIS] = 1.0
    return d


def _owner_filter(agent_id: str) -> Filter:
    """Match goals owned by this agent OR shared by all (OWNER_ALL)."""
    return Filter(should=[
        FieldCondition(key="owner", match=MatchValue(value=agent_id)),
        FieldCondition(key="owner", match=MatchValue(value=OWNER_ALL)),
    ])


async def prime_goals(
    agent_id: str,
    semantic_query: list[float],
    emotional_query: list[float],
    lambda_t: float = 0.5,
    broad_limit: int = DEFAULT_BROAD_LIMIT,
    activation_floor: float = DEFAULT_ACTIVATION_FLOOR,
    recall_top_k: int = DEFAULT_RECALL_TOP_K,
) -> GoalPrimingResult:
    """Resonance-prime one agent's goals against the current percept frame.

    Queries both vector axes, blends with lambda(t) (emotion vs. affordance),
    scales by each goal's base weight, and returns activations plus a curiosity
    nudge and recall hints for the strongest matches. Reads everything from the
    goal payloads, so it does not need the in-memory pool.
    """
    owner_filter = _owner_filter(agent_id)
    emotional_hits = await search_vector(
        collection=COLLECTION_GOALS, vector_name="emotional",
        query=emotional_query, limit=broad_limit, query_filter=owner_filter,
    )
    semantic_hits = await search_vector(
        collection=COLLECTION_GOALS, vector_name="semantic",
        query=semantic_query, limit=broad_limit, query_filter=owner_filter,
    )

    activations = _blend(emotional_hits, semantic_hits, lambda_t)
    result = GoalPrimingResult(agent_id=agent_id, activations=activations)
    if not activations:
        return result

    top = activations[0]
    if top.activation >= activation_floor:
        # Transient curiosity spike, scaled by the leading activation.
        gain = CURIOSITY_GAIN * top.activation
        result.nudge = [v * gain for v in curiosity_direction()]
        # Recall hints: the strongest goals, as plain recalled statements.
        result.recall = [
            a.statement for a in activations[:recall_top_k]
            if a.activation >= activation_floor and a.statement
        ]
    return result


def _blend(
    emotional_hits: list[dict],
    semantic_hits: list[dict],
    lambda_t: float,
) -> list[GoalActivation]:
    """Blend per goal: activation = (lambda*emo + (1-lambda)*sem) * base_weight.

    Mirrors memory_retrieval._merge_and_score but scales by the goal's base
    weight and sorts descending. Deduplicates by point ID.
    """
    by_id: dict[str, GoalActivation] = {}
    complement = 1.0 - lambda_t

    for hit in emotional_hits:
        pid = hit["id"]
        payload = hit.get("payload", {})
        by_id[pid] = GoalActivation(
            goal_id=pid,
            name=payload.get("name", ""),
            statement=payload.get("statement", ""),
            activation=lambda_t * hit["score"],
            emotional_score=hit["score"],
        )

    for hit in semantic_hits:
        pid = hit["id"]
        payload = hit.get("payload", {})
        contrib = complement * hit["score"]
        if pid in by_id:
            by_id[pid].semantic_score = hit["score"]
            by_id[pid].activation += contrib
        else:
            by_id[pid] = GoalActivation(
                goal_id=pid,
                name=payload.get("name", ""),
                statement=payload.get("statement", ""),
                activation=contrib,
                semantic_score=hit["score"],
            )

    # Scale each blended score by the goal's base weight (captured from
    # whichever axis payload carried it).
    weights: dict[str, float] = {}
    for hit in (*emotional_hits, *semantic_hits):
        weights[hit["id"]] = float(hit.get("payload", {}).get("base_weight", 1.0))
    for pid, act in by_id.items():
        act.activation *= weights.get(pid, 1.0)

    results = list(by_id.values())
    results.sort(key=lambda a: a.activation, reverse=True)
    return results
