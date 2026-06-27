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

from qdrant_client.models import FieldCondition, Filter, MatchValue

from shared.constants import COLLECTION_GOALS, EMOTIONAL_DIM
from progeny.src.goal_pool import OWNER_ALL
from progeny.src.qdrant_client import search_vector
from mindcore.goal import (
    GoalActivation, GoalPrimingResult, prime_from_hits,
    curiosity_direction,                              # re-exported for routes.py
    CURIOSITY_GAIN, STANDING_PULL_GAIN,
    DEFAULT_ACTIVATION_FLOOR, DEFAULT_RECALL_TOP_K, DEFAULT_BROAD_LIMIT,
)

logger = logging.getLogger(__name__)


# Re-exported from mindcore for backwards-compat with any existing imports.
__all__ = ["GoalActivation", "GoalPrimingResult", "prime_goals"]

# Axis indices — re-exported so tests that reference them directly still work.
EXCITEMENT_AXIS: int = 4
RESIDUAL_AXIS: int = 8

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

    return prime_from_hits(
        agent_id=agent_id,
        emotional_hits=emotional_hits,
        semantic_hits=semantic_hits,
        lambda_t=lambda_t,
        activation_floor=activation_floor,
        recall_top_k=recall_top_k,
    )
