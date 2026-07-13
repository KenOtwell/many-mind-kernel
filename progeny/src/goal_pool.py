"""
Goal pool — resonance-attractor goals for opportunistic pursuit.

Core goal types (GoalNode, GoalPool, GoalState, ACTIVE_STATES, vectorize_node,
OWNER_ALL) live in mindcore.goal and are imported here for backward
compatibility. This module owns only the Neo-specific seed catalogue
(build_seed_goals, seed_goals) and the Qdrant persistence path.
"""
from __future__ import annotations

import logging

from mindcore.goal import (
    GoalNode,
    GoalPool,
    GoalState,
    ACTIVE_STATES,
    vectorize_node,
    OWNER_ALL,
)
from progeny.src import qdrant_client as progeny_qdrant

# Re-export everything callers depend on through this module's namespace.
__all__ = [
    "GoalNode", "GoalPool", "GoalState", "ACTIVE_STATES",
    "vectorize_node", "OWNER_ALL",
    "build_seed_goals", "seed_goals",
]

logger = logging.getLogger(__name__)


def build_seed_goals(owner: str = OWNER_ALL) -> list[GoalNode]:
    """Hand-authored Phase 1 seed catalogue: the hunting goal graph.

    A single top goal (hunt for game) with a disjunctive prey subgoal and two
    enabler subgoals. Enough structure to validate that a rabbit percept
    resonates and surfaces the hunt as recalled content, with no auto-action.
    """
    return [
        GoalNode(
            name="hunt_for_game",
            statement="I set out to hunt game in these woods to bring back for dinner.",
            trigger_cues="hunting game for dinner, hunting for food, meat, prey, the hunt in the woods",
            affect_signature="focused eager purposeful anticipation, the quiet thrill of the hunt",
            role="goal",
            owner=owner,
            state=GoalState.PRIMED,
            base_weight=0.6,
            success_predicate="inventory_has(category='food')",
        ),
        GoalNode(
            name="locate_prey",
            statement="I am watching for game — a rabbit or a quail moving in the brush.",
            trigger_cues="a rabbit, a quail, a game animal, prey nearby, something moving in the brush",
            affect_signature="alert focused excitement at catching sight of prey",
            role="subgoal",
            owner=owner,
            state=GoalState.PRIMED,
            base_weight=0.5,
            parent="hunt_for_game",
            success_predicate="perceived('rabbit') or perceived('quail')",
        ),
        GoalNode(
            name="prey_rabbit",
            statement="There — a rabbit. That would do for the pot.",
            trigger_cues="a rabbit in the brush, a hare, a coney darting through the grass",
            affect_signature="quick alert excitement at spotting a rabbit",
            role="candidate",
            owner=owner,
            state=GoalState.PRIMED,
            base_weight=0.45,
            parent="locate_prey",
            success_predicate="perceived('rabbit')",
        ),
        GoalNode(
            name="prey_quail",
            statement="A quail, just there in the undergrowth.",
            trigger_cues="a quail, a small game bird flushing from cover",
            affect_signature="quick alert excitement at spotting a quail",
            role="candidate",
            owner=owner,
            state=GoalState.PRIMED,
            base_weight=0.4,
            parent="locate_prey",
            success_predicate="perceived('quail')",
        ),
        GoalNode(
            name="have_bow",
            statement="I should have my bow in hand for the hunt.",
            trigger_cues="a bow, a hunting bow, a weapon within reach",
            affect_signature="steady readiness, the comfort of a good weapon at hand",
            role="enabler",
            owner=owner,
            state=GoalState.PRIMED,
            base_weight=0.4,
            enables=["hunt_for_game"],
            enabler_predicate="has_equipped(category='weapon')",
        ),
        GoalNode(
            name="have_arrows",
            statement="I need arrows for my bow before I can hunt.",
            trigger_cues="arrows, a quiver, ammunition for a bow",
            affect_signature="practical preparation, mild worry about running short",
            role="enabler",
            owner=owner,
            state=GoalState.PRIMED,
            base_weight=0.35,
            enables=["hunt_for_game"],
            enabler_predicate="inventory_has(item='arrow')",
        ),
    ]


async def seed_goals(pool: GoalPool, owner: str = OWNER_ALL) -> int:
    """Register seed goals into the pool and persist them to Qdrant.

    Idempotent: deterministic goal IDs mean re-seeding overwrites in place
    rather than duplicating. Registers in-memory regardless; skips the Qdrant
    write for any node whose vectors could not be computed (embedding pipeline
    unavailable), so a cold start degrades gracefully.

    Returns the number of goals persisted to Qdrant.
    """
    nodes = build_seed_goals(owner=owner)
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
    logger.info("Seeded %d/%d goals (owner=%s)", written, len(nodes), owner)
    return written
