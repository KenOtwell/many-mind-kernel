"""
Goal pool — resonance-attractor goals for opportunistic pursuit.

Phase 1 of the goal-resonance design. Goals are dual-vector attractors in the
same space as memories: a 384d semantic trigger/affordance signature and a 9d
emotional affective signature. Live percepts resonate with them; the resonance
is surfaced to the mind as affect + recalled content (see goal_priming.py),
never as an imperative instruction.

This module owns the in-memory goal registry and the seed catalogue. Goal-state
lifecycle (candidate/committed/satisfied) and decomposition arrive in later
phases; Phase 1 ships hand-authored static goals so the priming channel can be
validated end to end.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
from uuid import NAMESPACE_DNS, uuid5

from mindcore import embedding as shared_embedding
from mindcore import emotional as shared_emotional
from shared.constants import ZERO_SEMAGRAM
from progeny.src import qdrant_client as progeny_qdrant

logger = logging.getLogger(__name__)

# Owner sentinel: a goal any agent can carry.
OWNER_ALL = "all"


class GoalState(str, Enum):
    """Lifecycle state of a goal node.

    Phase 1 uses LATENT/PRIMED only; the candidate -> committed -> satisfied
    commitment ramp and EXPIRED cleanup land with the Phase 2 lifecycle.
    """
    LATENT = "latent"
    PRIMED = "primed"
    CANDIDATE = "candidate"
    COMMITTED = "committed"
    SATISFIED = "satisfied"
    EXPIRED = "expired"


# States in which a goal still exerts motivational pull.
ACTIVE_STATES: frozenset[GoalState] = frozenset({
    GoalState.PRIMED, GoalState.CANDIDATE, GoalState.COMMITTED,
})


@dataclass
class GoalNode:
    """A single goal / subgoal / enabler, represented as a resonance attractor.

    trigger_cues feeds the 384d semantic vector (what perceptual language this
    goal resonates with). affect_signature feeds the 9d emotional vector (the
    felt shape of when the goal becomes salient). statement is the first-person
    recalled form surfaced to the LLM as ordinary recalled content.
    """
    name: str
    statement: str
    trigger_cues: str
    affect_signature: str
    role: str = "goal"            # goal | subgoal | enabler
    owner: str = OWNER_ALL
    state: GoalState = GoalState.PRIMED
    base_weight: float = 0.5
    parent: Optional[str] = None
    requires: list[str] = field(default_factory=list)
    enables: list[str] = field(default_factory=list)
    success_predicate: str = ""
    enabler_predicate: str = ""
    provenance: str = "seed"
    semantic_vec: list[float] = field(default_factory=list)
    emotional_vec: list[float] = field(default_factory=lambda: list(ZERO_SEMAGRAM))

    @property
    def goal_id(self) -> str:
        """Deterministic point ID so re-seeding is an idempotent upsert."""
        return str(uuid5(NAMESPACE_DNS, f"mmk:goal:{self.owner}:{self.name}"))

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES

    def to_payload(self) -> dict:
        """Qdrant payload — the server-side bookkeeping (never prompt-injected)."""
        return {
            "name": self.name,
            "role": self.role,
            "owner": self.owner,
            "state": self.state.value,
            "base_weight": self.base_weight,
            "parent": self.parent or "",
            "requires": self.requires,
            "enables": self.enables,
            "success_predicate": self.success_predicate,
            "enabler_predicate": self.enabler_predicate,
            "provenance": self.provenance,
        }


class GoalPool:
    """In-memory registry of goal nodes, keyed by goal_id."""

    def __init__(self) -> None:
        self._nodes: dict[str, GoalNode] = {}

    def __len__(self) -> int:
        return len(self._nodes)

    def register(self, node: GoalNode) -> None:
        self._nodes[node.goal_id] = node

    def get(self, goal_id: str) -> Optional[GoalNode]:
        return self._nodes.get(goal_id)

    def all_nodes(self) -> list[GoalNode]:
        return list(self._nodes.values())

    def by_owner(self, agent_id: str) -> list[GoalNode]:
        """Goals this agent carries — owned directly or shared (OWNER_ALL)."""
        return [
            n for n in self._nodes.values()
            if n.owner == agent_id or n.owner == OWNER_ALL
        ]

    def active_for(self, agent_id: str) -> list[GoalNode]:
        """Owned goals still exerting motivational pull (Phase 1 standing pull)."""
        return [n for n in self.by_owner(agent_id) if n.is_active]


def vectorize_node(node: GoalNode) -> bool:
    """Compute a node's semantic + emotional vectors from its text.

    Returns True on success. Requires the shared embedding + emotional
    pipelines to be loaded; returns False (leaving the zero emotional vector
    and empty semantic vector) otherwise so callers degrade gracefully rather
    than crash on a cold start.
    """
    if not shared_embedding.is_loaded() or not shared_emotional.is_loaded():
        return False
    try:
        node.semantic_vec = shared_embedding.embed_one(node.trigger_cues).tolist()
        affect_emb = shared_embedding.embed_one(node.affect_signature)
        node.emotional_vec = shared_emotional.project(affect_emb)
        return True
    except Exception:
        logger.exception("vectorize_node failed for goal %s", node.name)
        return False


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
