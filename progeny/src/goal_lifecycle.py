"""
Goal lifecycle and dissonance (Phase 2 of the goal-resonance design).

Phase 1 primed goals by resonance. Phase 2 makes them *defeasible*: each tick
their predicates are recomputed from current perception (never latched), and
their per-agent state moves along a commitment ramp that can also move
backward when the world changes.

Three pieces:

  * PerceptView + evaluate_predicate — ground a goal's success/enabler
    predicate against whatever the current tick actually reports. Unknown or
    unevaluable -> False (conservative: the goal stays open).
  * LifecycleStore + update_lifecycle — per-(agent, goal) state with the
    candidate -> committed -> satisfied ramp, defeasible reversion, and soft
    lateral inhibition between disjunctive sibling candidates (rabbit vs quail
    compete; the loser stays warm, never deleted).
  * compute_dissonance — an agent-level tension scalar from unmet-enabler
    weight + emotional volatility + uncertainty. Phase 2 uses it to amplify the
    standing motivational pull; Phase 3 will use it to gate decomposition.

State is per (agent, goal): the GoalNode is a shared attractor template, so a
node's lifecycle cannot be global — Lydia committing to the hunt must not mark
it committed for everyone.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

from progeny.src.goal_pool import ACTIVE_STATES, GoalNode, GoalState

if TYPE_CHECKING:
    from progeny.src.event_accumulator import TurnContext

logger = logging.getLogger(__name__)

# --- Commitment-ramp thresholds (activation is the blended resonance) -------
CANDIDATE_THRESHOLD = 0.20   # at/above this, a goal becomes a candidate
COMMIT_THRESHOLD = 0.35      # at/above this, persistence accrues toward commit
COMMIT_TICKS = 2             # consecutive high-activation ticks to commit
SWITCH_MARGIN = 0.10         # a challenger must beat the incumbent by this to win

# --- Dissonance weights (sum ~1.0) ------------------------------------------
W_ENABLER = 0.5
W_VOLATILITY = 0.3
W_UNCERTAINTY = 0.2

# Heuristic weapon vocabulary for has_equipped(category='weapon').
_WEAPON_KEYWORDS = (
    "sword", "bow", "axe", "dagger", "mace", "staff", "warhammer",
    "battleaxe", "greatsword", "crossbow", "spear", "halberd",
)

_ATOM_RE = re.compile(r"^\s*(\w+)\(([^)]*)\)\s*$")
_KWARG_RE = re.compile(r"(\w+)\s*=\s*'?([^',]+)'?")


# ---------------------------------------------------------------------------
# PerceptView — what the current tick lets us evaluate
# ---------------------------------------------------------------------------

@dataclass
class PerceptView:
    """A read-only snapshot of evaluable signals for one tick.

    percept_text is the lowercased scene blob (world events + player input +
    per-agent event text). equipment/inventory are per-agent sets of item
    names. Anything we cannot observe is simply absent, and predicates over it
    evaluate False — the goal stays open rather than falsely closing.
    """
    percept_text: str = ""
    equipment: dict[str, set[str]] = field(default_factory=dict)
    inventory: dict[str, set[str]] = field(default_factory=dict)

    @classmethod
    def from_turn_context(cls, turn_context: "TurnContext") -> "PerceptView":
        parts: list[str] = []
        equipment: dict[str, set[str]] = {}
        for ev in turn_context.world_events:
            if ev.raw_data:
                parts.append(ev.raw_data)
        if turn_context.player_input:
            parts.append(turn_context.player_input)
        for agent_id, buf in turn_context.agent_buffers.items():
            for ev in buf.events:
                if ev.raw_data:
                    parts.append(ev.raw_data)
                if ev.event_type == "addnpc" and ev.parsed_data:
                    eq = ev.parsed_data.get("equipment", {})
                    vals = {
                        str(v) for k, v in eq.items()
                        if v and not k.endswith("_baseid")
                    }
                    if vals:
                        equipment.setdefault(agent_id, set()).update(vals)
        return cls(percept_text=" ".join(parts).lower(), equipment=equipment)

    def perceives(self, token: str) -> bool:
        token = token.strip().strip("'\"").lower()
        return bool(token) and token in self.percept_text

    def has_equipped(self, agent_id: str, category: str) -> bool:
        items = self.equipment.get(agent_id, set())
        if category.lower() == "weapon":
            return any(
                kw in item.lower() for item in items for kw in _WEAPON_KEYWORDS
            )
        # Unknown category: match by substring against equipped item names.
        return any(category.lower() in item.lower() for item in items)

    def inventory_has(
        self, agent_id: str, item: Optional[str] = None, category: Optional[str] = None,
    ) -> bool:
        owned = {i.lower() for i in self.inventory.get(agent_id, set())}
        if item:
            return item.lower() in owned
        if category:
            return any(category.lower() in i for i in owned)
        return False


# ---------------------------------------------------------------------------
# Predicate evaluation — recomputed every tick (never latched)
# ---------------------------------------------------------------------------

def evaluate_predicate(predicate: str, view: PerceptView, agent_id: str) -> bool:
    """Evaluate a goal predicate against the current PerceptView.

    Grammar (Phase 2, deliberately tiny): atoms joined by top-level ` or `/
    ` and `. Atoms: perceived('x'), has_equipped(category='weapon'),
    inventory_has(item='arrow'|category='food'). Empty/unparseable -> False.
    """
    pred = (predicate or "").strip()
    if not pred:
        return False
    if " or " in pred:
        return any(evaluate_predicate(p, view, agent_id) for p in pred.split(" or "))
    if " and " in pred:
        return all(evaluate_predicate(p, view, agent_id) for p in pred.split(" and "))

    m = _ATOM_RE.match(pred)
    if not m:
        return False
    func, raw_args = m.group(1), m.group(2)

    if func == "perceived":
        # First positional arg is the token (e.g. perceived('rabbit')).
        first = raw_args.split(",")[0]
        return view.perceives(first)
    if func == "has_equipped":
        kwargs = dict(_KWARG_RE.findall(raw_args))
        category = kwargs.get("category", "")
        return view.has_equipped(agent_id, category) if category else False
    if func == "inventory_has":
        kwargs = dict(_KWARG_RE.findall(raw_args))
        return view.inventory_has(
            agent_id, item=kwargs.get("item"), category=kwargs.get("category"),
        )
    return False


# ---------------------------------------------------------------------------
# Per-(agent, goal) runtime state
# ---------------------------------------------------------------------------

@dataclass
class GoalRuntime:
    """Dynamic per-agent lifecycle state for one goal node."""
    state: GoalState
    commitment: float = 0.0
    lead_ticks: int = 0
    last_activation: float = 0.0

    @property
    def is_active(self) -> bool:
        return self.state in ACTIVE_STATES


@dataclass
class Transition:
    """A recorded state change, for telemetry/logging."""
    goal_id: str
    name: str
    old: GoalState
    new: GoalState


class LifecycleStore:
    """Per-(agent, goal) runtime registry. The GoalNode stays a shared template."""

    def __init__(self) -> None:
        self._runtimes: dict[tuple[str, str], GoalRuntime] = {}

    def get_or_create(self, agent_id: str, node: GoalNode) -> GoalRuntime:
        key = (agent_id, node.goal_id)
        rt = self._runtimes.get(key)
        if rt is None:
            rt = GoalRuntime(state=node.state)
            self._runtimes[key] = rt
        return rt

    def get(self, agent_id: str, goal_id: str) -> Optional[GoalRuntime]:
        return self._runtimes.get((agent_id, goal_id))

    def state_of(self, agent_id: str, node: GoalNode) -> GoalState:
        rt = self._runtimes.get((agent_id, node.goal_id))
        return rt.state if rt is not None else node.state

    def active_nodes(self, agent_id: str, owned: list[GoalNode]) -> list[GoalNode]:
        """Owned goals whose per-agent state still exerts pull (default = template)."""
        return [n for n in owned if self.state_of(agent_id, n) in ACTIVE_STATES]


# ---------------------------------------------------------------------------
# Lifecycle update — defeasible transitions + lateral inhibition
# ---------------------------------------------------------------------------

def _transition(rt: GoalRuntime, activation: float, success: bool) -> None:
    """Move one runtime along the commitment ramp. Recomputed, not latched.

    success -> SATISFIED. Otherwise activation drives candidacy, and a goal
    that was SATISFIED but whose predicate no longer holds reopens (the rabbit
    fled, the quiver emptied).
    """
    if success:
        rt.state = GoalState.SATISFIED
        rt.lead_ticks = 0
        rt.commitment = 1.0
        return

    if rt.state == GoalState.SATISFIED:
        # Predicate retracted — reopen and re-evaluate from activation.
        rt.state = GoalState.PRIMED
        rt.lead_ticks = 0
        rt.commitment = 0.0

    if activation >= COMMIT_THRESHOLD:
        rt.lead_ticks += 1
        rt.state = (
            GoalState.COMMITTED if rt.lead_ticks >= COMMIT_TICKS else GoalState.CANDIDATE
        )
    elif activation >= CANDIDATE_THRESHOLD:
        rt.state = GoalState.CANDIDATE
        rt.lead_ticks = 0
    else:
        rt.state = GoalState.PRIMED
        rt.lead_ticks = 0

    rt.commitment = min(1.0, rt.lead_ticks / COMMIT_TICKS)


def _sibling_groups(owned: list[GoalNode]) -> list[list[GoalNode]]:
    """Group candidate siblings by parent (disjunctive alternatives)."""
    groups: dict[str, list[GoalNode]] = {}
    for node in owned:
        if node.role == "candidate" and node.parent:
            groups.setdefault(node.parent, []).append(node)
    return [g for g in groups.values() if len(g) > 1]


def update_lifecycle(
    agent_id: str,
    owned: list[GoalNode],
    store: LifecycleStore,
    activations: dict[str, float],
    view: PerceptView,
) -> list[Transition]:
    """Recompute every owned goal's per-agent state for this tick.

    Returns the list of state changes (for telemetry). Applies the commitment
    ramp per goal, then soft lateral inhibition within each disjunctive sibling
    group: only the leader may hold CANDIDATE/COMMITTED; the others fall back to
    PRIMED (warm, not deleted). The incumbent keeps the lead unless a challenger
    beats it by SWITCH_MARGIN — cheap hysteresis against thrash.
    """
    old_states: dict[str, GoalState] = {}

    # Pass 1: per-node defeasible transition.
    for node in owned:
        rt = store.get_or_create(agent_id, node)
        old_states[node.goal_id] = rt.state
        activation = activations.get(node.goal_id, 0.0)
        success = evaluate_predicate(node.success_predicate, view, agent_id)
        _transition(rt, activation, success)
        rt.last_activation = activation

    # Pass 2: lateral inhibition among disjunctive sibling candidates.
    for group in _sibling_groups(owned):
        leader = _select_leader(agent_id, group, store)
        for node in group:
            rt = store.get_or_create(agent_id, node)
            if node is not leader and rt.state in (GoalState.CANDIDATE, GoalState.COMMITTED):
                rt.state = GoalState.PRIMED
                rt.lead_ticks = 0
                rt.commitment = 0.0

    # Collect transitions.
    transitions: list[Transition] = []
    for node in owned:
        new = store.state_of(agent_id, node)
        if new != old_states[node.goal_id]:
            transitions.append(Transition(node.goal_id, node.name, old_states[node.goal_id], new))
    return transitions


def _select_leader(
    agent_id: str, group: list[GoalNode], store: LifecycleStore,
) -> GoalNode:
    """Pick the winning sibling with incumbent preference + a switch margin."""
    def act(n: GoalNode) -> float:
        rt = store.get(agent_id, n.goal_id)
        return rt.last_activation if rt is not None else 0.0

    incumbent = next(
        (n for n in group if store.state_of(agent_id, n) in (GoalState.CANDIDATE, GoalState.COMMITTED)),
        None,
    )
    challenger = max(group, key=act)
    if incumbent is None:
        return challenger
    if challenger is incumbent:
        return incumbent
    # A challenger only unseats the incumbent by beating it by the margin.
    return challenger if act(challenger) >= act(incumbent) + SWITCH_MARGIN else incumbent


# ---------------------------------------------------------------------------
# Dissonance
# ---------------------------------------------------------------------------

def compute_dissonance(
    agent_id: str,
    owned: list[GoalNode],
    store: LifecycleStore,
    view: PerceptView,
    *,
    curvature: float = 0.0,
    snap: float = 0.0,
    coherence: float = 1.0,
    certainty: float = 1.0,
) -> float:
    """Agent-level tension in [0, 1].

    Combines unmet-enabler weight (active enablers whose predicate is false),
    emotional volatility (curvature, |snap|, 1-coherence), and uncertainty
    (1-certainty). Phase 2 uses this to amplify the standing pull; Phase 3 will
    gate decomposition on it.
    """
    enablers = [
        n for n in owned
        if n.role == "enabler" and store.state_of(agent_id, n) in ACTIVE_STATES
    ]
    if enablers:
        total = sum(n.base_weight for n in enablers)
        unmet = sum(
            n.base_weight for n in enablers
            if not evaluate_predicate(n.enabler_predicate, view, agent_id)
        )
        unmet_frac = unmet / total if total > 0 else 0.0
    else:
        unmet_frac = 0.0

    volatility = (
        min(1.0, abs(curvature)) + min(1.0, abs(snap)) + (1.0 - max(0.0, min(1.0, coherence)))
    ) / 3.0
    uncertainty = 1.0 - max(0.0, min(1.0, certainty))

    score = (
        W_ENABLER * unmet_frac
        + W_VOLATILITY * volatility
        + W_UNCERTAINTY * uncertainty
    )
    return max(0.0, min(1.0, score))
