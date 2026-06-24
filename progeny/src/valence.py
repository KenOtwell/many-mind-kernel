"""
Valence-conditioned approach — Phase 6c (Social Bootstrapping).

When another NPC is present, the observing agent runs ONE percept-cued
retrieval against its own memories — not a separate class lookup. The semantic
axis pulls back feature-similar episodes (a soldier's armour and bearing, the
gate where it happened) and the emotional axis — queried emotion-first — pulls
back affect-congruent episodes. The perceived person's occupation/tags (from
the identity kernel's ``class_signal``) only SHARPEN the semantic query text;
there is no faction taxonomy. This is how a bad history with soldiers
generalizes to a brand-new soldier with zero shared history: generalization is
anchored on the semantic axis, so it does not over-broaden to anyone who merely
triggers the same feeling.

From the recalled memories' emotional signatures we extract a VALENCE — warmth
vs wariness — for the (observer, subject) pair. Valence reuses the codebase's
documented decomposition (Living Doc, MAX-tier valence-trajectory compression):

    positive valence = love + excitement + joy + safety
    negative valence = fear + anger + disgust + sadness
    warmth           = positive - negative   (residual is affect-neutral)

Because the stored emotional key is the NPC's *reaction* (the second-thought
ritual: deviation from baseline), a memory keyed with dread reads wary and one
keyed with safety/joy reads warm — without any sentiment tag.

This module (6c-1) provides the single percept-cued retrieval and the valence
extraction only. It keeps the generalized prior (non-referent memories) and the
individual signal (memories about THIS subject) separate so that 6c-2 can let
the specific override the prior as a blend with hysteresis, and 6c-3 can gate
the get-acquainted resonance on the result. Everything here is a server-side
signal; nothing is injected into the prompt (surfacing discipline).

Model-free by design (like ``goal_priming``): callers pass pre-embedded
queries, so the retrieval/extraction logic is exercised without a model load.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from qdrant_client.models import FieldCondition, Filter, MatchValue

from shared.constants import COLLECTION_NPC_MEMORIES
from progeny.src.acquaintance import mentions
from progeny.src.qdrant_client import get_points_by_ids, search_vector

logger = logging.getLogger(__name__)

# Warmth/wariness membership over the 9d semagram (axis order: shared.constants
# EMOTIONAL_AXES). Mirrors the Living Doc's MAX-tier valence-trajectory
# compression — the single documented valence convention in the codebase.
POSITIVE_AXES: tuple[int, ...] = (2, 4, 6, 7)  # love, excitement, joy, safety
NEGATIVE_AXES: tuple[int, ...] = (0, 1, 3, 5)  # fear, anger, disgust, sadness
RESIDUAL_AXIS: int = 8                          # domain content — affect-neutral

# Emotion-first percept cue: the agent's affect drives the recall, the percept
# text only sharpens it. Matches the recognition path's lambda (routes.py).
DEFAULT_VALENCE_LAMBDA: float = 0.8
DEFAULT_BROAD_LIMIT: int = 20    # candidates per axis before blending
DEFAULT_TOP_K: int = 8           # memories whose affect feeds the reading
DEFAULT_RECALL_TOP_K: int = 2    # class-congruent snippets surfaced as recall
# Additive lift so memories about THIS subject stay relevant (referent
# precedence) even when the recall is mood-incongruent. Mirrors the intent of
# memory_retrieval._apply_referent_boost.
REFERENT_BOOST: float = 0.5


def valence_components(semagram: list[float]) -> tuple[float, float, float]:
    """Return ``(positive, negative, residual)`` per the documented split.

    Sums the raw axis coordinates, so a negative coordinate on a positive axis
    (e.g. ``safety`` going negative = *unsafe*) correctly erodes warmth.
    """
    pos = sum(semagram[i] for i in POSITIVE_AXES)
    neg = sum(semagram[i] for i in NEGATIVE_AXES)
    res = semagram[RESIDUAL_AXIS] if len(semagram) > RESIDUAL_AXIS else 0.0
    return float(pos), float(neg), float(res)


def warmth(semagram: list[float]) -> float:
    """Signed warmth (positive) vs wariness (negative) of a 9d semagram."""
    pos, neg, _ = valence_components(semagram)
    return pos - neg


@dataclass
class MemoryAffect:
    """One retrieved memory's contribution to a valence reading."""
    point_id: str
    weight: float        # blended retrieval relevance (lambda emo + (1-lambda) sem)
    warmth: float        # signed warmth of its stored emotional reaction
    is_individual: bool  # references THIS subject (referent payload or name mention)


@dataclass
class ValenceReading:
    """Warmth-vs-wariness extracted for one (observer, subject) percept.

    Keeps the generalized prior (``general``, from class/feature-similar
    memories) separate from the individual signal (``individual``, from
    memories about THIS subject). 6c-2 blends them with hysteresis; 6c-3 gates
    the get-acquainted resonance and behavioural expression on the result.
    """
    observer: str
    subject: str
    general: float = 0.0          # prior valence from non-referent memories
    individual: float = 0.0       # valence from referent-specific memories
    general_support: int = 0      # count of non-referent memories contributing
    individual_support: int = 0   # count of referent-specific memories contributing
    confidence: float = 0.0       # mean blended relevance of contributors
    recall: list[str] = field(default_factory=list)  # class-congruent recall snippets

    @property
    def has_individual(self) -> bool:
        """True once at least one memory about THIS subject has surfaced."""
        return self.individual_support > 0

    @property
    def has_signal(self) -> bool:
        """True if any memory (prior or individual) contributed."""
        return (self.general_support + self.individual_support) > 0


def _weighted_mean_warmth(items: list[MemoryAffect]) -> float:
    """Relevance-weighted mean warmth; falls back to a plain mean if all
    weights are ~0 (e.g. negative cosine similarities clamped away)."""
    if not items:
        return 0.0
    total_w = sum(max(c.weight, 0.0) for c in items)
    if total_w <= 0.0:
        return sum(c.warmth for c in items) / len(items)
    return sum(c.warmth * max(c.weight, 0.0) for c in items) / total_w


def extract_valence(
    observer: str, subject: str, contributions: list[MemoryAffect],
) -> ValenceReading:
    """Aggregate per-memory affect into a (observer, subject) valence reading.

    Pure: partitions contributions into individual (about THIS subject) vs
    general (the prior), then takes a relevance-weighted mean warmth of each.
    """
    reading = ValenceReading(observer=observer, subject=subject)
    individual = [c for c in contributions if c.is_individual]
    general = [c for c in contributions if not c.is_individual]
    reading.individual = _weighted_mean_warmth(individual)
    reading.general = _weighted_mean_warmth(general)
    reading.individual_support = len(individual)
    reading.general_support = len(general)
    if contributions:
        mean_weight = sum(max(c.weight, 0.0) for c in contributions) / len(contributions)
        reading.confidence = min(1.0, mean_weight)
    return reading


def build_percept_text(subject: str, class_signal: str = "") -> str:
    """Percept query text: the perceived person, sharpened by their kind.

    The subject's name anchors individual recall; ``class_signal`` (occupation
    + tags from the identity kernel) only sharpens the semantic query so
    feature-similar episodes about *that kind* of person resonate — no
    taxonomy, just richer query text. Callers embed the returned string.
    """
    subject = (subject or "").strip()
    cls = (class_signal or "").strip()
    if subject and cls:
        return f"{subject}, {cls}"
    return subject or cls


def _emotional_vector(vector: object) -> list[float] | None:
    """Pull the 9d 'emotional' named vector from a get_points_by_ids result."""
    if isinstance(vector, dict):
        emo = vector.get("emotional")
        if emo is not None:
            return list(emo)
    return None


def _is_about_subject(payload: dict, subject: str) -> bool:
    """True if a memory references THIS subject specifically.

    Matches a referent-payload entry (memories written via ``write_raw_event``)
    OR a name mention in the content. The mention fallback means the override
    still works on the live ``qdrant_wrapper.ingest`` path, which stores no
    referents but keeps the text. Reuses ``acquaintance.mentions`` for the
    name/first-token match.
    """
    subj_low = (subject or "").strip().lower()
    if not subj_low:
        return False
    referents = payload.get("referents") or []
    if any(str(r).strip().lower() == subj_low for r in referents):
        return True
    content = payload.get("content") or payload.get("text") or ""
    return mentions(content, subject)


def _blend_scores(
    emotional_hits: list[dict], semantic_hits: list[dict], lambda_t: float,
) -> dict[str, tuple[float, dict]]:
    """Blend the two axes into ``pid -> (weight, payload)``.

    weight = lambda * emotional_sim + (1 - lambda) * semantic_sim, mirroring
    ``memory_retrieval._merge_and_score`` and ``goal_priming._blend``.
    """
    by_id: dict[str, list] = {}
    for hit in emotional_hits:
        pid = hit["id"]
        by_id[pid] = [lambda_t * hit["score"], hit.get("payload") or {}]
    complement = 1.0 - lambda_t
    for hit in semantic_hits:
        pid = hit["id"]
        contrib = complement * hit["score"]
        if pid in by_id:
            by_id[pid][0] += contrib
        else:
            by_id[pid] = [contrib, hit.get("payload") or {}]
    return {pid: (w, payload) for pid, (w, payload) in by_id.items()}


async def percept_cued_valence(
    observer: str,
    subject: str,
    semantic_query: list[float],
    emotional_query: list[float],
    lambda_t: float = DEFAULT_VALENCE_LAMBDA,
    broad_limit: int = DEFAULT_BROAD_LIMIT,
    top_k: int = DEFAULT_TOP_K,
) -> ValenceReading:
    """Run ONE percept-cued retrieval over the observer's memories and extract
    a valence reading for the perceived subject.

    Args:
        observer:        The agent doing the perceiving (its memories are read).
        subject:         The perceived NPC the valence is about.
        semantic_query:  384d embedding of the percept text (perceived person,
                         sharpened by ``build_percept_text``). Anchors
                         generalization on the semantic axis.
        emotional_query: 9d affect query (typically the observer's deviation,
                         fast - slow), queried emotion-first.
        lambda_t:        Emotional vs semantic blend; defaults emotion-first.
        broad_limit:     Candidates fetched per axis before blending.
        top_k:           Memories whose affect feeds the reading.

    Returns:
        A ValenceReading. Empty-but-valid when the observer has no memories or
        Qdrant is unreachable (graceful degradation — never raises).
    """
    agent_filter = Filter(
        must=[FieldCondition(key="agent_id", match=MatchValue(value=observer))]
    )
    emotional_hits = await search_vector(
        collection=COLLECTION_NPC_MEMORIES, vector_name="emotional",
        query=emotional_query, limit=broad_limit, query_filter=agent_filter,
    )
    semantic_hits = await search_vector(
        collection=COLLECTION_NPC_MEMORIES, vector_name="semantic",
        query=semantic_query, limit=broad_limit, query_filter=agent_filter,
    )

    scored = _blend_scores(emotional_hits, semantic_hits, lambda_t)
    if not scored:
        return ValenceReading(observer=observer, subject=subject)

    # Referent precedence: lift memories about THIS subject above merely
    # feature-similar ones (mirrors memory_retrieval._apply_referent_boost), so
    # the specific outranks the general and stays relevant even when the recall
    # is mood-incongruent (you can be calm yet still wary on seeing them).
    ranked: list[tuple[str, float, dict, bool]] = []
    for pid, (weight, payload) in scored.items():
        is_individual = _is_about_subject(payload, subject)
        eff_weight = weight + (REFERENT_BOOST if is_individual else 0.0)
        ranked.append((pid, eff_weight, payload, is_individual))
    ranked.sort(key=lambda r: r[1], reverse=True)
    top = ranked[:top_k]

    # Fetch the emotional keys for the survivors — the affect signature lives in
    # the 'emotional' named vector, which search_vector does not return.
    top_ids = [pid for pid, _w, _payload, _ind in top]
    points = await get_points_by_ids(
        COLLECTION_NPC_MEMORIES, top_ids, with_vectors=True,
    )
    vec_by_id = {p["id"]: _emotional_vector(p.get("vector")) for p in points}

    contributions: list[MemoryAffect] = []
    recall_pool: list[tuple[float, bool, str]] = []
    for pid, eff_weight, payload, is_individual in top:
        emo = vec_by_id.get(pid)
        if emo is None:
            continue
        contributions.append(MemoryAffect(
            point_id=pid,
            weight=eff_weight,
            warmth=warmth(emo),
            is_individual=is_individual,
        ))
        text = (payload.get("content") or payload.get("text") or "").strip()
        if text:
            recall_pool.append((eff_weight, is_individual, text))

    reading = extract_valence(observer, subject, contributions)
    reading.recall = _select_recall(recall_pool)
    if reading.has_signal:
        logger.debug(
            "Valence: observer=%s subject=%s general=%.3f(n=%d) "
            "individual=%.3f(n=%d) conf=%.2f",
            observer, subject, reading.general, reading.general_support,
            reading.individual, reading.individual_support, reading.confidence,
        )
    return reading


# ---------------------------------------------------------------------------
# 6c-2: referent-precedence override — blend with hysteresis (not erasure)
#
# Specific memories of THIS individual outrank the generalized prior, but the
# override is gradual: the prior keeps tinting until enough individual evidence
# accumulates. A first-ever meeting is the pure-prior case (only the class
# colour shows); one kind encounter does not erase a wary prior; as individual
# memories accumulate the person emerges from the class shadow. The hysteresis
# is delivered by a saturating, evidence-ramped weight on the individual signal
# plus an in-session EMA that consolidates the individual estimate. Slow
# cross-session annealing/decay of the disposition is deferred to 6f.
# ---------------------------------------------------------------------------

# Individual encounters at which the override reaches ~0.5 weight. Larger = the
# class prior tints for longer before the individual takes over.
HYSTERESIS_K: float = 3.0
# In-session EMA rate consolidating repeated individual readings into a stable
# estimate. Symmetric here; asymmetric reinforce/decay belongs to 6f annealing.
ESTIMATE_ALPHA: float = 0.5


@dataclass
class _IndividualPrior:
    """Consolidated individual valence for one (observer, subject) pair."""
    estimate: float = 0.0
    encounters: int = 0


# Process-lifetime ledger (sibling to the acquaintance stranger ledger and the
# identity-kernel cache). Keyed (observer, subject-lowercased).
_ledger: dict[tuple[str, str], _IndividualPrior] = {}


def _ledger_key(observer: str, subject: str) -> tuple[str, str]:
    return (observer, (subject or "").strip().lower())


def individual_weight(encounters: int, hysteresis_k: float = HYSTERESIS_K) -> float:
    """Saturating override weight: ``n / (n + k)`` in [0, 1).

    Zero with no individual encounters (pure prior); ~0.5 at ``k`` encounters;
    approaches 1 as evidence accumulates. This ramp IS the hysteresis — a
    single encounter (n=1, k=3 -> 0.25) leaves the prior dominant.
    """
    if encounters <= 0:
        return 0.0
    return encounters / (encounters + hysteresis_k)


@dataclass
class BlendedValence:
    """The effective valence the agent acts on, after referent precedence.

    ``effective`` blends the consolidated ``individual`` estimate over the
    class ``prior`` by ``individual_weight``. 6c-3 gates the get-acquainted
    resonance and shapes behavioural expression on ``effective``.
    """
    observer: str
    subject: str
    effective: float
    prior: float
    individual: float
    individual_weight: float
    encounters: int
    confidence: float

    @property
    def is_wary(self) -> bool:
        return self.effective < 0.0

    @property
    def is_warm(self) -> bool:
        return self.effective > 0.0


def blend_valence(
    reading: ValenceReading,
    *,
    hysteresis_k: float = HYSTERESIS_K,
    estimate_alpha: float = ESTIMATE_ALPHA,
    update: bool = True,
) -> BlendedValence:
    """Apply referent precedence to a reading as a blend with hysteresis.

    When the reading carries an individual signal, consolidate it into the
    ledger (EMA) and count the encounter; the effective valence then blends the
    consolidated individual estimate over the class prior by an evidence-ramped
    weight. With no individual signal the stored disposition (if any) still
    applies; with neither, the result is the pure prior.

    Args:
        reading:        The (observer, subject) ValenceReading from 6c-1.
        hysteresis_k:   Encounters for the override to reach ~0.5 weight.
        estimate_alpha: In-session EMA rate for the individual estimate.
        update:         When False, compute without mutating the ledger
                        (read-only peek).

    Returns:
        A BlendedValence with the effective valence and its provenance.
    """
    key = _ledger_key(reading.observer, reading.subject)
    rec = _ledger.get(key)

    if reading.has_individual:
        if rec is None:
            rec = _IndividualPrior(estimate=reading.individual, encounters=1)
        else:
            blended = (
                (1.0 - estimate_alpha) * rec.estimate
                + estimate_alpha * reading.individual
            )
            rec = _IndividualPrior(estimate=blended, encounters=rec.encounters + 1)
        if update:
            _ledger[key] = rec

    individual_est = rec.estimate if rec is not None else 0.0
    encounters = rec.encounters if rec is not None else 0
    weight = individual_weight(encounters, hysteresis_k)
    effective = weight * individual_est + (1.0 - weight) * reading.general

    return BlendedValence(
        observer=reading.observer,
        subject=reading.subject,
        effective=effective,
        prior=reading.general,
        individual=individual_est,
        individual_weight=weight,
        encounters=encounters,
        confidence=reading.confidence,
    )


def peek_prior(observer: str, subject: str) -> tuple[float, int]:
    """Return the stored ``(individual_estimate, encounters)`` without mutating."""
    rec = _ledger.get(_ledger_key(observer, subject))
    return (rec.estimate, rec.encounters) if rec is not None else (0.0, 0)


def reset() -> None:
    """Drop all consolidated individual priors and social snapshots (session wipe)."""
    _ledger.clear()
    _snapshots.clear()


# ---------------------------------------------------------------------------
# 6c-3 helpers: recall selection + personality approachability
# ---------------------------------------------------------------------------

# Trait / communication-style keywords that gently raise or lower how readily
# an NPC approaches others. This is a lean on the nudge magnitude, not a
# controller — the dominant personality shaping stays the harmonic buffer's
# engine modulators (e.g. Confidence damping the wariness fear-component).
WARM_TRAIT_HINTS: frozenset[str] = frozenset({
    "friendly", "outgoing", "warm", "gregarious", "kind", "cheerful",
    "welcoming", "sociable", "affable", "open", "curious",
})
WARY_TRAIT_HINTS: frozenset[str] = frozenset({
    "suspicious", "reserved", "gruff", "aloof", "cold", "guarded", "wary",
    "distrustful", "shy", "taciturn", "stern", "surly",
})
APPROACHABILITY_STEP: float = 0.15
APPROACHABILITY_MIN: float = 0.5
APPROACHABILITY_MAX: float = 1.5


def _select_recall(
    recall_pool: list[tuple[float, bool, str]],
    top_k: int = DEFAULT_RECALL_TOP_K,
) -> list[str]:
    """Pick up to ``top_k`` distinct recall snippets, preferring general
    (class-congruent) memories over individual ones.

    The individual recall is surfaced separately by the recognition path, so
    valence contributes the class-congruent 'why' (e.g. memories of *other*
    soldiers behind wariness toward a brand-new one).
    """
    ordered = sorted(recall_pool, key=lambda r: (r[1], -r[0]))
    out: list[str] = []
    seen: set[str] = set()
    for _weight, _is_individual, text in ordered:
        if text in seen:
            continue
        seen.add(text)
        out.append(text)
        if len(out) >= top_k:
            break
    return out


def approachability(traits: list[str] | None, tone: str = "") -> float:
    """Gentle multiplier in [APPROACHABILITY_MIN, APPROACHABILITY_MAX] on the
    approach-nudge magnitude, from an NPC's traits / communication tone.

    Defaults to 1.0 (neutral) when nothing recognizable is present. Tokens are
    matched against small warm/wary keyword sets; each match leans the score by
    one step. A modulation, never a controller.
    """
    score = 1.0
    tokens: list[str] = []
    if traits:
        for trait in traits:
            tokens.extend(str(trait).lower().split())
    if tone:
        tokens.extend(str(tone).lower().split())
    for tok in tokens:
        if tok in WARM_TRAIT_HINTS:
            score += APPROACHABILITY_STEP
        elif tok in WARY_TRAIT_HINTS:
            score -= APPROACHABILITY_STEP
    return max(APPROACHABILITY_MIN, min(APPROACHABILITY_MAX, score))


# ---------------------------------------------------------------------------
# 6d support: social valence snapshot
#
# The latest effective valence (after the hysteretic blend) and the prior-vs-
# individual affect gap toward each subject, cached by the 6c conditioning pass
# and read by social_goals for the get-acquainted goal's activation gating and
# the dissonance affect-gap term. Process-lifetime; cleared on reset().
# ---------------------------------------------------------------------------


@dataclass
class SocialSnapshot:
    """Latest valence summary toward a subject, for downstream social goals."""
    effective: float = 0.0
    affect_gap: float = 0.0


_snapshots: dict[tuple[str, str], SocialSnapshot] = {}


def record_social(
    observer: str, subject: str, effective: float, affect_gap: float = 0.0,
) -> None:
    """Cache the effective valence + prior-vs-individual gap toward a subject."""
    _snapshots[_ledger_key(observer, subject)] = SocialSnapshot(effective, affect_gap)


def social_toward(observer: str, subject: str) -> SocialSnapshot | None:
    """Return the latest social snapshot toward a subject, or None."""
    return _snapshots.get(_ledger_key(observer, subject))
