"""Sleep-time memory reconsolidation — the latent dissonance probe (REM phase).

A cheap, no-LLM pass that ranks an agent's stored memories by how much their
frozen reaction key conflicts with the reaction the agent's CURRENT (matured)
mind would have. The expensive LLM reinterpretation (SWS phase, later sub-phase)
then runs only on the few most-dissonant memories this probe selects.

Per-memory dissonance is the stored-memory analog of cross-buffer decoherence
``d(F, S)`` (docs/ATTRACTOR_FLOW_DYNAMICS.md §1.1): a prediction error between
what the memory recorded feeling and what the present self would feel.

Two probe modes:
  content  (default) — ``predicted = normalize(project(content) - slow_now)``;
                       ``dissonance = 1 - cos(stored_key, predicted)``. The
                       prediction drifts as the slow buffer matures, so old
                       reactions that no longer fit the current self rise to top.
  dcross             — ``dissonance = 1 - cos(stored_key, slow_now)``. The most
                       direct ``d(F, S)`` reading; simpler, but tends to flag any
                       memory whose felt-tone differs from the current baseline.

Cosine ranges [-1, 1], so dissonance ranges [0, 2] (0 = identical reaction,
1 = orthogonal, 2 = inverted). Selection is threshold-gate + top-K in those
units; callers tune the threshold to the same scale.

Compute-once (perf): agent-level invariants (``slow_now``) are taken once per
call and ``project`` is vectorized across the whole scanned batch via
``project_batch`` — no per-memory recomputation of invariants inside the loop.
"""
from __future__ import annotations

import inspect
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional
from uuid import NAMESPACE_DNS, uuid5

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchValue

from shared import emotional
from shared.constants import COLLECTION_NPC_MEMORIES
from shared.schemas import CompressionTier

from .memory_writer import MemoryWriter
from .qdrant_client import scroll_filtered

logger = logging.getLogger(__name__)

# Probe modes.
MODE_CONTENT = "content"   # default: re-project content, re-centre on current baseline
MODE_DCROSS = "dcross"     # stored-memory analog of d(F, S)

# Selection defaults (bounded cost per night).
DEFAULT_TOP_K = 8
DEFAULT_SCAN_LIMIT = 200              # max RAW memories scanned per agent per night
DEFAULT_DISSONANCE_THRESHOLD = 0.5   # in [0, 2] cosine-distance units
# Age weighting: older memories get up to (1 + AGE_GAIN)x the selection score,
# so a night's reconsolidation revisits the old. Off unless a half-life is given.
AGE_GAIN = 0.5

# Drift control (design #4): keep a re-encoded key anchored near the stored key.
DEFAULT_BLEND = 0.5            # fraction of the way to move toward the proposal
DEFAULT_MAX_DRIFT = 0.5        # cap on per-pass L2 change from the stored key
# Recurrence block: extra margin to re-reconsolidate a source that already has a
# (non-stalled) RECON, so marginal changes don't redo work every night.
DEFAULT_HYSTERESIS = 0.1

_EPS = 1e-9

# A scanned memory as returned by qdrant_client.scroll_filtered/get_points_by_ids
# with with_vectors=True: {"id", "payload", "vector": {"semantic", "emotional"}}.
Memory = dict[str, Any]


@dataclass
class ProbeResult:
    """One selected memory and its (raw, unweighted) dissonance."""
    point_id: str
    dissonance: float
    memory: Memory


@dataclass
class ReconReport:
    """Telemetry for one reconsolidation pass."""
    scanned_agents: int = 0
    reconsolidated: int = 0   # RECON points written (resolved + stalled)
    resolved: int = 0         # residual dissonance fell below threshold
    stalled: int = 0          # residual stayed >= threshold (flagged for analysis)
    skipped_blocked: int = 0  # recurrence-blocked (stalled prior) or already consonant


def _l2_normalize(vec: np.ndarray) -> np.ndarray:
    """Normalize a 1d vector, guarding against zero magnitude."""
    norm = float(np.linalg.norm(vec))
    if norm < _EPS:
        return np.zeros_like(vec)
    return vec / norm


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalize an (M, D) matrix, guarding against zero rows."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    return mat / np.maximum(norms, _EPS)


def _row_cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of two (M, D) matrices."""
    denom = np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), _EPS)
    return np.sum(a * b, axis=1) / denom


def predicted_reaction(
    semantic: list[float] | np.ndarray,
    slow_now: list[float] | np.ndarray,
    project_fn: Callable[[np.ndarray], list[float]] = emotional.project,
) -> np.ndarray:
    """The reaction the current mind would have to a memory's content.

    Content-anchored: project the (immutable) content embedding into emotional
    space, then re-centre on the agent's current slow baseline. As the slow
    buffer matures the prediction drifts, which is what surfaces stale reactions.

    Returns a normalized 9d vector (direction is what the cosine probe compares).
    """
    proj = np.asarray(project_fn(np.asarray(semantic, dtype=np.float32)), dtype=np.float32)
    return _l2_normalize(proj - np.asarray(slow_now, dtype=np.float32))


def memory_dissonance(
    stored_key: list[float] | np.ndarray,
    predicted: list[float] | np.ndarray,
) -> float:
    """Content-anchored per-memory dissonance: 1 - cos(stored_key, predicted)."""
    a = np.asarray(stored_key, dtype=np.float32)
    b = np.asarray(predicted, dtype=np.float32)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), _EPS)
    return 1.0 - float(np.dot(a, b) / denom)


def dcross_dissonance(
    stored_key: list[float] | np.ndarray,
    slow_now: list[float] | np.ndarray,
) -> float:
    """Cross-decoherence variant: 1 - cos(stored_key, slow_now)."""
    a = np.asarray(stored_key, dtype=np.float32)
    b = np.asarray(slow_now, dtype=np.float32)
    denom = max(float(np.linalg.norm(a) * np.linalg.norm(b)), _EPS)
    return 1.0 - float(np.dot(a, b) / denom)


def select_dissonant(
    memories: list[Memory],
    slow_now: list[float] | np.ndarray,
    *,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_DISSONANCE_THRESHOLD,
    mode: str = MODE_CONTENT,
    exclude_ids: Optional[set[str]] = None,
    now_ts: Optional[float] = None,
    age_halflife: Optional[float] = None,
    project_batch_fn: Callable[[np.ndarray], np.ndarray] = emotional.project_batch,
) -> list[ProbeResult]:
    """Rank scanned memories by per-memory dissonance and return the top-K.

    Args:
        memories: scanned points (with_vectors=True); each needs
            ``vector["emotional"]`` and, in content mode, ``vector["semantic"]``.
        slow_now: the agent's current 9d slow buffer (the matured baseline).
        top_k: max memories to return (bounds the LLM cost downstream).
        threshold: minimum raw dissonance to be eligible (cosine-distance units).
        mode: MODE_CONTENT (default) or MODE_DCROSS.
        exclude_ids: point ids to skip (e.g. recurrence-blocked / stalled sources).
        now_ts, age_halflife: if both set, weight the *selection score* toward
            older memories (raw dissonance, threshold, and provenance stay
            unweighted). Off by default.
        project_batch_fn: injected (N,384)->(N,9) projector; defaults to the
            shared emotional projection. Called once over the whole batch.

    Returns:
        ProbeResults sorted by selection score (desc), capped at top_k.
    """
    if not memories:
        return []
    if mode not in (MODE_CONTENT, MODE_DCROSS):
        raise ValueError(f"unknown probe mode: {mode!r}")

    exclude = exclude_ids or set()

    # Single pass to gather kept memories + aligned vector lists (compute-once:
    # we build the batch once, then vectorize the math below).
    kept: list[Memory] = []
    stored_keys: list[Any] = []
    semantics: list[Any] = []
    for mem in memories:
        point_id = str(mem.get("id"))
        if point_id in exclude:
            continue
        vectors = mem.get("vector") or {}
        emo = vectors.get("emotional") if isinstance(vectors, dict) else None
        if emo is None:
            continue
        if mode == MODE_CONTENT:
            sem = vectors.get("semantic")
            if sem is None:
                continue
            semantics.append(sem)
        kept.append(mem)
        stored_keys.append(emo)

    if not kept:
        return []

    stored = np.asarray(stored_keys, dtype=np.float32)          # (M, 9)
    slow = np.asarray(slow_now, dtype=np.float32)               # (9,)

    if mode == MODE_CONTENT:
        projected = np.asarray(
            project_batch_fn(np.asarray(semantics, dtype=np.float32)), dtype=np.float32,
        )                                                        # (M, 9) — one batch call
        predicted = _normalize_rows(projected - slow)            # broadcast re-centre
        sims = _row_cosine(stored, predicted)
    else:  # MODE_DCROSS
        sims = _row_cosine(stored, np.broadcast_to(slow, stored.shape))

    dissonance = 1.0 - sims                                       # (M,)

    # Selection score: optionally bias toward older memories.
    score = dissonance.copy()
    if age_halflife and age_halflife > 0 and now_ts is not None:
        ages = np.array(
            [max(now_ts - float((m.get("payload") or {}).get("game_ts", now_ts)), 0.0) for m in kept],
            dtype=np.float32,
        )
        score = dissonance * (1.0 + AGE_GAIN * (1.0 - np.exp(-ages / age_halflife)))

    results: list[ProbeResult] = []
    for idx in np.argsort(-score):
        raw = float(dissonance[idx])
        if raw < threshold:
            continue
        mem = kept[idx]
        results.append(ProbeResult(point_id=str(mem.get("id")), dissonance=raw, memory=mem))
        if len(results) >= top_k:
            break

    logger.debug(
        "Dissonance probe (mode=%s): scanned=%d eligible=%d selected=%d (threshold=%.2f)",
        mode, len(memories), len(kept), len(results), threshold,
    )
    return results


# ---------------------------------------------------------------------------
# Drift control + recurrence block (design #4)
# ---------------------------------------------------------------------------


def clamp_drift(
    stored_key: list[float] | np.ndarray,
    proposed_key: list[float] | np.ndarray,
    *,
    blend: float = DEFAULT_BLEND,
    max_drift: float = DEFAULT_MAX_DRIFT,
) -> list[float]:
    """Anchor a re-encoded reaction key near the stored key (no free-run).

    Blend toward the proposal rather than replacing it, then cap the per-pass
    change magnitude so a single night cannot yank the reaction arbitrarily far
    from what was originally felt. Returns the clamped 9d key as a list.
    """
    stored = np.asarray(stored_key, dtype=np.float32)
    proposed = np.asarray(proposed_key, dtype=np.float32)
    delta = blend * (proposed - stored)
    dist = float(np.linalg.norm(delta))
    if max_drift > 0 and dist > max_drift:
        delta = delta * (max_drift / dist)
    return (stored + delta).tolist()


def eligible_to_reconsolidate(
    prior_payload: Optional[dict[str, Any]],
    trigger_dissonance: float,
    *,
    threshold: float = DEFAULT_DISSONANCE_THRESHOLD,
    hysteresis: float = DEFAULT_HYSTERESIS,
) -> bool:
    """Trigger gate + recurrence block for one source memory.

    - No prior RECON: eligible once the trigger dissonance clears the threshold.
    - Prior RECON marked ``recon_stalled``: BLOCKED — a previous pass failed to
      resolve it, so we do not silently retry (the stall is flagged for analysis).
    - Prior RECON not stalled: require ``threshold + hysteresis`` so a
      marginally-changed source is not needlessly re-synthesized every night.
    """
    if prior_payload is None:
        return trigger_dissonance >= threshold
    if prior_payload.get("recon_stalled"):
        return False
    return trigger_dissonance >= threshold + hysteresis


def is_stalled_after(
    residual_dissonance: float,
    *,
    threshold: float = DEFAULT_DISSONANCE_THRESHOLD,
) -> bool:
    """True if the pass left dissonance >= threshold (failed to resolve)."""
    return residual_dissonance >= threshold


def next_version(prior_payload: Optional[dict[str, Any]]) -> int:
    """Monotonic RECON version: prior + 1, or 1 for a first reconsolidation."""
    return int(prior_payload.get("version", 0)) + 1 if prior_payload else 1


def next_attempts(prior_payload: Optional[dict[str, Any]]) -> int:
    """Cumulative reconsolidation attempts against a source set."""
    return int(prior_payload.get("recon_attempts", 0)) + 1 if prior_payload else 1


# ---------------------------------------------------------------------------
# Sleep-pass orchestration (design #5) — REM (probe) then SWS (reinterpret).
# ---------------------------------------------------------------------------


def reconsolidated_point_id(agent_id: str, raw_point_ids: list[str], version: int) -> str:
    """Deterministic RECON point id (a valid UUID) for one source set + version.

    Re-running the same pass version upserts in place; a later version gets a
    fresh id and supersedes the prior (JTMS chain).
    """
    key = f"mmk:recon:{agent_id}:{'|'.join(sorted(raw_point_ids))}:v{version}"
    return str(uuid5(NAMESPACE_DNS, key))


def _heuristic_gist(content: str, max_chars: int = 200) -> str:
    """Terse fallback gist when no LLM reinterpreter is wired (cf ArcCompressor)."""
    text = (content or "").strip()
    if not text:
        return ""
    head = re.split(r"(?<=[.!?])\s", text, maxsplit=1)[0]
    return head[:max_chars]


async def _prior_recon_by_source(agent_id: str, limit: int) -> dict[str, Memory]:
    """Map each source RAW id → its latest (highest-version) RECON for an agent.

    One scan per agent (compute-once); grouping is done in Python so we do not
    depend on array-membership filtering in the vector store. Vectors are
    included so the orchestrator can re-gate against the prior reinterpretation's
    key (a memory already resolved by a prior pass is only revisited if the mind
    has since drifted).
    """
    points = await scroll_filtered(
        COLLECTION_NPC_MEMORIES,
        Filter(must=[
            FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
            FieldCondition(key="tier", match=MatchValue(value=CompressionTier.RECON.value)),
        ]),
        limit=limit,
        with_vectors=True,
    )
    latest: dict[str, Memory] = {}
    for point in points:
        version = (point.get("payload") or {}).get("version", 0)
        for rid in (point.get("payload") or {}).get("raw_point_ids", []):
            current = latest.get(rid)
            if current is None or version > (current.get("payload") or {}).get("version", 0):
                latest[rid] = point
    return latest


async def run_reconsolidation(
    agent_ids: list[str],
    slow_buffer_fn: Callable[[str], list[float]],
    *,
    writer: MemoryWriter,
    reinterpret_fn: Optional[Callable[[str], str]] = None,
    top_k: int = DEFAULT_TOP_K,
    threshold: float = DEFAULT_DISSONANCE_THRESHOLD,
    hysteresis: float = DEFAULT_HYSTERESIS,
    blend: float = DEFAULT_BLEND,
    max_drift: float = DEFAULT_MAX_DRIFT,
    mode: str = MODE_CONTENT,
    scan_limit: int = DEFAULT_SCAN_LIMIT,
    age_halflife: Optional[float] = None,
    game_ts: Optional[float] = None,
    project_fn: Callable[[np.ndarray], list[float]] = emotional.project,
    project_batch_fn: Callable[[np.ndarray], np.ndarray] = emotional.project_batch,
) -> ReconReport:
    """Run the goodnight reconsolidation pass for a set of agents.

    REM (cheap): scan each agent's immutable RAW originals and probe-select the
    most dissonant few. SWS (expensive): for each selected source, re-encode the
    subjective key toward the current mind (drift-clamped), reinterpret the gist
    (injected ``reinterpret_fn``; terse heuristic fallback), and write a RECON
    point that supersedes any prior one. The source RAW is never modified.

    Re-gating against the latest reinterpretation prevents needless churn: a
    source already resolved by a prior pass is only revisited if the mind has
    drifted enough that the prior RECON key is itself dissonant again.

    Recurrence block (the user's caveat): a source whose latest RECON is
    ``recon_stalled`` is skipped, and a freshly-written RECON whose residual
    dissonance stays at/above ``threshold`` is itself flagged ``recon_stalled``
    and logged, so the same memory is not re-synthesized night after night.

    Non-fatal: runs offline and must never crash the caller; per-agent and
    per-memory failures are logged and skipped. Returns a ReconReport.
    """
    report = ReconReport()
    ts = game_ts if game_ts is not None else time.time()
    reinterpret = reinterpret_fn or _heuristic_gist

    for agent_id in agent_ids:
        report.scanned_agents += 1
        try:
            slow_now = list(slow_buffer_fn(agent_id))
            raw = await scroll_filtered(
                COLLECTION_NPC_MEMORIES,
                Filter(must=[
                    FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                    FieldCondition(key="tier", match=MatchValue(value=CompressionTier.RAW.value)),
                ]),
                limit=scan_limit,
                with_vectors=True,
                order_by="game_ts",
            )
        except Exception:
            logger.exception("Reconsolidation scan failed — agent=%s", agent_id)
            continue
        if not raw:
            continue

        prior_by_source = await _prior_recon_by_source(agent_id, scan_limit)
        selected = select_dissonant(
            raw, slow_now, top_k=top_k, threshold=threshold, mode=mode,
            now_ts=ts, age_halflife=age_halflife, project_batch_fn=project_batch_fn,
        )

        for result in selected:
            payload = result.memory.get("payload") or {}
            vectors = result.memory.get("vector") or {}
            semantic = vectors.get("semantic")
            stored_key = vectors.get("emotional")
            content = payload.get("content") or payload.get("text") or ""
            if semantic is None or stored_key is None or not content:
                continue

            prior = prior_by_source.get(result.point_id)
            prior_payload = prior["payload"] if prior else None

            predicted = predicted_reaction(semantic, slow_now, project_fn=project_fn)
            # Re-gate against the latest representation: the prior RECON key when
            # one exists (continue maturing from where we left off), else the RAW.
            if prior_payload is not None:
                base_key = (prior.get("vector") or {}).get("emotional") or stored_key
                trigger = memory_dissonance(base_key, predicted)
            else:
                base_key = stored_key
                trigger = result.dissonance

            if not eligible_to_reconsolidate(
                prior_payload, trigger, threshold=threshold, hysteresis=hysteresis,
            ):
                report.skipped_blocked += 1
                continue

            new_key = clamp_drift(base_key, predicted, blend=blend, max_drift=max_drift)
            residual = memory_dissonance(new_key, predicted)
            stalled = is_stalled_after(residual, threshold=threshold)
            version = next_version(prior_payload)
            point_id = reconsolidated_point_id(agent_id, [result.point_id], version)

            # Gist: the reinterpreter may be sync or async; empty falls back.
            gist = reinterpret(content)
            if inspect.isawaitable(gist):
                gist = await gist
            gist = (gist or "").strip() or _heuristic_gist(content)

            try:
                await writer.write_reconsolidated_summary(
                    agent_id=agent_id,
                    gist_text=gist,
                    semantic_vector=list(semantic),
                    emotional_vector=new_key,
                    raw_point_ids=[result.point_id],
                    game_ts=ts,
                    referents=payload.get("referents", []),
                    slow_snapshot=slow_now,
                    version=version,
                    dissonance_at_pass=trigger,
                    residual_dissonance=residual,
                    supersedes=(prior["id"] if prior else None),
                    recon_attempts=next_attempts(prior_payload),
                    recon_stalled=stalled,
                    point_id=point_id,
                )
            except Exception:
                logger.exception(
                    "RECON write failed — agent=%s source=%s", agent_id, result.point_id,
                )
                continue

            report.reconsolidated += 1
            if stalled:
                report.stalled += 1
                logger.warning(
                    "Reconsolidation STALLED — agent=%s source=%s residual=%.2f trigger=%.2f "
                    "v%d (recon_stalled flagged for analysis)",
                    agent_id, result.point_id, residual, trigger, version,
                )
            else:
                report.resolved += 1

    logger.info(
        "Reconsolidation pass: agents=%d reconsolidated=%d resolved=%d stalled=%d blocked=%d",
        report.scanned_agents, report.reconsolidated, report.resolved,
        report.stalled, report.skipped_blocked,
    )
    return report
