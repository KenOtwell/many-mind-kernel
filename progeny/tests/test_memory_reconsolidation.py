"""Tests for sleep-time memory reconsolidation.

Phase T1 (design #1 — reconsolidate the subjective key): the reconsolidation
writer produces a NEW derived RECON point that re-encodes the subjective key and
carries a reframed gist, while the source RAW stays immutable and recoverable.

In-memory Qdrant + hand vectors, no embedding model and no LLM (mirrors the
fixture pattern in test_social_bootstrapping.py).
"""
from __future__ import annotations

from uuid import NAMESPACE_DNS, uuid5

import numpy as np
import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

import progeny.src.qdrant_client as client_mod
from progeny.src import memory_reconsolidation as recon
from progeny.src.memory_retrieval import MemoryRetriever, RetrievalResult
from progeny.src.memory_writer import MemoryWriter
from progeny.src.qdrant_client import ensure_collections, get_points_by_ids, scroll_filtered
from shared.constants import COLLECTION_NPC_MEMORIES, EMOTIONAL_DIM, SEMANTIC_DIM
from shared.schemas import CompressionTier


@pytest.fixture
async def qdrant():
    mem = AsyncQdrantClient(location=":memory:")
    client_mod.configure(mem)
    await ensure_collections()
    yield mem
    client_mod.configure(None)


@pytest.fixture
def writer():
    return MemoryWriter()


def _sem(val: float = 0.1) -> list[float]:
    """A normalized 384d semantic vector (cosine-friendly)."""
    v = [val] * SEMANTIC_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo(*vals: float) -> list[float]:
    """A 9d emotional vector; positional args fill the leading axes."""
    if not vals:
        return [0.2] * EMOTIONAL_DIM
    out = list(vals) + [0.0] * (EMOTIONAL_DIM - len(vals))
    return out[:EMOTIONAL_DIM]


def _pid(name: str) -> str:
    """Deterministic, valid-UUID point id (Qdrant requires UUID or uint ids)."""
    return str(uuid5(NAMESPACE_DNS, name))


def _recon_filter(agent_id: str) -> Filter:
    return Filter(must=[
        FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
        FieldCondition(key="tier", match=MatchValue(value=CompressionTier.RECON.value)),
    ])


class TestReconWriter:
    async def test_writes_recon_point_with_provenance(self, qdrant, writer):
        pid = await writer.write_reconsolidated_summary(
            agent_id="Lydia",
            gist_text="The ambush taught me to scout ahead — I handle that road calmly now.",
            semantic_vector=_sem(0.2),
            emotional_vector=_emo(0.0, 0.0, 0.3, 0.0, 0.0, 0.0, 0.4, 0.5),
            raw_point_ids=["raw-1", "raw-2"],
            game_ts=100.0,
            referents=["Bandit"],
            slow_snapshot=_emo(0.1, 0.0, 0.2),
            version=1,
            dissonance_at_pass=0.62,
        )
        pts = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [pid], with_vectors=True)
        assert len(pts) == 1
        payload = pts[0]["payload"]
        assert payload["tier"] == CompressionTier.RECON.value
        assert payload["event_type"] == "reconsolidated_summary"
        assert payload["raw_point_ids"] == ["raw-1", "raw-2"]
        assert payload["referents"] == ["Bandit"]
        assert payload["version"] == 1
        assert payload["dissonance_at_pass"] == pytest.approx(0.62)
        assert payload["residual_dissonance"] == pytest.approx(0.0)
        assert payload["supersedes"] is None
        assert payload["recon_attempts"] == 1
        assert payload["recon_stalled"] is False
        assert len(payload["slow_snapshot"]) == EMOTIONAL_DIM
        vectors = pts[0]["vector"]
        assert set(vectors.keys()) == {"semantic", "emotional"}
        assert len(vectors["semantic"]) == SEMANTIC_DIM
        assert len(vectors["emotional"]) == EMOTIONAL_DIM

    async def test_source_raw_is_immutable(self, qdrant, writer):
        # The original memory: a frightened, angry reaction to an ambush.
        raw_id = await writer.write_raw_event(
            agent_id="Lydia",
            content="A bandit ambushed us on the road.",
            semantic_vector=_sem(0.2),
            emotional_vector=_emo(0.8, 0.6),  # fear, anger — the reaction at the time
            game_ts=10.0,
            event_type="event",
            referents=["Bandit"],
            point_id=_pid("raw-immutable"),
        )
        before = (await get_points_by_ids(
            COLLECTION_NPC_MEMORIES, [raw_id], with_vectors=True,
        ))[0]

        # Reconsolidate: anchor semantic to the ORIGINAL content, re-encode the key.
        await writer.write_reconsolidated_summary(
            agent_id="Lydia",
            gist_text="Reframed: I was afraid then; I'm seasoned on that road now.",
            semantic_vector=before["vector"]["semantic"],   # anchored content
            emotional_vector=_emo(0.0, 0.0, 0.2, 0.0, 0.0, 0.0, 0.3, 0.5),  # re-encoded
            raw_point_ids=[raw_id],
            game_ts=200.0,
        )

        after = (await get_points_by_ids(
            COLLECTION_NPC_MEMORIES, [raw_id], with_vectors=True,
        ))[0]
        # The RAW point is untouched: content, tier, and both vectors unchanged.
        assert after["payload"]["content"] == before["payload"]["content"]
        assert after["payload"]["tier"] == CompressionTier.RAW.value
        assert after["vector"]["emotional"] == before["vector"]["emotional"]
        assert after["vector"]["semantic"] == before["vector"]["semantic"]

    async def test_deterministic_id_is_idempotent(self, qdrant, writer):
        kwargs = dict(
            agent_id="Lydia",
            gist_text="v1 gist",
            semantic_vector=_sem(0.2),
            emotional_vector=_emo(0.3),
            raw_point_ids=["raw-1"],
            game_ts=100.0,
            point_id=_pid("recon-fixed"),
        )
        await writer.write_reconsolidated_summary(**kwargs)
        await writer.write_reconsolidated_summary(**kwargs)
        recon = await scroll_filtered(COLLECTION_NPC_MEMORIES, _recon_filter("Lydia"))
        assert len(recon) == 1  # same id -> upsert in place, no duplicate

    async def test_supersession_fields_persist(self, qdrant, writer):
        v1 = await writer.write_reconsolidated_summary(
            agent_id="Lydia", gist_text="v1", semantic_vector=_sem(0.2),
            emotional_vector=_emo(0.3), raw_point_ids=["raw-1"], game_ts=100.0,
            version=1, point_id=_pid("recon-v1"),
        )
        v2 = await writer.write_reconsolidated_summary(
            agent_id="Lydia", gist_text="v2 — more mature", semantic_vector=_sem(0.2),
            emotional_vector=_emo(0.1), raw_point_ids=["raw-1"], game_ts=300.0,
            version=2, supersedes=v1, recon_attempts=2, point_id=_pid("recon-v2"),
        )
        payload = (await get_points_by_ids(COLLECTION_NPC_MEMORIES, [v2]))[0]["payload"]
        assert payload["version"] == 2
        assert payload["supersedes"] == v1
        assert payload["recon_attempts"] == 2

    async def test_stalled_flag_persists(self, qdrant, writer):
        pid = await writer.write_reconsolidated_summary(
            agent_id="Lydia", gist_text="still unresolved", semantic_vector=_sem(0.2),
            emotional_vector=_emo(0.3), raw_point_ids=["raw-1"], game_ts=100.0,
            dissonance_at_pass=0.71, residual_dissonance=0.68, recon_stalled=True,
        )
        payload = (await get_points_by_ids(COLLECTION_NPC_MEMORIES, [pid]))[0]["payload"]
        assert payload["recon_stalled"] is True
        assert payload["residual_dissonance"] == pytest.approx(0.68)


# ---------------------------------------------------------------------------
# Dissonance probe (T2 / design #2) — pure functions, no model, no Qdrant.
# ---------------------------------------------------------------------------


def _mem(
    point_id: str,
    emotional: list[float],
    semantic: list[float] | None = None,
    game_ts: float = 0.0,
) -> dict:
    """A scanned-memory dict shaped like scroll_filtered(with_vectors=True)."""
    vectors: dict = {"emotional": emotional}
    if semantic is not None:
        vectors["semantic"] = semantic
    return {"id": point_id, "payload": {"game_ts": game_ts}, "vector": vectors}


class TestDissonanceProbe:
    def test_memory_dissonance_bounds(self):
        key = [1.0, 0, 0, 0, 0, 0, 0, 0, 0]
        assert recon.memory_dissonance(key, key) == pytest.approx(0.0, abs=1e-6)
        assert recon.memory_dissonance(
            key, [0, 1.0, 0, 0, 0, 0, 0, 0, 0],
        ) == pytest.approx(1.0, abs=1e-6)
        assert recon.memory_dissonance(
            key, [-1.0, 0, 0, 0, 0, 0, 0, 0, 0],
        ) == pytest.approx(2.0, abs=1e-6)

    def test_predicted_reaction_recenters_on_baseline(self):
        # project_fn returns a fixed projected vector; the baseline shifts it.
        proj = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0]  # joy + safety
        slow = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]  # already-joyful baseline
        pred = recon.predicted_reaction([0.1] * SEMANTIC_DIM, slow, project_fn=lambda e: proj)
        # proj - slow = safety only -> normalized points at the safety axis (dim 7).
        assert pred[7] == pytest.approx(1.0, abs=1e-6)
        assert pred[6] == pytest.approx(0.0, abs=1e-6)

    def test_select_dissonant_content_ranks_and_thresholds(self):
        slow = [0.0] * EMOTIONAL_DIM
        key = [1.0] + [0.0] * 8
        mems = [
            _mem("low", key, _sem(0.1)),
            _mem("mid", key, _sem(0.2)),
            _mem("high", key, _sem(0.3)),
        ]
        # slow=0 -> predicted = normalize(projected): aligned / orthogonal / inverted.
        projected = np.array([
            [1.0] + [0.0] * 8,        # low:  cos 1  -> diss 0
            [0.0, 1.0] + [0.0] * 7,   # mid:  cos 0  -> diss 1
            [-1.0] + [0.0] * 8,       # high: cos -1 -> diss 2
        ], dtype=np.float32)

        def fake_batch(embs):
            assert embs.shape[0] == 3  # one batch call over all kept memories
            return projected

        out = recon.select_dissonant(
            mems, slow, top_k=2, threshold=0.5, mode=recon.MODE_CONTENT,
            project_batch_fn=fake_batch,
        )
        assert [r.point_id for r in out] == ["high", "mid"]  # desc; low filtered out
        assert out[0].dissonance == pytest.approx(2.0, abs=1e-6)

    def test_select_dissonant_excludes_ids(self):
        slow = [0.0] * EMOTIONAL_DIM
        key = [1.0] + [0.0] * 8
        mems = [_mem("a", key, _sem(0.1)), _mem("b", key, _sem(0.2))]
        out = recon.select_dissonant(
            mems, slow, threshold=0.5, exclude_ids={"a"},
            project_batch_fn=lambda e: np.array([[-1.0] + [0.0] * 8] * e.shape[0], dtype=np.float32),
        )
        assert [r.point_id for r in out] == ["b"]

    def test_content_mode_projects_once(self):
        slow = [0.0] * EMOTIONAL_DIM
        key = [1.0] + [0.0] * 8
        mems = [_mem(str(i), key, _sem(0.1 + i * 0.01)) for i in range(5)]
        calls = {"n": 0}

        def counting_batch(embs):
            calls["n"] += 1
            return np.tile(np.array([[-1.0] + [0.0] * 8], dtype=np.float32), (embs.shape[0], 1))

        recon.select_dissonant(mems, slow, project_batch_fn=counting_batch)
        assert calls["n"] == 1  # compute-once: a single batched projection

    def test_dcross_mode_skips_projection(self):
        slow = [1.0] + [0.0] * 8
        mems = [
            _mem("aligned", [1.0] + [0.0] * 8),    # cos 1  -> diss 0
            _mem("opposed", [-1.0] + [0.0] * 8),   # cos -1 -> diss 2
        ]
        calls = {"n": 0}

        def must_not_call(embs):
            calls["n"] += 1
            return embs

        out = recon.select_dissonant(
            mems, slow, threshold=0.5, mode=recon.MODE_DCROSS, project_batch_fn=must_not_call,
        )
        assert calls["n"] == 0  # dcross needs no content projection
        assert [r.point_id for r in out] == ["opposed"]


# ---------------------------------------------------------------------------
# Recall preference (T3 / design #3) — RECON supersedes its source RAW/MOD,
# original RAW kept lazy in expandable_refs.
# ---------------------------------------------------------------------------


def _anchor(
    point_id: str,
    content: str,
    tier: str,
    *,
    raw_point_ids: list[str] | None = None,
    game_ts: float = 0.0,
    score: float = 1.0,
) -> RetrievalResult:
    payload: dict = {"game_ts": game_ts}
    if raw_point_ids is not None:
        payload["raw_point_ids"] = raw_point_ids
    return RetrievalResult(
        point_id=point_id, content=content, tier=tier, score=score, payload=payload,
    )


class TestRecallPrefersRecon:
    async def test_recon_suppresses_source_raw_and_keeps_it_lazy(self):
        retriever = MemoryRetriever()
        anchors = [
            _anchor("raw-1", "A bandit ambushed us.", "RAW", game_ts=1.0),
            _anchor(
                "recon-1", "I handle that road calmly now.", "RECON",
                raw_point_ids=["raw-1"], game_ts=2.0,
            ),
        ]
        bundle = await retriever._expand_to_bundle("Lydia", anchors)
        assert [s["text"] for s in bundle.summaries] == ["I handle that road calmly now."]
        assert bundle.summaries[0]["tier"] == "RECON"
        assert bundle.recent == []                  # source RAW suppressed
        assert bundle.expandable_refs == ["raw-1"]  # ...but kept one hop away

    async def test_uncovered_raw_still_surfaces(self):
        retriever = MemoryRetriever()
        anchors = [
            _anchor("raw-1", "Unrelated memory.", "RAW", game_ts=1.0),
            _anchor(
                "recon-2", "Reframed other memory.", "RECON",
                raw_point_ids=["raw-2"], game_ts=2.0,
            ),
        ]
        bundle = await retriever._expand_to_bundle("Lydia", anchors)
        assert [r["text"] for r in bundle.recent] == ["Unrelated memory."]
        assert bundle.expandable_refs == ["raw-2"]
        assert len(bundle.summaries) == 1

    async def test_mod_fully_covered_by_recon_is_dropped(self):
        retriever = MemoryRetriever()
        anchors = [
            _anchor(
                "mod-1", "Arc summary.", "MOD",
                raw_point_ids=["raw-1", "raw-2"], game_ts=1.0,
            ),
            _anchor(
                "recon-3", "Reframed arc.", "RECON",
                raw_point_ids=["raw-1", "raw-2"], game_ts=2.0,
            ),
        ]
        bundle = await retriever._expand_to_bundle("Lydia", anchors)
        assert [s["tier"] for s in bundle.summaries] == ["RECON"]  # MOD dropped
        assert bundle.recent == []
        assert set(bundle.expandable_refs) == {"raw-1", "raw-2"}

    async def test_partially_covered_mod_pulls_only_uncovered_raw(self, qdrant, writer):
        # raw-3 is uncovered and must be fetched into recent[]; raw-1 is covered.
        raw3 = _pid("raw-3")
        await writer.write_raw_event(
            agent_id="Lydia", content="Uncovered detail.", semantic_vector=_sem(0.2),
            emotional_vector=_emo(0.3), game_ts=5.0, event_type="event", point_id=raw3,
        )
        retriever = MemoryRetriever()
        anchors = [
            _anchor(
                "mod-1", "Arc summary.", "MOD",
                raw_point_ids=[_pid("raw-1"), raw3], game_ts=1.0,
            ),
            _anchor(
                "recon-4", "Reframed.", "RECON",
                raw_point_ids=[_pid("raw-1")], game_ts=2.0,
            ),
        ]
        bundle = await retriever._expand_to_bundle("Lydia", anchors)
        # MOD kept (not fully covered); only the uncovered raw-3 pulled; raw-1 lazy.
        assert any(s["tier"] == "MOD" for s in bundle.summaries)
        assert any(s["tier"] == "RECON" for s in bundle.summaries)
        assert [r["text"] for r in bundle.recent] == ["Uncovered detail."]
        assert _pid("raw-1") in bundle.expandable_refs


# ---------------------------------------------------------------------------
# Drift control + recurrence block (T4 / design #4)
# ---------------------------------------------------------------------------


class TestDriftControl:
    def test_clamp_drift_blends_not_replaces(self):
        stored = [1.0] + [0.0] * 8
        proposed = [0.0, 1.0] + [0.0] * 7
        out = recon.clamp_drift(stored, proposed, blend=0.5, max_drift=10.0)
        assert out[0] == pytest.approx(0.5, abs=1e-6)
        assert out[1] == pytest.approx(0.5, abs=1e-6)

    def test_clamp_drift_caps_per_pass(self):
        stored = [0.0] * EMOTIONAL_DIM
        proposed = [10.0] + [0.0] * 8
        out = recon.clamp_drift(stored, proposed, blend=1.0, max_drift=0.5)
        moved = float(np.linalg.norm(np.array(out) - np.array(stored)))
        assert moved == pytest.approx(0.5, abs=1e-6)


class TestRecurrenceBlock:
    def test_new_source_eligible_over_threshold(self):
        assert recon.eligible_to_reconsolidate(None, 0.6, threshold=0.5) is True
        assert recon.eligible_to_reconsolidate(None, 0.4, threshold=0.5) is False

    def test_stalled_prior_is_blocked(self):
        prior = {"recon_stalled": True, "version": 1, "recon_attempts": 1}
        assert recon.eligible_to_reconsolidate(prior, 0.9, threshold=0.5) is False

    def test_prior_requires_hysteresis_margin(self):
        prior = {"recon_stalled": False, "version": 1, "recon_attempts": 1}
        assert recon.eligible_to_reconsolidate(prior, 0.55, threshold=0.5, hysteresis=0.1) is False
        assert recon.eligible_to_reconsolidate(prior, 0.65, threshold=0.5, hysteresis=0.1) is True

    def test_is_stalled_after(self):
        assert recon.is_stalled_after(0.6, threshold=0.5) is True
        assert recon.is_stalled_after(0.4, threshold=0.5) is False

    def test_version_and_attempts_increment(self):
        assert recon.next_version(None) == 1
        assert recon.next_attempts(None) == 1
        prior = {"version": 1, "recon_attempts": 2}
        assert recon.next_version(prior) == 2
        assert recon.next_attempts(prior) == 3


# ---------------------------------------------------------------------------
# Sleep-pass orchestration (T5 / design #5) — REM->SWS, in-memory Qdrant.
# ---------------------------------------------------------------------------


def _fake_single(_emb):
    return [1.0] + [0.0] * 8


def _fake_batch(embs):
    return np.tile(np.array([[1.0] + [0.0] * 8], dtype=np.float32), (embs.shape[0], 1))


def _fake_single_b(_emb):
    return [0.0, 0.0, 1.0] + [0.0] * 6


def _fake_batch_b(embs):
    return np.tile(np.array([[0.0, 0.0, 1.0] + [0.0] * 6], dtype=np.float32), (embs.shape[0], 1))


def _zero_slow(_agent_id):
    return [0.0] * EMOTIONAL_DIM


class TestRunReconsolidation:
    async def test_writes_recon_for_dissonant_skips_consonant(self, qdrant, writer):
        agent = "Lydia"
        await writer.write_raw_event(
            agent_id=agent, content="A calm market day.", semantic_vector=_sem(0.2),
            emotional_vector=[1.0] + [0.0] * 8, game_ts=1.0, event_type="event",
            point_id=_pid("raw-consonant"),
        )
        await writer.write_raw_event(
            agent_id=agent, content="The duel that terrified me.", semantic_vector=_sem(0.3),
            emotional_vector=[0.0, 1.0] + [0.0] * 7, game_ts=2.0, event_type="event",
            point_id=_pid("raw-dissonant"),
        )
        report = await recon.run_reconsolidation(
            [agent], _zero_slow, writer=writer, reinterpret_fn=lambda c: f"REFRAMED: {c}",
            threshold=0.5, blend=1.0, max_drift=10.0, game_ts=100.0,
            project_fn=_fake_single, project_batch_fn=_fake_batch,
        )
        assert (report.reconsolidated, report.resolved, report.stalled) == (1, 1, 0)
        points = await scroll_filtered(COLLECTION_NPC_MEMORIES, _recon_filter(agent))
        assert len(points) == 1
        payload = points[0]["payload"]
        assert payload["raw_point_ids"] == [_pid("raw-dissonant")]
        assert payload["content"] == "REFRAMED: The duel that terrified me."
        assert payload["recon_stalled"] is False

    async def test_stalls_and_flags_when_unresolved(self, qdrant, writer):
        agent = "Lydia"
        await writer.write_raw_event(
            agent_id=agent, content="An inverted feeling.", semantic_vector=_sem(0.3),
            emotional_vector=[-1.0] + [0.0] * 8, game_ts=2.0, event_type="event",
            point_id=_pid("raw-inverted"),
        )
        report = await recon.run_reconsolidation(
            [agent], _zero_slow, writer=writer, reinterpret_fn=lambda c: f"REFRAMED: {c}",
            threshold=0.5, blend=0.5, max_drift=0.5, game_ts=100.0,
            project_fn=_fake_single, project_batch_fn=_fake_batch,
        )
        assert report.stalled == 1
        payload = (await scroll_filtered(COLLECTION_NPC_MEMORIES, _recon_filter(agent)))[0]["payload"]
        assert payload["recon_stalled"] is True
        assert payload["residual_dissonance"] >= 0.5

    async def test_recurrence_block_skips_stalled_source(self, qdrant, writer):
        agent = "Lydia"
        src = _pid("raw-inverted")
        await writer.write_raw_event(
            agent_id=agent, content="An inverted feeling.", semantic_vector=_sem(0.3),
            emotional_vector=[-1.0] + [0.0] * 8, game_ts=2.0, event_type="event", point_id=src,
        )
        await writer.write_reconsolidated_summary(
            agent_id=agent, gist_text="prior attempt", semantic_vector=_sem(0.3),
            emotional_vector=[-0.5] + [0.0] * 8, raw_point_ids=[src], game_ts=50.0,
            version=1, residual_dissonance=0.9, recon_stalled=True, point_id=_pid("recon-prior"),
        )
        report = await recon.run_reconsolidation(
            [agent], _zero_slow, writer=writer, reinterpret_fn=lambda c: f"REFRAMED: {c}",
            threshold=0.5, blend=0.5, max_drift=0.5, game_ts=100.0,
            project_fn=_fake_single, project_batch_fn=_fake_batch,
        )
        assert report.skipped_blocked == 1
        assert report.reconsolidated == 0
        points = await scroll_filtered(COLLECTION_NPC_MEMORIES, _recon_filter(agent))
        assert len(points) == 1  # only the prior stalled RECON remains

    async def test_resolved_memory_not_reprocessed_when_mind_stable(self, qdrant, writer):
        agent = "Lydia"
        await writer.write_raw_event(
            agent_id=agent, content="A duel.", semantic_vector=_sem(0.3),
            emotional_vector=[0.0, 1.0] + [0.0] * 7, game_ts=2.0, event_type="event",
            point_id=_pid("raw-ortho"),
        )
        common = dict(
            writer=writer, reinterpret_fn=lambda c: "gist", threshold=0.5,
            blend=1.0, max_drift=10.0, project_fn=_fake_single, project_batch_fn=_fake_batch,
        )
        r1 = await recon.run_reconsolidation([agent], _zero_slow, game_ts=100.0, **common)
        r2 = await recon.run_reconsolidation([agent], _zero_slow, game_ts=200.0, **common)
        assert r1.reconsolidated == 1
        assert r2.reconsolidated == 0       # already resolved + mind stable -> skip
        assert r2.skipped_blocked == 1
        points = await scroll_filtered(COLLECTION_NPC_MEMORIES, _recon_filter(agent))
        assert len(points) == 1             # no churn

    async def test_second_pass_supersedes_when_mind_drifts(self, qdrant, writer):
        agent = "Lydia"
        src = _pid("raw-ortho")
        await writer.write_raw_event(
            agent_id=agent, content="A duel.", semantic_vector=_sem(0.3),
            emotional_vector=[0.0, 1.0] + [0.0] * 7, game_ts=2.0, event_type="event", point_id=src,
        )
        # Pass 1: current reaction points at axis 0 -> resolve toward it (v1).
        await recon.run_reconsolidation(
            [agent], _zero_slow, writer=writer, reinterpret_fn=lambda c: "v1",
            threshold=0.5, blend=1.0, max_drift=10.0, game_ts=100.0,
            project_fn=_fake_single, project_batch_fn=_fake_batch,
        )
        # Pass 2: the mind has matured (reaction now points at axis 2), so the v1
        # key is dissonant again -> v2 supersedes v1.
        report = await recon.run_reconsolidation(
            [agent], _zero_slow, writer=writer, reinterpret_fn=lambda c: "v2",
            threshold=0.5, hysteresis=0.0, blend=1.0, max_drift=10.0, game_ts=200.0,
            project_fn=_fake_single_b, project_batch_fn=_fake_batch_b,
        )
        assert report.reconsolidated == 1
        points = await scroll_filtered(COLLECTION_NPC_MEMORIES, _recon_filter(agent))
        assert sorted(p["payload"]["version"] for p in points) == [1, 2]
        v2 = next(p for p in points if p["payload"]["version"] == 2)["payload"]
        assert v2["supersedes"] == recon.reconsolidated_point_id(agent, [src], 1)


# ---------------------------------------------------------------------------
# Goodnight trigger wiring (T6 / design #5) — end-to-end through the route.
# ---------------------------------------------------------------------------


class TestGoodnightTrigger:
    async def test_goodnight_runs_reconsolidation(self, qdrant, writer, monkeypatch):
        from progeny.api import routes
        from shared import emotional as shared_emotional
        from shared.schemas import AckResponse, TickPackage, TypedEvent

        agent = "Lydia"
        src = _pid("raw-e2e")
        # Stored reaction is the inverse of how the content now projects, so the
        # memory is guaranteed dissonant under the real projection.
        real_proj = list(shared_emotional.project(np.array(_sem(0.3), dtype=np.float32)))
        await writer.write_raw_event(
            agent_id=agent, content="The ambush on the road.", semantic_vector=_sem(0.3),
            emotional_vector=[-v for v in real_proj], game_ts=1.0, event_type="event",
            referents=["Bandit"], point_id=src,
        )

        # Seed live emotional state (slow baseline ~ 0) so the agent is known.
        routes._harmonic_state.reset()
        routes._accumulator._handle_reset()
        routes._harmonic_state.update(agent, [0.0] * EMOTIONAL_DIM)

        # Avoid the real LLM: deterministic reframing.
        monkeypatch.setattr(routes, "_reconsolidation_reinterpreter", lambda c: f"REFRAMED: {c}")

        pkg = TickPackage(events=[
            TypedEvent(event_type="goodnight", local_ts="t", game_ts=1.0, raw_data="goodnight"),
        ])
        try:
            result = await routes._ingest_inner(pkg)
            assert isinstance(result, AckResponse)
            points = await scroll_filtered(COLLECTION_NPC_MEMORIES, _recon_filter(agent))
            assert len(points) == 1
            assert points[0]["payload"]["raw_point_ids"] == [src]
            assert points[0]["payload"]["content"].startswith("REFRAMED:")
        finally:
            routes._harmonic_state.reset()
            routes._accumulator._handle_reset()
