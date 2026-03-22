"""Tests for progeny.src.rehydration (async function-based API).

Tests ref expansion (MAX→MOD→RAW chain), stabilization detection,
and post-interruption stash recovery.
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

import progeny.src.qdrant_client as client_mod
from progeny.src.qdrant_client import ensure_collections, get_points_by_ids
from progeny.src.memory_writer import MemoryWriter
from progeny.src.rehydration import Rehydrator
from shared.constants import (
    COLLECTION_NPC_MEMORIES,
    COLLECTION_SESSION_CONTEXT,
    EMOTIONAL_DIM,
    SEMANTIC_DIM,
)


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


@pytest.fixture
def rehydrator():
    return Rehydrator()


def _sem(val: float = 0.1) -> list[float]:
    v = [val] * SEMANTIC_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo(val: float = 0.5) -> list[float]:
    v = [val] * EMOTIONAL_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


# ---------------------------------------------------------------------------
# check_stabilization (pure logic, no Qdrant)
# ---------------------------------------------------------------------------

class TestCheckStabilization:
    def test_not_stable_initially(self):
        r = Rehydrator()
        assert r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3) is False

    def test_stable_after_enough_ticks(self):
        r = Rehydrator()
        r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        assert r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3) is True

    def test_resets_on_curvature_spike(self):
        r = Rehydrator()
        r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        # Spike resets counter
        r.check_stabilization("Lydia", 0.5, threshold=0.1, required_ticks=3)
        assert r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3) is False

    def test_independent_per_agent(self):
        r = Rehydrator()
        for _ in range(3):
            r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        # Lydia is stable, Belethor is not
        assert r.check_stabilization("Belethor", 0.05, threshold=0.1, required_ticks=3) is False

    def test_reset_agent_clears_counter(self):
        r = Rehydrator()
        r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        r.reset_agent("Lydia")
        assert r.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3) is False


# ---------------------------------------------------------------------------
# expand_refs (MAX → MOD → RAW chain)
# ---------------------------------------------------------------------------

class TestExpandRefs:
    async def test_empty_refs_returns_empty(self, qdrant, rehydrator):
        results = await rehydrator.expand_refs([])
        assert results == []

    async def test_expands_max_to_raw(self, qdrant, writer, rehydrator):
        # Build the chain: RAW → MOD (arc summary with raw_point_ids) → MAX (with source_arc_ids)
        raw_id = await writer.write_raw_event(
            agent_id="Lydia", content="Fought a dragon.",
            semantic_vector=_sem(), emotional_vector=_emo(),
            game_ts=100.0, event_type="info",
        )
        arc_id = await writer.write_arc_summary(
            agent_id="Lydia", summary_text="Dragon fight arc.",
            semantic_vector=_sem(), emotional_vector=_emo(),
            arc_start_ts=90.0, arc_end_ts=110.0,
            raw_point_ids=[raw_id], game_ts=110.0,
        )
        max_id = await writer.write_compressed_essence(
            agent_id="Lydia", essence_text="Veteran fighter.",
            semantic_vector=_sem(), emotional_vector=_emo(),
            source_arc_ids=[arc_id], game_ts=500.0,
        )

        results = await rehydrator.expand_refs([max_id])
        assert len(results) >= 1
        texts = [r["text"] for r in results]
        assert any("dragon" in t.lower() for t in texts)

    async def test_missing_refs_returns_empty(self, qdrant, rehydrator):
        results = await rehydrator.expand_refs(["nonexistent-id"])
        assert results == []


# ---------------------------------------------------------------------------
# recover_stashed_context
# ---------------------------------------------------------------------------

class TestRecoverStashedContext:
    async def test_recovers_unrehydrated_stash(self, qdrant, writer, rehydrator):
        await writer.stash_session_context(
            agent_id="Lydia", context_text="We were talking about the war.",
            semantic_vector=_sem(),
        )
        results = await rehydrator.recover_stashed_context("Lydia")
        assert len(results) == 1
        assert "war" in results[0]["text"]

    async def test_marks_stash_as_rehydrated(self, qdrant, writer, rehydrator):
        await writer.stash_session_context(
            agent_id="Lydia", context_text="Discussion interrupted.",
            semantic_vector=_sem(),
        )
        # First recovery
        await rehydrator.recover_stashed_context("Lydia")
        # Second recovery should find nothing (already rehydrated)
        results = await rehydrator.recover_stashed_context("Lydia")
        assert results == []

    async def test_no_stash_returns_empty(self, qdrant, rehydrator):
        results = await rehydrator.recover_stashed_context("Nobody")
        assert results == []

    async def test_resets_stabilization_counter(self, qdrant, writer, rehydrator):
        # Build up stabilization counter
        rehydrator.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        rehydrator.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3)
        # Stash and recover
        await writer.stash_session_context(
            agent_id="Lydia", context_text="Context.",
            semantic_vector=_sem(),
        )
        await rehydrator.recover_stashed_context("Lydia")
        # Counter should be reset — not immediately stable
        assert rehydrator.check_stabilization("Lydia", 0.05, threshold=0.1, required_ticks=3) is False
