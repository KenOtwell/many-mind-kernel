"""Tests for progeny.src.compression (async function-based API).

Populates RAW data via MemoryWriter, then tests ArcCompressor and
EssenceDistiller against in-memory Qdrant.
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

import progeny.src.qdrant_client as client_mod
from progeny.src.qdrant_client import ensure_collections, get_points_by_ids
from progeny.src.memory_writer import MemoryWriter
from progeny.src.compression import ArcCompressor, EssenceDistiller
from shared.constants import COLLECTION_NPC_MEMORIES, EMOTIONAL_DIM, SEMANTIC_DIM


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
    v = [val] * SEMANTIC_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo(val: float = 0.5) -> list[float]:
    v = [val] * EMOTIONAL_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


# ---------------------------------------------------------------------------
# ArcCompressor
# ---------------------------------------------------------------------------

class TestArcCompressor:
    async def test_should_generate_arc_true_above_threshold(self):
        compressor = ArcCompressor(writer=MemoryWriter())
        assert compressor.should_generate_arc(0.5, threshold=0.3) is True

    async def test_should_generate_arc_false_below_threshold(self):
        compressor = ArcCompressor(writer=MemoryWriter())
        assert compressor.should_generate_arc(0.1, threshold=0.3) is False

    async def test_generate_arc_summary_with_raw_data(self, qdrant, writer):
        # Populate RAW points in a time window
        for i in range(3):
            await writer.write_raw_event(
                agent_id="Lydia", content=f"Battle event {i}.",
                semantic_vector=_sem(0.1 + i * 0.01),
                emotional_vector=_emo(0.1 + i * 0.05),
                game_ts=100.0 + i * 10, event_type="info",
            )

        compressor = ArcCompressor(writer=writer)
        arc_id = await compressor.generate_arc_summary(
            agent_id="Lydia",
            arc_start_ts=95.0,
            arc_end_ts=125.0,
            semantic_vector=_sem(),
            emotional_vector=_emo(),
            game_ts=125.0,
        )
        assert arc_id is not None

        # Verify the MOD point was written
        results = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [arc_id])
        assert len(results) == 1
        assert results[0]["payload"]["tier"] == "MOD"

    async def test_generate_arc_returns_none_for_empty_window(self, qdrant, writer):
        compressor = ArcCompressor(writer=writer)
        arc_id = await compressor.generate_arc_summary(
            agent_id="Lydia",
            arc_start_ts=0.0,
            arc_end_ts=1.0,
            semantic_vector=_sem(),
            emotional_vector=_emo(),
            game_ts=1.0,
        )
        assert arc_id is None

    async def test_heuristic_summarize_short_text(self):
        result = ArcCompressor._heuristic_summarize("One line only.")
        assert result == "One line only."

    async def test_heuristic_summarize_long_text(self):
        lines = "\n".join(f"Line {i} of the arc." for i in range(10))
        result = ArcCompressor._heuristic_summarize(lines)
        assert "Line 0" in result
        assert "Line 9" in result
        assert len(result) <= 500


# ---------------------------------------------------------------------------
# EssenceDistiller
# ---------------------------------------------------------------------------

class TestEssenceDistiller:
    async def test_distill_arcs_writes_max_tier(self, qdrant, writer):
        # Write some MOD-tier arc summaries
        arc_ids = []
        for i in range(2):
            pid = await writer.write_arc_summary(
                agent_id="Lydia", summary_text=f"Arc summary {i}.",
                semantic_vector=_sem(0.1 + i * 0.01),
                emotional_vector=_emo(0.1 + i * 0.05),
                arc_start_ts=float(i * 100), arc_end_ts=float(i * 100 + 50),
                raw_point_ids=[], game_ts=float(i * 100 + 50),
            )
            arc_ids.append(pid)

        distiller = EssenceDistiller(writer=writer)
        essence_id = await distiller.distill_arcs(
            agent_id="Lydia",
            arc_point_ids=arc_ids,
            semantic_vector=_sem(),
            emotional_vector=_emo(),
            game_ts=500.0,
        )
        assert essence_id is not None

        results = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [essence_id])
        assert results[0]["payload"]["tier"] == "MAX"

    async def test_distill_returns_none_for_missing_arcs(self, qdrant, writer):
        distiller = EssenceDistiller(writer=writer)
        result = await distiller.distill_arcs(
            agent_id="Lydia",
            arc_point_ids=["nonexistent-1", "nonexistent-2"],
            semantic_vector=_sem(),
            emotional_vector=_emo(),
            game_ts=100.0,
        )
        assert result is None
