"""Tests for progeny.src.memory_writer (async function-based API).

Uses the in-memory Qdrant fixture from test_qdrant_client.
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

import progeny.src.qdrant_client as client_mod
from progeny.src.qdrant_client import ensure_collections, get_points_by_ids
from progeny.src.memory_writer import MemoryWriter
from shared.constants import (
    COLLECTION_AGENT_STATE,
    COLLECTION_LORE,
    COLLECTION_NPC_MEMORIES,
    COLLECTION_SESSION_CONTEXT,
    COLLECTION_WORLD_EVENTS,
    EMOTIONAL_DIM,
    SEMANTIC_DIM,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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
# RAW tier
# ---------------------------------------------------------------------------

class TestWriteRawEvent:
    async def test_returns_uuid(self, qdrant, writer):
        import uuid
        pid = await writer.write_raw_event(
            agent_id="Lydia", content="Drew her sword.", semantic_vector=_sem(),
            emotional_vector=_emo(), game_ts=100.0, event_type="info",
        )
        uuid.UUID(pid)

    async def test_payload_correct(self, qdrant, writer):
        pid = await writer.write_raw_event(
            agent_id="Lydia", content="She spoke.", semantic_vector=_sem(),
            emotional_vector=_emo(), game_ts=200.0, event_type="_speech",
            referents=["Belethor"], location="WhiterunMarket",
        )
        results = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [pid])
        p = results[0]["payload"]
        assert p["agent_id"] == "Lydia"
        assert p["tier"] == "RAW"
        assert p["event_type"] == "_speech"
        assert p["referents"] == ["Belethor"]
        assert p["location"] == "WhiterunMarket"

    async def test_extra_payload_merged(self, qdrant, writer):
        pid = await writer.write_raw_event(
            agent_id="Lydia", content="x", semantic_vector=_sem(),
            emotional_vector=_emo(), game_ts=1.0, event_type="info",
            extra_payload={"custom": True},
        )
        results = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [pid])
        assert results[0]["payload"]["custom"] is True


class TestWriteRawBatch:
    async def test_batch_writes_all(self, qdrant, writer):
        events = [
            {"agent_id": "Lydia", "content": f"Event {i}", "semantic_vector": _sem(),
             "emotional_vector": _emo(), "game_ts": float(i), "event_type": "info"}
            for i in range(3)
        ]
        ids = await writer.write_raw_batch(events)
        assert len(ids) == 3
        results = await get_points_by_ids(COLLECTION_NPC_MEMORIES, ids)
        assert len(results) == 3

    async def test_empty_batch_returns_empty(self, qdrant, writer):
        ids = await writer.write_raw_batch([])
        assert ids == []


# ---------------------------------------------------------------------------
# MOD tier
# ---------------------------------------------------------------------------

class TestWriteArcSummary:
    async def test_writes_mod_tier(self, qdrant, writer):
        pid = await writer.write_arc_summary(
            agent_id="Lydia", summary_text="Dragon encounter arc.",
            semantic_vector=_sem(), emotional_vector=_emo(),
            arc_start_ts=100.0, arc_end_ts=200.0,
            raw_point_ids=["raw-1", "raw-2"], game_ts=200.0,
        )
        results = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [pid])
        p = results[0]["payload"]
        assert p["tier"] == "MOD"
        assert p["arc_start_ts"] == 100.0
        assert p["arc_end_ts"] == 200.0
        assert p["raw_point_ids"] == ["raw-1", "raw-2"]


# ---------------------------------------------------------------------------
# MAX tier
# ---------------------------------------------------------------------------

class TestWriteCompressedEssence:
    async def test_writes_max_tier(self, qdrant, writer):
        pid = await writer.write_compressed_essence(
            agent_id="Lydia", essence_text="Veteran warrior essence.",
            semantic_vector=_sem(), emotional_vector=_emo(),
            source_arc_ids=["arc-1", "arc-2"], game_ts=500.0,
        )
        results = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [pid])
        p = results[0]["payload"]
        assert p["tier"] == "MAX"
        assert p["source_arc_ids"] == ["arc-1", "arc-2"]


# ---------------------------------------------------------------------------
# World events
# ---------------------------------------------------------------------------

class TestWriteWorldEvent:
    async def test_writes_to_world_events_collection(self, qdrant, writer):
        pid = await writer.write_world_event(
            event_type="location", content="Entered Whiterun",
            semantic_vector=_sem(), emotional_vector=_emo(),
            game_ts=100.0, location="Whiterun",
            involved_npcs=["Lydia", "Guard"],
        )
        results = await get_points_by_ids(COLLECTION_WORLD_EVENTS, [pid])
        assert len(results) == 1
        assert results[0]["payload"]["location"] == "Whiterun"


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class TestWriteAgentState:
    async def test_writes_to_agent_state_collection(self, qdrant, writer):
        pid = await writer.write_agent_state(
            agent_id="Lydia", emotional_vector=_emo(),
            harmonic_buffers={"fast": _emo(0.8), "medium": _emo(0.5), "slow": _emo(0.1)},
            curvature=0.3, snap=0.05, lambda_t=0.6, coherence=0.8,
        )
        results = await get_points_by_ids(COLLECTION_AGENT_STATE, [pid])
        assert len(results) == 1
        assert results[0]["payload"]["agent_id"] == "Lydia"
        assert results[0]["payload"]["curvature"] == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Session stash
# ---------------------------------------------------------------------------

class TestStashSessionContext:
    async def test_writes_to_session_context(self, qdrant, writer):
        pid = await writer.stash_session_context(
            agent_id="Lydia", context_text="We were discussing the war.",
            semantic_vector=_sem(), stash_reason="snap_threshold",
        )
        results = await get_points_by_ids(COLLECTION_SESSION_CONTEXT, [pid])
        assert len(results) == 1
        p = results[0]["payload"]
        assert p["agent_id"] == "Lydia"
        assert p["rehydrated"] is False
        assert p["stash_reason"] == "snap_threshold"


# ---------------------------------------------------------------------------
# Lore
# ---------------------------------------------------------------------------

class TestWriteLoreEntry:
    async def test_writes_to_lore_collection(self, qdrant, writer):
        pid = await writer.write_lore_entry(
            topic="Civil War", content="The conflict between...",
            semantic_vector=_sem(),
        )
        results = await get_points_by_ids(COLLECTION_LORE, [pid])
        assert len(results) == 1
        assert results[0]["payload"]["topic"] == "Civil War"
