"""Tests for progeny.src.memory_retrieval (async function-based API).

Populates in-memory Qdrant with test data via MemoryWriter, then exercises
the MemoryRetriever pipeline.
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

import progeny.src.qdrant_client as client_mod
from progeny.src.qdrant_client import ensure_collections
from progeny.src.memory_writer import MemoryWriter
from progeny.src.memory_retrieval import MemoryRetriever, MemoryBundle
from shared.constants import EMOTIONAL_DIM, SEMANTIC_DIM


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
def retriever():
    return MemoryRetriever()


def _sem(val: float = 0.1) -> list[float]:
    v = [val] * SEMANTIC_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo(val: float = 0.5) -> list[float]:
    v = [val] * EMOTIONAL_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


async def _populate(writer: MemoryWriter, agent: str = "Lydia", n: int = 5) -> list[str]:
    """Write n test RAW memories. Returns point IDs."""
    ids = []
    for i in range(n):
        pid = await writer.write_raw_event(
            agent_id=agent, content=f"Memory {i} for {agent}.",
            semantic_vector=_sem(0.1 + i * 0.01),
            emotional_vector=_emo(0.1 + i * 0.05),
            game_ts=float(i * 100), event_type="info",
        )
        ids.append(pid)
    return ids


class TestRetrieveForAgent:
    async def test_returns_memory_bundle(self, qdrant, writer, retriever):
        await _populate(writer)
        bundle = await retriever.retrieve_for_agent(
            agent_id="Lydia",
            semantic_query=_sem(),
            emotional_query=_emo(),
            lambda_t=0.5,
            current_game_ts=500.0,
        )
        assert isinstance(bundle, MemoryBundle)
        assert bundle.agent_id == "Lydia"

    async def test_returns_recent_entries(self, qdrant, writer, retriever):
        await _populate(writer, n=5)
        bundle = await retriever.retrieve_for_agent(
            agent_id="Lydia",
            semantic_query=_sem(),
            emotional_query=_emo(),
            lambda_t=0.5,
            current_game_ts=500.0,
        )
        # Should have some recent RAW entries
        assert len(bundle.recent) > 0

    async def test_empty_collection_returns_empty_bundle(self, qdrant, retriever):
        bundle = await retriever.retrieve_for_agent(
            agent_id="Lydia",
            semantic_query=_sem(),
            emotional_query=_emo(),
            lambda_t=0.5,
            current_game_ts=100.0,
        )
        assert bundle.recent == []
        assert bundle.summaries == []

    async def test_lambda_high_weights_emotional(self, qdrant, writer, retriever):
        """High λ should still return results (emotion-first)."""
        await _populate(writer)
        bundle = await retriever.retrieve_for_agent(
            agent_id="Lydia",
            semantic_query=_sem(),
            emotional_query=_emo(),
            lambda_t=0.9,
            current_game_ts=500.0,
        )
        assert len(bundle.recent) > 0

    async def test_lambda_low_weights_semantic(self, qdrant, writer, retriever):
        """Low λ should still return results (domain-first)."""
        await _populate(writer)
        bundle = await retriever.retrieve_for_agent(
            agent_id="Lydia",
            semantic_query=_sem(),
            emotional_query=_emo(),
            lambda_t=0.1,
            current_game_ts=500.0,
        )
        assert len(bundle.recent) > 0


class TestRetrieveLore:
    async def test_returns_results_when_populated(self, qdrant, writer, retriever):
        await writer.write_lore_entry(
            topic="Civil War", content="The conflict between the Stormcloaks and the Empire.",
            semantic_vector=_sem(),
        )
        results = await retriever.retrieve_lore(semantic_query=_sem(), limit=3)
        assert len(results) > 0

    async def test_empty_collection_returns_empty(self, qdrant, retriever):
        results = await retriever.retrieve_lore(semantic_query=_sem())
        assert results == []
