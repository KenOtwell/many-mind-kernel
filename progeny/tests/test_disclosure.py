"""Tests for disclosure -> hearsay propagation (Phase 6e).

Pure tests for the content builders and deterministic IDs, plus integration
tests (in-memory Qdrant, hand-made vectors, no model) for the reciprocal
propagation: the listener's hearsay memory, the speaker's telling memory, the
symbolic identity fact, and the both-ways stranger-ledger clear.
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue

import progeny.src.qdrant_client as client_mod
from progeny.src import acquaintance
from progeny.src import disclosure
from progeny.src.fact_pool import FactPool
from progeny.src.identity_kernel import IdentityKernel
from progeny.src.memory_writer import MemoryWriter
from progeny.src.qdrant_client import ensure_collections, get_points_by_ids, scroll_filtered
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


@pytest.fixture(autouse=True)
def _clean_ledger():
    acquaintance.clear()
    yield
    acquaintance.clear()


def _sem(val: float = 0.1) -> list[float]:
    v = [val] * SEMANTIC_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo(val: float = 0.2) -> list[float]:
    return [val] * EMOTIONAL_DIM


def _kernel(agent_id: str = "Bjorn", occupation: str = "blacksmith", origin: str = "Riften") -> IdentityKernel:
    return IdentityKernel(
        agent_id=agent_id, slug=agent_id.lower(),
        public={"occupation": occupation, "origin": origin},
    )


async def _agent_point_count(agent_id: str) -> int:
    points = await scroll_filtered(
        COLLECTION_NPC_MEMORIES,
        Filter(must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]),
    )
    return len(points)


# ---------------------------------------------------------------------------
# Pure: content + IDs
# ---------------------------------------------------------------------------

class TestContentAndIds:
    def test_descriptor_from_kernel(self):
        assert disclosure.identity_descriptor(_kernel()) == "a blacksmith, from Riften"

    def test_descriptor_empty_without_profile(self):
        assert disclosure.identity_descriptor(IdentityKernel(agent_id="X", slug="x")) == ""

    def test_content_builders(self):
        assert disclosure.hearsay_content("Bjorn", "a blacksmith") == "Bjorn introduced themselves — a blacksmith."
        assert disclosure.hearsay_content("Bjorn", "") == "Bjorn introduced themselves."
        assert "Mara" in disclosure.telling_content("Mara")
        assert disclosure.fact_content("Bjorn", "a blacksmith") == "Bjorn is a blacksmith."

    def test_ids_are_deterministic_and_distinct(self):
        assert disclosure.hearsay_id("Mara", "Bjorn") == disclosure.hearsay_id("Mara", "Bjorn")
        # Direction matters and the three id families do not collide.
        assert disclosure.hearsay_id("Mara", "Bjorn") != disclosure.hearsay_id("Bjorn", "Mara")
        assert disclosure.telling_id("Bjorn", "Mara") != disclosure.hearsay_id("Mara", "Bjorn")
        assert disclosure.identity_fact_id("Bjorn", "Mara") != disclosure.telling_id("Bjorn", "Mara")


# ---------------------------------------------------------------------------
# Integration: reciprocal propagation
# ---------------------------------------------------------------------------

class TestPropagateIntroduction:
    async def test_both_parties_get_a_memory(self, qdrant, writer):
        speaker, listener = "Bjorn", "Mara"
        await disclosure.propagate_introduction(
            writer=writer, fact_pool=None, speaker=speaker, listener=listener,
            speaker_kernel=_kernel(speaker), identity_semantic_vec=_sem(),
            listener_reaction=_emo(0.3), speaker_reaction=_emo(0.1), game_ts=1.0,
        )
        # Listener holds a hearsay memory about the speaker.
        hearsay = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [disclosure.hearsay_id(listener, speaker)])
        assert hearsay and hearsay[0]["payload"]["referents"] == [speaker]
        assert speaker in hearsay[0]["payload"]["content"]
        # Speaker holds the reciprocal telling memory about the listener.
        telling = await get_points_by_ids(COLLECTION_NPC_MEMORIES, [disclosure.telling_id(speaker, listener)])
        assert telling and telling[0]["payload"]["referents"] == [listener]
        assert listener in telling[0]["payload"]["content"]

    async def test_identity_fact_makes_listener_acquainted(self, qdrant, writer):
        speaker, listener = "Bjorn", "Mara"
        pool = FactPool()
        await disclosure.propagate_introduction(
            writer=writer, fact_pool=pool, speaker=speaker, listener=listener,
            speaker_kernel=_kernel(speaker), identity_semantic_vec=_sem(),
            listener_reaction=_emo(), speaker_reaction=_emo(), game_ts=1.0,
        )
        assert acquaintance.are_acquainted(pool, listener, speaker) is True

    async def test_clears_stranger_ledger_both_ways(self, qdrant, writer):
        speaker, listener = "Bjorn", "Mara"
        acquaintance.record_stranger(listener, speaker)
        acquaintance.record_stranger(speaker, listener)
        await disclosure.propagate_introduction(
            writer=writer, fact_pool=None, speaker=speaker, listener=listener,
            speaker_kernel=_kernel(speaker), identity_semantic_vec=_sem(),
            listener_reaction=_emo(), speaker_reaction=_emo(), game_ts=1.0,
        )
        assert acquaintance.is_known_stranger(listener, speaker) is False
        assert acquaintance.is_known_stranger(speaker, listener) is False

    async def test_retelling_is_idempotent(self, qdrant, writer):
        speaker, listener = "Bjorn", "Mara"
        for _ in range(3):
            await disclosure.propagate_introduction(
                writer=writer, fact_pool=None, speaker=speaker, listener=listener,
                speaker_kernel=_kernel(speaker), identity_semantic_vec=_sem(),
                listener_reaction=_emo(), speaker_reaction=_emo(), game_ts=1.0,
            )
        # Deterministic IDs -> one point per party, reinforced not duplicated.
        assert await _agent_point_count(listener) == 1
        assert await _agent_point_count(speaker) == 1
