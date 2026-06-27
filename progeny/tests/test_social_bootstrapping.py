"""Cross-arc integration tests for Social Bootstrapping (Phase 6g).

Exercises the real modules together (no LLM, no embedding model — in-memory
Qdrant + hand-made vectors) against the plan's headline scenarios:

  A. Two strangers meet, introduce themselves, and on the next meeting
     recognize each other without re-introducing (reciprocal loop closure).
  B. Valence drives the get-acquainted goal: a wary class-prior suppresses the
     approach below candidacy; a warm one promotes it.
  C. Eulogy: after a speaker "despawns", a listener who was told about them can
     still recall them from the durable store.
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

import progeny.src.qdrant_client as client_mod
from progeny.src import acquaintance, disclosure, identity_kernel, social_goals, valence
from progeny.src.fact_pool import FactPool
from progeny.src.identity_kernel import IdentityKernel
from progeny.src.memory_retrieval import MemoryRetriever
from progeny.src.memory_writer import MemoryWriter
from progeny.src.qdrant_client import ensure_collections
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


@pytest.fixture(autouse=True)
def _clean_state():
    acquaintance.clear()
    valence.reset()
    identity_kernel.clear()
    yield
    acquaintance.clear()
    valence.reset()
    identity_kernel.clear()


def _sem(val: float = 0.1) -> list[float]:
    v = [val] * SEMANTIC_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo(val: float = 0.2) -> list[float]:
    return [val] * EMOTIONAL_DIM


def _emo_warm(scale: float = 0.6) -> list[float]:
    v = [0.0] * EMOTIONAL_DIM
    v[2] = v[6] = v[7] = scale  # love, joy, safety
    return v


def _emo_wary(scale: float = 0.6) -> list[float]:
    v = [0.0] * EMOTIONAL_DIM
    v[0] = v[1] = v[3] = scale  # fear, anger, disgust
    return v


def _kernel(agent_id: str, occupation: str, origin: str) -> IdentityKernel:
    return IdentityKernel(
        agent_id=agent_id, slug=agent_id.lower(),
        public={"occupation": occupation, "origin": origin},
    )


async def _introduce(writer, pool, speaker, listener, kernel):
    await disclosure.propagate_introduction(
        writer=writer, fact_pool=pool, speaker=speaker, listener=listener,
        speaker_kernel=kernel, identity_semantic_vec=_sem(),
        listener_reaction=_emo(), speaker_reaction=_emo(), game_ts=1.0,
    )


# ---------------------------------------------------------------------------
# A. Strangers meet -> introduce -> recognize (no re-introduction)
# ---------------------------------------------------------------------------

class TestStrangersMeetAndRecognize:
    async def test_full_loop_closure(self, qdrant, writer, retriever):
        a, b = "Alvor", "Bjorn"
        pool = FactPool()

        # First meeting: nothing known -> both are strangers.
        assert acquaintance.is_stranger(pool, a, b, recognition_empty=True)
        assert acquaintance.is_stranger(pool, b, a, recognition_empty=True)
        acquaintance.record_stranger(a, b)
        acquaintance.record_stranger(b, a)

        # Both speak -> reciprocal introductions.
        await _introduce(writer, pool, a, b, _kernel(a, "smith", "Riverwood"))
        await _introduce(writer, pool, b, a, _kernel(b, "blacksmith", "Riften"))

        # Loop closed: no longer strangers, acquainted both ways.
        assert not acquaintance.is_known_stranger(a, b)
        assert not acquaintance.is_known_stranger(b, a)
        assert acquaintance.are_acquainted(pool, a, b)
        assert acquaintance.are_acquainted(pool, b, a)

        # Second meeting: each recalls the other (referent-keyed recognition).
        bundle_a = await retriever.retrieve_for_agent(
            agent_id=a, semantic_query=_sem(), emotional_query=_emo(),
            lambda_t=0.8, current_game_ts=2.0, referents=[b],
        )
        bundle_b = await retriever.retrieve_for_agent(
            agent_id=b, semantic_query=_sem(), emotional_query=_emo(),
            lambda_t=0.8, current_game_ts=2.0, referents=[a],
        )
        assert bundle_a.recent
        assert bundle_b.recent

    async def test_idempotent_reintroduction_does_not_duplicate(self, qdrant, writer):
        a, b = "Alvor", "Bjorn"
        pool = FactPool()
        for _ in range(3):
            await _introduce(writer, pool, a, b, _kernel(a, "smith", "Riverwood"))
        # Deterministic IDs -> one hearsay (under B) + one telling (under A).
        from qdrant_client.models import FieldCondition, Filter, MatchValue
        from progeny.src.qdrant_client import scroll_filtered
        from shared.constants import COLLECTION_NPC_MEMORIES

        b_points = await scroll_filtered(
            COLLECTION_NPC_MEMORIES,
            Filter(must=[FieldCondition(key="agent_id", match=MatchValue(value=b))]),
        )
        assert len(b_points) == 1


# ---------------------------------------------------------------------------
# B. Valence drives the get-acquainted goal
# ---------------------------------------------------------------------------

class TestValenceDrivesGetAcquainted:
    async def test_wary_class_prior_suppresses_approach(self, qdrant, writer):
        observer, soldier = "Mara", "Ralof"
        # A wary memory about a DIFFERENT soldier (class prior, not this one).
        await writer.write_raw_event(
            agent_id=observer, content="A soldier robbed me at the gate.",
            semantic_vector=_sem(), emotional_vector=_emo_wary(),
            game_ts=1.0, event_type="event", referents=["Hadvar"],
        )
        acquaintance.record_stranger(observer, soldier)

        reading = await valence.percept_cued_valence(observer, soldier, _sem(), _emo_wary())
        blended = valence.blend_valence(reading)
        valence.record_social(observer, soldier, blended.effective)

        assert blended.is_wary
        # Suppressed below the neutral baseline curiosity.
        assert social_goals.social_activation(observer, {soldier}) < social_goals.BASE_SOCIAL_ACTIVATION

    async def test_warm_class_prior_promotes_approach(self, qdrant, writer):
        observer, soldier = "Mara", "Ralof"
        await writer.write_raw_event(
            agent_id=observer, content="A soldier once shared his rations with me.",
            semantic_vector=_sem(), emotional_vector=_emo_warm(),
            game_ts=1.0, event_type="event", referents=["Hadvar"],
        )
        acquaintance.record_stranger(observer, soldier)

        reading = await valence.percept_cued_valence(observer, soldier, _sem(), _emo_warm())
        blended = valence.blend_valence(reading)
        valence.record_social(observer, soldier, blended.effective)

        assert blended.is_warm
        assert social_goals.social_activation(observer, {soldier}) > social_goals.BASE_SOCIAL_ACTIVATION


# ---------------------------------------------------------------------------
# C. Eulogy — hearsay survives the subject's despawn
# ---------------------------------------------------------------------------

class TestEulogy:
    async def test_listener_recalls_speaker_after_despawn(self, qdrant, writer, retriever):
        speaker, listener = "Ulfric", "Mara"
        await _introduce(writer, FactPool(), speaker, listener, _kernel(speaker, "jarl", "Windhelm"))

        # The speaker is gone from the scene, but the listener's durable hearsay
        # remains and is recoverable by referent.
        bundle = await retriever.retrieve_for_agent(
            agent_id=listener, semantic_query=_sem(), emotional_query=_emo(),
            lambda_t=0.8, current_game_ts=999.0, referents=[speaker],
        )
        assert bundle.recent
        assert any(speaker in entry.get("text", "") for entry in bundle.recent)
