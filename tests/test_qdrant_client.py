"""Tests for progeny.src.qdrant_client.

Uses AsyncQdrantClient(':memory:') injected via the `qdrant` fixture —
no live Qdrant instance required. All tests are async (asyncio_mode=auto).
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient

import progeny.src.qdrant_client as client_mod
from progeny.src.qdrant_client import (
    _agent_state_point_id,
    ensure_collections,
    read_agent_state,
    search_memories,
    write_agent_state,
    write_memory,
)
from shared.constants import (
    COLLECTION_AGENT_STATE,
    COLLECTION_NPC_MEMORIES,
    EMOTIONAL_DIM,
    SEMANTIC_DIM,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
async def qdrant():
    """In-memory Qdrant client with MMK collections already created.

    Injects into the module-level singleton so all client functions
    use this instance. Restores None on teardown.
    """
    mem = AsyncQdrantClient(location=":memory:")
    client_mod.configure(mem)
    await ensure_collections()
    yield mem
    client_mod.configure(None)


def _sem_vec(val: float = 0.1) -> list[float]:
    """Deterministic 384d unit-ish vector for test use."""
    v = [val] * SEMANTIC_DIM
    # Normalize so it's a valid cosine query target
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


def _emo_vec(val: float = 0.5) -> list[float]:
    """Deterministic 9d emotional vector for test use."""
    v = [val] * EMOTIONAL_DIM
    mag = sum(x * x for x in v) ** 0.5
    return [x / mag for x in v]


# ---------------------------------------------------------------------------
# configure / get_client
# ---------------------------------------------------------------------------

class TestLifecycle:
    def test_get_client_raises_without_init(self):
        saved = client_mod._client
        client_mod._client = None
        with pytest.raises(RuntimeError, match="not initialized"):
            client_mod.get_client()
        client_mod._client = saved

    async def test_configure_sets_client(self):
        mem = AsyncQdrantClient(location=":memory:")
        client_mod.configure(mem)
        assert client_mod.get_client() is mem
        client_mod.configure(None)

    async def test_health_check_ok(self, qdrant):
        assert await client_mod.health_check() is True

    async def test_health_check_fails_gracefully_when_unreachable(self):
        """Points at a port nothing listens on — should return False, not raise."""
        bad = AsyncQdrantClient(host="127.0.0.1", port=19999)
        client_mod.configure(bad)
        result = await client_mod.health_check()
        assert result is False
        client_mod.configure(None)


# ---------------------------------------------------------------------------
# ensure_collections
# ---------------------------------------------------------------------------

class TestEnsureCollections:
    async def test_creates_npc_memories(self, qdrant):
        names = {c.name for c in (await qdrant.get_collections()).collections}
        assert COLLECTION_NPC_MEMORIES in names

    async def test_creates_agent_state(self, qdrant):
        names = {c.name for c in (await qdrant.get_collections()).collections}
        assert COLLECTION_AGENT_STATE in names

    async def test_idempotent_on_second_call(self, qdrant):
        """Calling ensure_collections again should not raise or duplicate."""
        await ensure_collections()  # second call
        names = {c.name for c in (await qdrant.get_collections()).collections}
        assert COLLECTION_NPC_MEMORIES in names
        assert COLLECTION_AGENT_STATE in names

    async def test_does_not_touch_preexisting_collections(self, qdrant):
        """A pre-existing alien collection should survive untouched."""
        await qdrant.create_collection(
            "garden_archive",
            vectors_config={"semantic": __import__("qdrant_client.models", fromlist=["VectorParams"]).VectorParams(
                size=384,
                distance=__import__("qdrant_client.models", fromlist=["Distance"]).Distance.COSINE,
            )},
        )
        await ensure_collections()
        names = {c.name for c in (await qdrant.get_collections()).collections}
        assert "garden_archive" in names  # still there, unmolested


# ---------------------------------------------------------------------------
# write_memory
# ---------------------------------------------------------------------------

class TestWriteMemory:
    async def test_returns_uuid_string(self, qdrant):
        point_id = await write_memory(
            agent_id="Lydia",
            text="She drew her sword.",
            semantic_vec=_sem_vec(),
            emotional_vec=_emo_vec(),
            game_ts=100.0,
        )
        import uuid
        uuid.UUID(point_id)  # raises if not valid UUID

    async def test_point_retrievable(self, qdrant):
        point_id = await write_memory(
            agent_id="Lydia",
            text="She drew her sword.",
            semantic_vec=_sem_vec(),
            emotional_vec=_emo_vec(),
            game_ts=100.0,
        )
        points, _ = await qdrant.scroll(
            COLLECTION_NPC_MEMORIES, limit=10, with_payload=True,
        )
        ids = [p.id for p in points]
        assert point_id in ids

    async def test_payload_fields_written(self, qdrant):
        point_id = await write_memory(
            agent_id="Lydia",
            text="I am sworn to carry your burdens.",
            semantic_vec=_sem_vec(),
            emotional_vec=_emo_vec(),
            game_ts=54321.0,
            tier="RAW",
        )
        points, _ = await qdrant.scroll(
            COLLECTION_NPC_MEMORIES,
            limit=1,
            with_payload=True,
        )
        p = next(pt for pt in points if pt.id == point_id)
        assert p.payload["agent_id"] == "Lydia"
        assert p.payload["tier"] == "RAW"
        assert p.payload["text"] == "I am sworn to carry your burdens."
        assert p.payload["game_ts"] == pytest.approx(54321.0)

    async def test_arc_id_stored_in_payload(self, qdrant):
        arc_id = "arc-001"
        point_id = await write_memory(
            agent_id="Lydia",
            text="Dragon encounter arc.",
            semantic_vec=_sem_vec(),
            emotional_vec=_emo_vec(),
            game_ts=200.0,
            tier="MOD",
            arc_id=arc_id,
        )
        points, _ = await qdrant.scroll(COLLECTION_NPC_MEMORIES, limit=10, with_payload=True)
        p = next(pt for pt in points if pt.id == point_id)
        assert p.payload["arc_id"] == arc_id

    async def test_extra_payload_merged(self, qdrant):
        point_id = await write_memory(
            agent_id="Lydia",
            text="In the heat of battle.",
            semantic_vec=_sem_vec(),
            emotional_vec=_emo_vec(),
            game_ts=300.0,
            extra_payload={"location": "Western Watchtower", "combat": True},
        )
        points, _ = await qdrant.scroll(COLLECTION_NPC_MEMORIES, limit=10, with_payload=True)
        p = next(pt for pt in points if pt.id == point_id)
        assert p.payload["location"] == "Western Watchtower"
        assert p.payload["combat"] is True

    async def test_multiple_agents_stored_independently(self, qdrant):
        id1 = await write_memory("Lydia", "Lydia speaks.", _sem_vec(0.1), _emo_vec(0.3), 100.0)
        id2 = await write_memory("Belethor", "Belethor speaks.", _sem_vec(0.2), _emo_vec(0.6), 101.0)
        assert id1 != id2

        points, _ = await qdrant.scroll(COLLECTION_NPC_MEMORIES, limit=10, with_payload=True)
        agent_ids = {p.payload["agent_id"] for p in points}
        assert "Lydia" in agent_ids
        assert "Belethor" in agent_ids


# ---------------------------------------------------------------------------
# write_agent_state / read_agent_state
# ---------------------------------------------------------------------------

_FAST = _emo_vec(0.8)
_MED  = _emo_vec(0.5)
_SLOW = _emo_vec(0.1)


class TestAgentState:
    async def test_write_then_read_roundtrip(self, qdrant):
        await write_agent_state("Lydia", _FAST, _MED, _SLOW, 0.42, True)
        state = await read_agent_state("Lydia")
        assert state is not None
        assert state["agent_id"] == "Lydia"
        assert state["harmonic_fast"] == pytest.approx(_FAST, abs=1e-5)
        assert state["harmonic_medium"] == pytest.approx(_MED, abs=1e-5)
        assert state["harmonic_slow"] == pytest.approx(_SLOW, abs=1e-5)
        assert state["prev_curvature"] == pytest.approx(0.42)
        assert state["initialized"] is True

    async def test_read_returns_none_for_unknown_agent(self, qdrant):
        result = await read_agent_state("Nobody")
        assert result is None

    async def test_idempotent_upsert_overwrites_previous(self, qdrant):
        """Second write to same agent updates the payload, not creates a duplicate."""
        await write_agent_state("Lydia", _FAST, _MED, _SLOW, 0.1, True)
        new_fast = _emo_vec(0.9)
        await write_agent_state("Lydia", new_fast, _MED, _SLOW, 0.5, True)

        state = await read_agent_state("Lydia")
        assert state["harmonic_fast"] == pytest.approx(new_fast, abs=1e-5)
        assert state["prev_curvature"] == pytest.approx(0.5)

        # Still only one point for Lydia
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        pts, _ = await qdrant.scroll(
            COLLECTION_AGENT_STATE,
            scroll_filter=Filter(must=[FieldCondition(key="agent_id", match=MatchValue(value="Lydia"))]),
            limit=10,
        )
        assert len(pts) == 1

    async def test_deterministic_point_id_stable(self):
        """Same agent always maps to the same point ID."""
        id1 = _agent_state_point_id("Lydia")
        id2 = _agent_state_point_id("Lydia")
        assert id1 == id2

    async def test_different_agents_different_ids(self):
        assert _agent_state_point_id("Lydia") != _agent_state_point_id("Belethor")

    async def test_multiple_agents_stored_independently(self, qdrant):
        # Use a non-uniform vector for Belethor so the values are distinct
        # after normalization (uniform vectors always normalize to the same result)
        belethor_fast = [float(i) / 10 for i in range(1, EMOTIONAL_DIM + 1)]
        await write_agent_state("Lydia", _FAST, _MED, _SLOW, 0.1, True)
        await write_agent_state("Belethor", belethor_fast, _MED, _SLOW, 0.3, True)

        lydia = await read_agent_state("Lydia")
        belethor = await read_agent_state("Belethor")
        assert lydia is not None
        assert belethor is not None
        assert lydia["harmonic_fast"] != belethor["harmonic_fast"]


# ---------------------------------------------------------------------------
# search_memories
# ---------------------------------------------------------------------------

class TestSearchMemories:
    async def _populate(self, qdrant, n: int = 5, agent: str = "Lydia") -> list[str]:
        """Write n test memories for the given agent. Returns point IDs."""
        ids = []
        for i in range(n):
            pid = await write_memory(
                agent_id=agent,
                text=f"Memory {i} for {agent}.",
                semantic_vec=_sem_vec(0.1 + i * 0.01),
                emotional_vec=_emo_vec(0.1 + i * 0.05),
                game_ts=float(i * 100),
            )
            ids.append(pid)
        return ids

    async def test_returns_results(self, qdrant):
        await self._populate(qdrant)
        results = await search_memories(_emo_vec(), _sem_vec())
        assert len(results) > 0

    async def test_results_have_payload(self, qdrant):
        await self._populate(qdrant)
        results = await search_memories(_emo_vec(), _sem_vec())
        for r in results:
            assert r.payload is not None
            assert "agent_id" in r.payload

    async def test_agent_filter_restricts_results(self, qdrant):
        await self._populate(qdrant, agent="Lydia")
        await self._populate(qdrant, agent="Belethor")
        results = await search_memories(_emo_vec(), _sem_vec(), agent_id="Lydia")
        assert all(r.payload["agent_id"] == "Lydia" for r in results)

    async def test_limit_respected(self, qdrant):
        await self._populate(qdrant, n=10)
        results = await search_memories(_emo_vec(), _sem_vec(), limit=3)
        assert len(results) <= 3

    async def test_empty_collection_returns_empty_list(self, qdrant):
        results = await search_memories(_emo_vec(), _sem_vec())
        assert results == []

    async def test_returns_empty_on_qdrant_error(self, monkeypatch):
        """Qdrant failure returns [] without raising."""
        async def _boom(*a, **kw):
            raise RuntimeError("connection lost")
        monkeypatch.setattr(client_mod, "_client", object())  # break get_client flow
        # Directly patch query_points on a real client that's broken
        mem = AsyncQdrantClient(location=":memory:")
        client_mod.configure(mem)
        await ensure_collections()
        monkeypatch.setattr(mem, "query_points", _boom)
        result = await search_memories(_emo_vec(), _sem_vec())
        assert result == []
        client_mod.configure(None)
