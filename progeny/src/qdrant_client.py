"""
Qdrant async client for Progeny — dual-vector memory store.

Thin async wrapper around qdrant-client. Progeny is the single write
authority for all memory tiers (RAW/MOD/MAX) and agent state. Falcon
never touches Qdrant.

Connection target: Gaming PC Qdrant at config.qdrant.host:rest_port.
Both services share the same instance; Progeny owns all writes.

Named vector schema for skyrim_npc_memories:
  "semantic"  — 384d Cosine  (all-MiniLM content embedding)
  "emotional" — 9d Cosine    (harmonic basis projection)

Dual-vector retrieval: prefetch both axes independently, fuse with RRF.
λ-weighting and re-ranking live in memory_retrieval.py, not here —
search_memories() returns raw fused candidates.

Error policy: all operations catch Exception, log, and return a safe
default. Qdrant unreachability must never crash the turn cycle.
"""
from __future__ import annotations

import logging
import time
from typing import Any
from uuid import uuid4, uuid5, NAMESPACE_DNS

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    PointStruct,
    Prefetch,
    ScoredPoint,
    VectorParams,
)

from shared.config import settings
from shared.constants import (
    COLLECTION_AGENT_STATE,
    COLLECTION_LORE,
    COLLECTION_NPC_MEMORIES,
    COLLECTION_SESSION_CONTEXT,
    COLLECTION_WORLD_EVENTS,
    EMOTIONAL_DIM,
    SEMANTIC_DIM,
)

logger = logging.getLogger(__name__)

# Module-level singleton — initialized by init() at startup
_client: AsyncQdrantClient | None = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

def configure(client: AsyncQdrantClient) -> None:
    """Inject a client instance (used in tests to supply an in-memory client)."""
    global _client
    _client = client


def get_client() -> AsyncQdrantClient:
    """Return the module-level client, raising if not yet initialized."""
    if _client is None:
        raise RuntimeError(
            "Qdrant client not initialized — call client.init() first "
            "(or client.configure() in tests)"
        )
    return _client


async def init(host: str | None = None, port: int | None = None) -> None:
    """Initialize the async Qdrant client. Called from server.py lifespan.

    Args:
        host: Override for settings.qdrant.host (LAN IP of Gaming PC).
        port: Override for settings.qdrant.rest_port.
    """
    global _client
    _host = host or settings.qdrant.host
    _port = port or settings.qdrant.rest_port
    _client = AsyncQdrantClient(host=_host, port=_port)
    logger.info("Qdrant client initialized → %s:%d", _host, _port)


async def health_check() -> bool:
    """Check if Qdrant is reachable."""
    try:
        await get_client().get_collections()
        return True
    except Exception as exc:
        logger.warning("Qdrant health check failed: %s", exc)
        return False


async def ensure_collections() -> None:
    """Create required MMK collections if they don't already exist.

    Safe to call on startup even when the Gaming PC Qdrant already holds
    data from prior runs (garden_archive, trinity_archive, etc.). This
    function NEVER modifies or migrates existing collections.

    Collections created:
      skyrim_npc_memories — dual-vector (semantic 384d + emotional 9d)
      skyrim_agent_state  — single emotional vector (slow buffer = personality)
    """
    client = get_client()
    try:
        existing = {c.name for c in (await client.get_collections()).collections}
    except Exception as exc:
        logger.error("Qdrant ensure_collections: cannot list collections: %s", exc)
        return

    if COLLECTION_NPC_MEMORIES not in existing:
        try:
            await client.create_collection(
                collection_name=COLLECTION_NPC_MEMORIES,
                vectors_config={
                    "semantic": VectorParams(
                        size=SEMANTIC_DIM, distance=Distance.COSINE,
                    ),
                    "emotional": VectorParams(
                        size=EMOTIONAL_DIM, distance=Distance.COSINE,
                    ),
                },
            )
            logger.info("Created collection: %s", COLLECTION_NPC_MEMORIES)
        except Exception as exc:
            logger.error("Failed to create %s: %s", COLLECTION_NPC_MEMORIES, exc)

    if COLLECTION_AGENT_STATE not in existing:
        try:
            await client.create_collection(
                collection_name=COLLECTION_AGENT_STATE,
                vectors_config={
                    # slow buffer = personality substrate; enables future
                    # "find agents with similar deep personality" searches.
                    "emotional": VectorParams(
                        size=EMOTIONAL_DIM, distance=Distance.COSINE,
                    ),
                },
            )
            logger.info("Created collection: %s", COLLECTION_AGENT_STATE)
        except Exception as exc:
            logger.error("Failed to create %s: %s", COLLECTION_AGENT_STATE, exc)

    if COLLECTION_WORLD_EVENTS not in existing:
        try:
            await client.create_collection(
                collection_name=COLLECTION_WORLD_EVENTS,
                vectors_config={
                    "semantic": VectorParams(
                        size=SEMANTIC_DIM, distance=Distance.COSINE,
                    ),
                    "emotional": VectorParams(
                        size=EMOTIONAL_DIM, distance=Distance.COSINE,
                    ),
                },
            )
            logger.info("Created collection: %s", COLLECTION_WORLD_EVENTS)
        except Exception as exc:
            logger.error("Failed to create %s: %s", COLLECTION_WORLD_EVENTS, exc)

    if COLLECTION_SESSION_CONTEXT not in existing:
        try:
            await client.create_collection(
                collection_name=COLLECTION_SESSION_CONTEXT,
                vectors_config={
                    "semantic": VectorParams(
                        size=SEMANTIC_DIM, distance=Distance.COSINE,
                    ),
                },
            )
            logger.info("Created collection: %s", COLLECTION_SESSION_CONTEXT)
        except Exception as exc:
            logger.error("Failed to create %s: %s", COLLECTION_SESSION_CONTEXT, exc)

    if COLLECTION_LORE not in existing:
        try:
            await client.create_collection(
                collection_name=COLLECTION_LORE,
                vectors_config={
                    "semantic": VectorParams(
                        size=SEMANTIC_DIM, distance=Distance.COSINE,
                    ),
                },
            )
            logger.info("Created collection: %s", COLLECTION_LORE)
        except Exception as exc:
            logger.error("Failed to create %s: %s", COLLECTION_LORE, exc)


# ---------------------------------------------------------------------------
# Memory writes — RAW and MOD/MAX tiers
# ---------------------------------------------------------------------------

async def write_memory(
    agent_id: str,
    text: str,
    semantic_vec: list[float],
    emotional_vec: list[float],
    game_ts: float,
    tier: str = "RAW",
    arc_id: str | None = None,
    extra_payload: dict[str, Any] | None = None,
) -> str:
    """Write one memory point to skyrim_npc_memories.

    Used for every tier:
      RAW — every significant event; immutable, never deleted.
      MOD — arc summaries; generated on snap threshold crossings.
      MAX — compressed essence; generated by compression.py.

    Args:
        agent_id:      The agent this memory belongs to.
        text:          Original text content (event text or arc summary).
        semantic_vec:  Pre-computed 384d content embedding.
        emotional_vec: Pre-computed 9d semagram at encoding time.
        game_ts:       Skyrim internal game timestamp.
        tier:          "RAW", "MOD", or "MAX".
        arc_id:        For MOD/MAX: parent arc identifier.
        extra_payload: Additional fields merged into the payload.

    Returns:
        UUID of the written point (for arc linking, bundle expansion).
    """
    point_id = str(uuid4())
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "tier": tier,
        "text": text,
        "game_ts": game_ts,
        "wall_ts": time.time(),
    }
    if arc_id is not None:
        payload["arc_id"] = arc_id
    if extra_payload:
        payload.update(extra_payload)

    try:
        await get_client().upsert(
            collection_name=COLLECTION_NPC_MEMORIES,
            points=[
                PointStruct(
                    id=point_id,
                    vector={"semantic": semantic_vec, "emotional": emotional_vec},
                    payload=payload,
                )
            ],
        )
    except Exception as exc:
        logger.error(
            "write_memory failed — agent=%s tier=%s: %s", agent_id, tier, exc,
        )

    return point_id


# ---------------------------------------------------------------------------
# Agent state persistence
# ---------------------------------------------------------------------------

def _agent_state_point_id(agent_id: str) -> str:
    """Deterministic UUID for an agent's state point.

    Always the same UUID for the same agent_id — enables idempotent
    upserts without a prior read. One point per agent in the collection.
    """
    return str(uuid5(NAMESPACE_DNS, f"mmk:agent_state:{agent_id}"))


async def write_agent_state(
    agent_id: str,
    fast: list[float],
    medium: list[float],
    slow: list[float],
    prev_curvature: float,
    initialized: bool,
) -> None:
    """Persist per-agent harmonic state to skyrim_agent_state.

    Idempotent: always upserts the same deterministic point ID for a
    given agent. The slow buffer is stored as the named vector — it IS
    the personality substrate, and makes future similarity searches
    (find agents with similar deep personality) possible.

    Args:
        agent_id:       NPC name/ID.
        fast:           9d fast EMA buffer.
        medium:         9d medium EMA buffer.
        slow:           9d slow EMA buffer (stored as vector).
        prev_curvature: Curvature from last update.
        initialized:    Whether the buffer has received any updates.
    """
    point_id = _agent_state_point_id(agent_id)
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "harmonic_fast": fast,
        "harmonic_medium": medium,
        "harmonic_slow": slow,
        "prev_curvature": prev_curvature,
        "initialized": initialized,
        "wall_ts": time.time(),
    }
    try:
        await get_client().upsert(
            collection_name=COLLECTION_AGENT_STATE,
            points=[
                PointStruct(
                    id=point_id,
                    vector={"emotional": slow},
                    payload=payload,
                )
            ],
        )
    except Exception as exc:
        logger.error("write_agent_state failed — agent=%s: %s", agent_id, exc)


async def read_agent_state(agent_id: str) -> dict[str, Any] | None:
    """Read persisted harmonic state for one agent.

    Returns the payload dict on success, None if no state exists (new
    agent, post-wipe, or Qdrant unreachable). The caller reconstructs
    HarmonicBuffer fields from the payload keys:
      harmonic_fast, harmonic_medium, harmonic_slow, prev_curvature,
      initialized.

    Used on Progeny startup to recover live state without cold buffers.
    """
    try:
        points, _ = await get_client().scroll(
            collection_name=COLLECTION_AGENT_STATE,
            scroll_filter=Filter(
                must=[FieldCondition(
                    key="agent_id", match=MatchValue(value=agent_id),
                )]
            ),
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return points[0].payload if points else None
    except Exception as exc:
        logger.error("read_agent_state failed — agent=%s: %s", agent_id, exc)
        return None


# ---------------------------------------------------------------------------
# Dual-vector search
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Generic helpers — used by memory_writer, compression, retrieval, rehydration
# ---------------------------------------------------------------------------

async def get_points_by_ids(
    collection: str,
    point_ids: list[str],
    with_vectors: bool = False,
) -> list[dict[str, Any]]:
    """Retrieve points by their IDs.

    Returns list of dicts with 'id', 'payload', and optionally 'vector'.
    Empty list if Qdrant is unreachable.
    """
    try:
        results = await get_client().retrieve(
            collection_name=collection,
            ids=point_ids,
            with_vectors=with_vectors,
            with_payload=True,
        )
        return [
            {
                "id": str(r.id),
                "payload": r.payload,
                "vector": r.vector if with_vectors else None,
            }
            for r in results
        ]
    except Exception as exc:
        logger.error("get_points_by_ids failed — collection=%s: %s", collection, exc)
        return []


async def scroll_filtered(
    collection: str,
    scroll_filter: Filter,
    limit: int = 100,
    with_vectors: bool = False,
    order_by: str | None = None,
) -> list[dict[str, Any]]:
    """Scroll through points matching a filter.

    Returns list of dicts with 'id', 'payload', and optionally 'vector',
    sorted by order_by payload field if specified.
    """
    try:
        records, _ = await get_client().scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
            limit=limit,
            with_vectors=with_vectors,
            with_payload=True,
        )
        results = [
            {
                "id": str(r.id),
                "payload": r.payload,
                "vector": r.vector if with_vectors else None,
            }
            for r in records
        ]
        if order_by and results and order_by in (results[0].get("payload") or {}):
            results.sort(key=lambda r: r["payload"].get(order_by, 0))
        return results
    except Exception as exc:
        logger.error("scroll_filtered failed — collection=%s: %s", collection, exc)
        return []


async def search_vector(
    collection: str,
    vector_name: str,
    query: list[float],
    limit: int = 10,
    query_filter: Filter | None = None,
    score_threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Search on a single named vector axis.

    Returns list of dicts with 'id', 'score', 'payload'.
    Used by memory_retrieval for separate emotional/semantic passes
    before client-side λ-blending.
    """
    try:
        result = await get_client().query_points(
            collection_name=collection,
            query=query,
            using=vector_name,
            limit=limit,
            query_filter=query_filter,
            score_threshold=score_threshold,
            with_payload=True,
        )
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in result.points
        ]
    except Exception as exc:
        logger.error(
            "search_vector failed — collection=%s vector=%s: %s",
            collection, vector_name, exc,
        )
        return []


async def set_point_payload(
    collection: str,
    point_ids: list[str],
    payload: dict[str, Any],
) -> None:
    """Update payload fields on existing points."""
    try:
        await get_client().set_payload(
            collection_name=collection,
            payload=payload,
            points=point_ids,
        )
    except Exception as exc:
        logger.error(
            "set_point_payload failed — collection=%s: %s", collection, exc,
        )


async def search_memories(
    emotional_vec: list[float],
    semantic_vec: list[float],
    agent_id: str | None = None,
    limit: int = 30,
) -> list[ScoredPoint]:
    """Dual-vector search on skyrim_npc_memories using RRF fusion.

    Prefetches candidates on both axes independently (emotional and
    semantic), then fuses scores with Reciprocal Rank Fusion. Returns
    the top `limit` fused candidates ordered by RRF score.

    λ-weighting and re-ranking (referent filtering, recency decay,
    anchor boosting) are handled by memory_retrieval.py — this function
    returns the raw fused set for downstream processing.

    Args:
        emotional_vec: 9d query vector — current harmonic state.
        semantic_vec:  384d query vector — current content embedding.
        agent_id:      Optional: restrict search to one agent's memories.
        limit:         Number of fused candidates to return.

    Returns:
        List of ScoredPoints ordered by RRF score, highest first.
        Empty list if Qdrant is unreachable.
    """
    payload_filter: Filter | None = None
    if agent_id is not None:
        payload_filter = Filter(
            must=[FieldCondition(
                key="agent_id", match=MatchValue(value=agent_id),
            )]
        )

    try:
        result = await get_client().query_points(
            collection_name=COLLECTION_NPC_MEMORIES,
            prefetch=[
                Prefetch(
                    query=emotional_vec,
                    using="emotional",
                    limit=limit * 2,
                    filter=payload_filter,
                ),
                Prefetch(
                    query=semantic_vec,
                    using="semantic",
                    limit=limit * 2,
                    filter=payload_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
            with_payload=True,
        )
        return result.points
    except Exception as exc:
        logger.error("search_memories failed: %s", exc)
        return []
