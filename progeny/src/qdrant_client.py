"""
Qdrant REST client for the Many-Mind Kernel.

Handles connection to the Qdrant instance on the Gaming PC (Falcon/StealthVI)
over LAN. Provides dual-vector upsert and search for skyrim_npc_memories and
skyrim_world_events collections (semantic 384d + emotional 9d named vectors).

All Qdrant writes go through Progeny — single write authority.
"""
from __future__ import annotations

import logging
from typing import Any, Optional
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    NamedVector,
    PointStruct,
    Range,
    SearchParams,
    VectorParams,
    VectorsConfig,
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

# Collections that use dual named vectors (semantic + emotional)
DUAL_VECTOR_COLLECTIONS = frozenset({
    COLLECTION_NPC_MEMORIES,
    COLLECTION_WORLD_EVENTS,
})

# Collections with a single vector
SINGLE_VECTOR_COLLECTIONS = {
    COLLECTION_SESSION_CONTEXT: SEMANTIC_DIM,
    COLLECTION_AGENT_STATE: EMOTIONAL_DIM,
    COLLECTION_LORE: SEMANTIC_DIM,
}


class MMKQdrantClient:
    """
    Qdrant client for the Many-Mind Kernel.

    Connects to the shared Qdrant instance and provides typed operations
    for all MMK collections. Manages connection lifecycle and health checks.

    Usage:
        client = MMKQdrantClient()
        client.connect()
        client.upsert_dual_vector(COLLECTION_NPC_MEMORIES, point_id, semantic, emotional, payload)
        client.close()
    """

    def __init__(
        self,
        host: Optional[str] = None,
        port: Optional[int] = None,
        grpc_port: Optional[int] = None,
    ):
        cfg = settings.qdrant
        self._host = host or cfg.host
        self._port = port or cfg.rest_port
        self._grpc_port = grpc_port or cfg.grpc_port
        self._client: Optional[QdrantClient] = None

    # ------------------------------------------------------------------
    # Connection lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Establish connection to Qdrant. Idempotent."""
        if self._client is not None:
            return
        self._client = QdrantClient(
            host=self._host,
            port=self._port,
            grpc_port=self._grpc_port,
            prefer_grpc=False,  # REST is fine for our write volume
            timeout=10.0,
            check_compatibility=False,  # Client 1.15 vs server 1.12 — compatible
        )
        logger.info("Connected to Qdrant at %s:%d", self._host, self._port)

    def close(self) -> None:
        """Close the Qdrant connection."""
        if self._client is not None:
            self._client.close()
            self._client = None
            logger.info("Qdrant connection closed.")

    @property
    def client(self) -> QdrantClient:
        """Get the underlying client, connecting if needed."""
        if self._client is None:
            self.connect()
        return self._client

    def health_check(self) -> dict[str, Any]:
        """
        Check Qdrant health and return collection stats.

        Returns dict with 'healthy' bool and per-collection point counts.
        """
        try:
            collections = self.client.get_collections().collections
            stats = {}
            for col in collections:
                col_name = col.name
                if col_name.startswith("skyrim_"):
                    info = self.client.get_collection(col_name)
                    stats[col_name] = info.points_count
            return {"healthy": True, "collections": stats}
        except Exception as exc:
            logger.error("Qdrant health check failed: %s", exc)
            return {"healthy": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Dual-vector operations (npc_memories, world_events)
    # ------------------------------------------------------------------

    def upsert_dual_vector(
        self,
        collection: str,
        point_id: str,
        semantic_vector: list[float],
        emotional_vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """
        Upsert a point with both semantic and emotional named vectors.

        Args:
            collection: Must be a dual-vector collection.
            point_id: UUID string for the point.
            semantic_vector: 384d all-MiniLM embedding.
            emotional_vector: 9d emotional semagram.
            payload: Metadata (agent_id, tier, game_ts, content, etc.)
        """
        assert collection in DUAL_VECTOR_COLLECTIONS, (
            f"{collection} is not a dual-vector collection"
        )
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector={
                        "semantic": semantic_vector,
                        "emotional": emotional_vector,
                    },
                    payload=payload,
                )
            ],
        )

    def batch_upsert_dual_vector(
        self,
        collection: str,
        points: list[dict[str, Any]],
    ) -> None:
        """
        Batch upsert points with dual vectors.

        Each dict in points must have: id, semantic, emotional, payload.
        """
        assert collection in DUAL_VECTOR_COLLECTIONS
        structs = [
            PointStruct(
                id=p["id"],
                vector={
                    "semantic": p["semantic"],
                    "emotional": p["emotional"],
                },
                payload=p["payload"],
            )
            for p in points
        ]
        self.client.upsert(collection_name=collection, points=structs)

    def search_semantic(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: Optional[Filter] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search by semantic vector similarity.

        Returns list of dicts with 'id', 'score', 'payload'.
        """
        results = self.client.search(
            collection_name=collection,
            query_vector=NamedVector(name="semantic", vector=query_vector)
            if collection in DUAL_VECTOR_COLLECTIONS
            else query_vector,
            limit=limit,
            query_filter=filter_conditions,
            score_threshold=score_threshold,
        )
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in results
        ]

    def search_emotional(
        self,
        collection: str,
        query_vector: list[float],
        limit: int = 10,
        filter_conditions: Optional[Filter] = None,
        score_threshold: Optional[float] = None,
    ) -> list[dict[str, Any]]:
        """
        Search by emotional vector similarity (mood-congruent recall).

        For dual-vector collections, searches the 'emotional' named vector.
        For skyrim_agent_state (single 9d), searches the default vector.
        """
        if collection in DUAL_VECTOR_COLLECTIONS:
            qv = NamedVector(name="emotional", vector=query_vector)
        else:
            qv = query_vector
        results = self.client.search(
            collection_name=collection,
            query_vector=qv,
            limit=limit,
            query_filter=filter_conditions,
            score_threshold=score_threshold,
        )
        return [
            {"id": str(r.id), "score": r.score, "payload": r.payload}
            for r in results
        ]

    # ------------------------------------------------------------------
    # Single-vector operations (session_context, agent_state, lore)
    # ------------------------------------------------------------------

    def upsert_single_vector(
        self,
        collection: str,
        point_id: str,
        vector: list[float],
        payload: dict[str, Any],
    ) -> None:
        """Upsert a point with a single vector."""
        self.client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=point_id, vector=vector, payload=payload)
            ],
        )

    # ------------------------------------------------------------------
    # Payload filter helpers
    # ------------------------------------------------------------------

    @staticmethod
    def filter_by_agent(agent_id: str) -> Filter:
        """Filter points belonging to a specific agent."""
        return Filter(
            must=[FieldCondition(key="agent_id", match=MatchValue(value=agent_id))]
        )

    @staticmethod
    def filter_by_agent_and_tier(agent_id: str, tier: str) -> Filter:
        """Filter by agent_id AND compression tier (RAW/MOD/MAX)."""
        return Filter(
            must=[
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                FieldCondition(key="tier", match=MatchValue(value=tier)),
            ]
        )

    @staticmethod
    def filter_by_game_ts_range(
        gte: Optional[float] = None,
        lte: Optional[float] = None,
    ) -> Filter:
        """Filter by game timestamp range."""
        return Filter(
            must=[
                FieldCondition(
                    key="game_ts",
                    range=Range(gte=gte, lte=lte),
                )
            ]
        )

    # ------------------------------------------------------------------
    # Point retrieval by ID
    # ------------------------------------------------------------------

    def get_points(
        self,
        collection: str,
        point_ids: list[str],
        with_vectors: bool = False,
    ) -> list[dict[str, Any]]:
        """Retrieve points by their IDs."""
        results = self.client.retrieve(
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

    def scroll_by_filter(
        self,
        collection: str,
        filter_conditions: Filter,
        limit: int = 100,
        with_vectors: bool = False,
        order_by: Optional[str] = None,
    ) -> list[dict[str, Any]]:
        """
        Scroll through points matching a filter.

        Useful for retrieving all RAW points in a game_ts window
        (arc expansion during rehydration).
        """
        records, _next_offset = self.client.scroll(
            collection_name=collection,
            scroll_filter=filter_conditions,
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
        # Sort by game_ts if present in payload
        if order_by and results and order_by in (results[0].get("payload") or {}):
            results.sort(key=lambda r: r["payload"].get(order_by, 0))
        return results

    # ------------------------------------------------------------------
    # Delete
    # ------------------------------------------------------------------

    def delete_points(self, collection: str, point_ids: list[str]) -> None:
        """Delete points by ID."""
        self.client.delete(
            collection_name=collection,
            points_selector=point_ids,
        )

    # ------------------------------------------------------------------
    # Collection stats
    # ------------------------------------------------------------------

    def collection_count(self, collection: str) -> int:
        """Get point count for a collection."""
        info = self.client.get_collection(collection)
        return info.points_count
