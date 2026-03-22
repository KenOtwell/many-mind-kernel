"""
Memory writer — single write authority for all Qdrant tiers.

Writes RAW events (immutable, every event), MOD arc summaries (on snap
threshold crossings), and MAX compressed essences (operational compaction).

Every write attaches both semantic and emotional vectors. The emotional
vector IS the retrieval key for mood-congruent recall — not a side channel.

See living doc §Memory Architecture and §Storage Trigger.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from uuid import uuid4

from qdrant_client.models import PointStruct

from shared.constants import (
    COLLECTION_AGENT_STATE,
    COLLECTION_LORE,
    COLLECTION_NPC_MEMORIES,
    COLLECTION_SESSION_CONTEXT,
    COLLECTION_WORLD_EVENTS,
)
from shared.schemas import CompressionTier, PrivacyLevel

from .qdrant_client import get_client

logger = logging.getLogger(__name__)


class MemoryWriter:
    """
    Handles all Qdrant writes for the Many-Mind Kernel.

    Three tiers:
      RAW  — every event, always stored, never modified. Immutable log.
      MOD  — arc summaries. Condensed descriptions of emotional arcs.
             Search aids to find relevant RAW points.
      MAX  — compressed essences. Operational compaction when RAW count
             impacts performance. Not part of core cognitive loop.

    All writes go through this class. Single write authority simplifies
    consistency and makes the write log auditable.
    """

    # ------------------------------------------------------------------
    # RAW tier — immutable event log
    # ------------------------------------------------------------------

    async def write_raw_event(
        self,
        agent_id: str,
        content: str,
        semantic_vector: list[float],
        emotional_vector: list[float],
        game_ts: float,
        event_type: str,
        referents: Optional[list[str]] = None,
        location: Optional[str] = None,
        privacy_level: str = PrivacyLevel.COLLECTIVE.value,
        extra_payload: Optional[dict[str, Any]] = None,
    ) -> str:
        """
        Write a RAW event point. Called for every event that enters
        the emotional delta pipeline — both inbound game events AND
        outbound LLM response text (bidirectional).

        Returns the point ID.
        """
        point_id = str(uuid4())
        payload: dict[str, Any] = {
            "agent_id": agent_id,
            "tier": CompressionTier.RAW.value,
            "content": content,
            "event_type": event_type,
            "game_ts": game_ts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "referents": referents or [],
            "location": location or "",
            "privacy_level": privacy_level,
        }
        if extra_payload:
            payload.update(extra_payload)

        try:
            await get_client().upsert(
                collection_name=COLLECTION_NPC_MEMORIES,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"semantic": semantic_vector, "emotional": emotional_vector},
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            logger.error("RAW write failed — agent=%s event=%s: %s", agent_id, event_type, exc)

        logger.debug(
            "RAW write: agent=%s event=%s id=%s", agent_id, event_type, point_id
        )
        return point_id

    async def write_raw_batch(
        self,
        events: list[dict[str, Any]],
    ) -> list[str]:
        """
        Batch write RAW events. Each dict needs: agent_id, content,
        semantic_vector, emotional_vector, game_ts, event_type.
        Optional: referents, location, privacy_level, extra_payload.

        Returns list of point IDs.
        """
        points = []
        ids = []
        now = datetime.now(timezone.utc).isoformat()
        for evt in events:
            point_id = str(uuid4())
            ids.append(point_id)
            payload = {
                "agent_id": evt["agent_id"],
                "tier": CompressionTier.RAW.value,
                "content": evt["content"],
                "event_type": evt["event_type"],
                "game_ts": evt["game_ts"],
                "timestamp": now,
                "referents": evt.get("referents", []),
                "location": evt.get("location", ""),
                "privacy_level": evt.get(
                    "privacy_level", PrivacyLevel.COLLECTIVE.value
                ),
            }
            extra = evt.get("extra_payload")
            if extra:
                payload.update(extra)
            points.append(
                PointStruct(
                    id=point_id,
                    vector={
                        "semantic": evt["semantic_vector"],
                        "emotional": evt["emotional_vector"],
                    },
                    payload=payload,
                )
            )
        if points:
            try:
                await get_client().upsert(
                    collection_name=COLLECTION_NPC_MEMORIES,
                    points=points,
                )
            except Exception as exc:
                logger.error("RAW batch write failed — %d points: %s", len(points), exc)
            logger.debug("RAW batch write: %d points", len(points))
        return ids

    # ------------------------------------------------------------------
    # MOD tier — arc summaries
    # ------------------------------------------------------------------

    async def write_arc_summary(
        self,
        agent_id: str,
        summary_text: str,
        semantic_vector: list[float],
        emotional_vector: list[float],
        arc_start_ts: float,
        arc_end_ts: float,
        raw_point_ids: list[str],
        game_ts: float,
        location: Optional[str] = None,
    ) -> str:
        """
        Write a MOD-tier arc summary.

        Generated when snap exceeds threshold (event boundary detected).
        The summary spans from arc_start_ts to arc_end_ts and references
        the RAW points it was derived from.

        Returns the summary point ID.
        """
        point_id = str(uuid4())
        payload = {
            "agent_id": agent_id,
            "tier": CompressionTier.MOD.value,
            "content": summary_text,
            "event_type": "arc_summary",
            "game_ts": game_ts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "arc_start_ts": arc_start_ts,
            "arc_end_ts": arc_end_ts,
            "raw_point_ids": raw_point_ids,
            "location": location or "",
            "referents": [],
            "privacy_level": PrivacyLevel.COLLECTIVE.value,
        }
        try:
            await get_client().upsert(
                collection_name=COLLECTION_NPC_MEMORIES,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"semantic": semantic_vector, "emotional": emotional_vector},
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            logger.error(
                "MOD write failed — agent=%s arc=[%.2f, %.2f]: %s",
                agent_id, arc_start_ts, arc_end_ts, exc,
            )

        logger.debug(
            "MOD write (arc summary): agent=%s arc=[%.2f, %.2f] id=%s",
            agent_id, arc_start_ts, arc_end_ts, point_id,
        )
        return point_id

    # ------------------------------------------------------------------
    # MAX tier — compressed essences (operational compaction)
    # ------------------------------------------------------------------

    async def write_compressed_essence(
        self,
        agent_id: str,
        essence_text: str,
        semantic_vector: list[float],
        emotional_vector: list[float],
        source_arc_ids: list[str],
        game_ts: float,
    ) -> str:
        """
        Write a MAX-tier compressed essence.

        Returns the essence point ID.
        """
        point_id = str(uuid4())
        payload = {
            "agent_id": agent_id,
            "tier": CompressionTier.MAX.value,
            "content": essence_text,
            "event_type": "compressed_essence",
            "game_ts": game_ts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source_arc_ids": source_arc_ids,
            "privacy_level": PrivacyLevel.COLLECTIVE.value,
        }
        try:
            await get_client().upsert(
                collection_name=COLLECTION_NPC_MEMORIES,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"semantic": semantic_vector, "emotional": emotional_vector},
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            logger.error(
                "MAX write failed — agent=%s from %d arcs: %s",
                agent_id, len(source_arc_ids), exc,
            )

        logger.debug(
            "MAX write (essence): agent=%s from %d arcs id=%s",
            agent_id, len(source_arc_ids), point_id,
        )
        return point_id

    # ------------------------------------------------------------------
    # World events
    # ------------------------------------------------------------------

    async def write_world_event(
        self,
        event_type: str,
        content: str,
        semantic_vector: list[float],
        emotional_vector: list[float],
        game_ts: float,
        location: Optional[str] = None,
        involved_npcs: Optional[list[str]] = None,
    ) -> str:
        """Write a world-level event (weather, location changes, deaths, etc.)."""
        point_id = str(uuid4())
        payload = {
            "event_type": event_type,
            "content": content,
            "game_ts": game_ts,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "location": location or "",
            "involved_npcs": involved_npcs or [],
        }
        try:
            await get_client().upsert(
                collection_name=COLLECTION_WORLD_EVENTS,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"semantic": semantic_vector, "emotional": emotional_vector},
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            logger.error("World event write failed — type=%s: %s", event_type, exc)
        return point_id

    # ------------------------------------------------------------------
    # Agent state snapshots
    # ------------------------------------------------------------------

    async def write_agent_state(
        self,
        agent_id: str,
        emotional_vector: list[float],
        harmonic_buffers: dict[str, list[float]],
        curvature: float,
        snap: float,
        lambda_t: float,
        coherence: float,
        actor_values: Optional[dict[str, int]] = None,
    ) -> str:
        """
        Snapshot current agent harmonic state to skyrim_agent_state.

        Called after each turn to persist the agent's emotional trajectory.
        The emotional vector enables "find past states that felt like this"
        queries — supporting the habituation → instinct pathway.
        """
        point_id = str(uuid4())
        payload = {
            "agent_id": agent_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "harmonic_buffers": harmonic_buffers,
            "curvature": curvature,
            "snap": snap,
            "lambda_t": lambda_t,
            "coherence": coherence,
            "actor_values": actor_values or {},
        }
        try:
            await get_client().upsert(
                collection_name=COLLECTION_AGENT_STATE,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"emotional": emotional_vector},
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            logger.error("Agent state write failed — agent=%s: %s", agent_id, exc)
        return point_id

    # ------------------------------------------------------------------
    # Session context (pre-interruption stash)
    # ------------------------------------------------------------------

    async def stash_session_context(
        self,
        agent_id: str,
        context_text: str,
        semantic_vector: list[float],
        stash_reason: str = "snap_threshold",
    ) -> str:
        """
        Stash conversational context on snap spike (pre-interruption).

        Stored in skyrim_session_context for rehydration when curvature
        stabilizes. See living doc §Pre-Interruption Stash & Context
        Rehydration.
        """
        point_id = str(uuid4())
        payload = {
            "agent_id": agent_id,
            "content": context_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "stash_reason": stash_reason,
            "rehydrated": False,
        }
        try:
            await get_client().upsert(
                collection_name=COLLECTION_SESSION_CONTEXT,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"semantic": semantic_vector},
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            logger.error("Session stash failed — agent=%s: %s", agent_id, exc)

        logger.debug(
            "Session stash: agent=%s reason=%s id=%s",
            agent_id, stash_reason, point_id,
        )
        return point_id

    # ------------------------------------------------------------------
    # Lore
    # ------------------------------------------------------------------

    async def write_lore_entry(
        self,
        topic: str,
        content: str,
        semantic_vector: list[float],
        source: str = "oghma_infinium",
    ) -> str:
        """Write a static lore entry (Oghma Infinium, book content, etc.)."""
        point_id = str(uuid4())
        payload = {
            "topic": topic,
            "content": content,
            "source": source,
        }
        try:
            await get_client().upsert(
                collection_name=COLLECTION_LORE,
                points=[
                    PointStruct(
                        id=point_id,
                        vector={"semantic": semantic_vector},
                        payload=payload,
                    )
                ],
            )
        except Exception as exc:
            logger.error("Lore write failed — topic=%s: %s", topic, exc)
        return point_id
