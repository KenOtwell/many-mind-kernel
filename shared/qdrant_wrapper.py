"""
Qdrant enrichment wrapper — single ingestion gate for the MMK.

Every piece of text entering the system flows through ingest():
  text → embed (384d semantic) → project (9d emotional) → store → return key

Both Falcon (inbound dialogue) and Progeny (LLM responses) call the
same API. One write interface, one embedding path. Nothing is stored
without its semantic fingerprint.

Also provides read_text() for key-based point lookup — used by Falcon
to read response text for wire formatting.

Requires:
  - An initialized AsyncQdrantClient (injected, not owned)
  - shared/embedding.py loaded (load_model() called at startup)
  - shared/emotional.py loaded (load_bases() called at startup)
"""
from __future__ import annotations

import logging
import time
from typing import Any, Optional
from uuid import uuid4

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

from shared import embedding, emotional

logger = logging.getLogger(__name__)


async def ingest(
    client: AsyncQdrantClient,
    text: str,
    collection: str,
    agent_id: str,
    game_ts: float,
    event_type: str = "dialogue",
    tier: str = "RAW",
    extra_payload: Optional[dict[str, Any]] = None,
) -> Optional[str]:
    """Embed text, project to 9d emotional semagram, write to Qdrant.

    The single enrichment gate: text in → key out. Both vectors
    (semantic 384d + emotional 9d) are computed here and stored with
    the point. Callers never need to touch embeddings directly.

    Args:
        client:        Async Qdrant client (initialized by the service).
        text:          Raw text to embed and store.
        collection:    Qdrant collection name.
        agent_id:      NPC or entity this text belongs to.
        game_ts:       Skyrim game timestamp.
        event_type:    Event classification (dialogue, speech, response, etc.).
        tier:          Compression tier (RAW, MOD, MAX).
        extra_payload: Additional metadata merged into the point payload.

    Returns:
        Qdrant point ID (UUID string) on success, None on failure.
    """
    if not text or not text.strip():
        return None

    # Embed + project — the core enrichment step
    try:
        emb = embedding.embed_one(text)
        semantic_vec = emb.tolist()
        emotional_vec = emotional.project(emb)
    except Exception:
        logger.exception("Enrichment failed (embed/project) for %s event", event_type)
        return None

    # Build payload
    point_id = str(uuid4())
    payload: dict[str, Any] = {
        "agent_id": agent_id,
        "tier": tier,
        "text": text,
        "event_type": event_type,
        "game_ts": game_ts,
        "wall_ts": time.time(),
    }
    if extra_payload:
        payload.update(extra_payload)

    # Write to Qdrant
    try:
        await client.upsert(
            collection_name=collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector={"semantic": semantic_vec, "emotional": emotional_vec},
                    payload=payload,
                )
            ],
        )
    except Exception:
        logger.exception(
            "Qdrant write failed — collection=%s agent=%s event=%s",
            collection, agent_id, event_type,
        )
        return None

    logger.debug(
        "Ingested: collection=%s agent=%s event=%s id=%s",
        collection, agent_id, event_type, point_id,
    )
    return point_id


async def read_text(
    client: AsyncQdrantClient,
    collection: str,
    point_id: str,
) -> Optional[str]:
    """Read stored text from a Qdrant point by ID.

    Key-based lookup — not a search. Used by Falcon to fetch response
    text for wire formatting when Progeny returns utterance keys.

    Args:
        client:     Async Qdrant client.
        collection: Qdrant collection name.
        point_id:   The point ID (UUID string) returned by ingest().

    Returns:
        The stored text string, or None if not found / unreachable.
    """
    try:
        results = await client.retrieve(
            collection_name=collection,
            ids=[point_id],
            with_payload=True,
            with_vectors=False,
        )
        if results:
            return results[0].payload.get("text")
        logger.warning("Point not found: collection=%s id=%s", collection, point_id)
        return None
    except Exception:
        logger.exception(
            "Qdrant read failed — collection=%s id=%s", collection, point_id,
        )
        return None
