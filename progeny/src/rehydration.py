"""
Rehydration — expand compressed memory refs and recover stashed context.

Two rehydration modes:

1. **Ref expansion** — Expand MAX-tier expandable_refs from the memory
   bundle into full context. Walks the compression chain:
   MAX essence → source arc IDs → MOD summaries → raw_point_ids → RAW text.

2. **Post-interruption recovery** — When curvature stabilizes after a snap
   spike, retrieve the stashed pre-interruption context from
   skyrim_session_context and re-inject into the prompt. The LLM naturally
   produces recovery dialogue: "Where were we?", "As I was saying..."

Asymmetric timing: stash is instant (one snap spike). Restore is slow
(governed by curvature decay through harmonic buffers). You go under
instantly; you come back gradually.

See living doc §Pre-Interruption Stash & Context Rehydration.
"""
from __future__ import annotations

import logging
from typing import Any, Optional

from qdrant_client.http.models import (
    FieldCondition,
    Filter,
    MatchValue,
)

from shared.constants import (
    COLLECTION_NPC_MEMORIES,
    COLLECTION_SESSION_CONTEXT,
)

from .qdrant_client import MMKQdrantClient

logger = logging.getLogger(__name__)

# Curvature threshold below which we consider the agent "stabilized"
# and ready for context rehydration. Tunable per-agent.
DEFAULT_STABILIZATION_THRESHOLD = 0.1

# Number of consecutive ticks below threshold required before rehydrating.
# Prevents premature rehydration during brief lulls in combat.
DEFAULT_STABILIZATION_TICKS = 3


class Rehydrator:
    """
    Expands compressed memory references and recovers stashed context.

    Works with the memory retrieval pipeline: after retrieval returns
    a MemoryBundle with expandable_refs, the rehydrator can chase those
    refs down to full RAW text for inclusion in the prompt.
    """

    def __init__(self, client: MMKQdrantClient):
        self._client = client
        # Per-agent stabilization counters: agent_id → ticks_below_threshold
        self._stabilization_counters: dict[str, int] = {}

    # ------------------------------------------------------------------
    # Ref expansion (MAX → MOD → RAW)
    # ------------------------------------------------------------------

    def expand_refs(
        self,
        expandable_refs: list[str],
        max_raw_points: int = 20,
    ) -> list[dict[str, Any]]:
        """
        Expand MAX-tier point IDs into full context.

        Walks the compression chain:
          MAX point → source_arc_ids → MOD arc summaries
          MOD arc summary → raw_point_ids → RAW event text

        Returns a list of dicts with 'text', 'game_ts', 'event_type',
        sorted chronologically.
        """
        if not expandable_refs:
            return []

        # Step 1: Retrieve MAX points
        max_points = self._client.get_points(
            COLLECTION_NPC_MEMORIES, expandable_refs
        )
        if not max_points:
            return []

        # Step 2: Collect all source arc IDs
        arc_ids: list[str] = []
        for mp in max_points:
            arc_ids.extend(mp["payload"].get("source_arc_ids", []))

        if not arc_ids:
            # No arc chain — return the MAX essence text directly
            return [
                {
                    "text": mp["payload"].get("content", ""),
                    "game_ts": mp["payload"].get("game_ts", 0),
                    "event_type": "compressed_essence",
                }
                for mp in max_points
            ]

        # Step 3: Retrieve MOD arc summaries
        arc_points = self._client.get_points(COLLECTION_NPC_MEMORIES, arc_ids)

        # Step 4: Collect all raw point IDs from arcs
        raw_ids: list[str] = []
        for ap in arc_points:
            raw_ids.extend(ap["payload"].get("raw_point_ids", []))

        # Cap the number of RAW points to prevent prompt explosion
        raw_ids = raw_ids[:max_raw_points]

        if not raw_ids:
            # No RAW chain — return arc summary text
            return [
                {
                    "text": ap["payload"].get("content", ""),
                    "game_ts": ap["payload"].get("game_ts", 0),
                    "event_type": "arc_summary",
                }
                for ap in arc_points
            ]

        # Step 5: Retrieve RAW points
        raw_points = self._client.get_points(COLLECTION_NPC_MEMORIES, raw_ids)

        results = [
            {
                "text": rp["payload"].get("content", ""),
                "game_ts": rp["payload"].get("game_ts", 0),
                "event_type": rp["payload"].get("event_type", ""),
            }
            for rp in raw_points
        ]
        results.sort(key=lambda r: r.get("game_ts", 0))
        return results

    # ------------------------------------------------------------------
    # Post-interruption context recovery
    # ------------------------------------------------------------------

    def check_stabilization(
        self,
        agent_id: str,
        curvature: float,
        threshold: float = DEFAULT_STABILIZATION_THRESHOLD,
        required_ticks: int = DEFAULT_STABILIZATION_TICKS,
    ) -> bool:
        """
        Check if an agent's curvature has stabilized enough for rehydration.

        Returns True when curvature has been below threshold for
        required_ticks consecutive ticks.
        """
        if abs(curvature) < threshold:
            count = self._stabilization_counters.get(agent_id, 0) + 1
            self._stabilization_counters[agent_id] = count
            return count >= required_ticks
        else:
            # Reset counter — curvature spiked again
            self._stabilization_counters[agent_id] = 0
            return False

    def recover_stashed_context(
        self,
        agent_id: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Retrieve un-rehydrated stashed context for an agent.

        Returns stashed context entries (most recent first) that haven't
        been rehydrated yet. After retrieval, marks them as rehydrated.
        """
        stash_filter = Filter(
            must=[
                FieldCondition(key="agent_id", match=MatchValue(value=agent_id)),
                FieldCondition(key="rehydrated", match=MatchValue(value=False)),
            ]
        )
        stashed = self._client.scroll_by_filter(
            collection=COLLECTION_SESSION_CONTEXT,
            filter_conditions=stash_filter,
            limit=limit,
        )

        if not stashed:
            return []

        # Mark as rehydrated so we don't re-inject on subsequent turns
        for entry in stashed:
            self._client.client.set_payload(
                collection_name=COLLECTION_SESSION_CONTEXT,
                payload={"rehydrated": True},
                points=[entry["id"]],
            )

        # Reset stabilization counter
        self._stabilization_counters.pop(agent_id, None)

        logger.info(
            "Rehydrated %d stashed contexts for %s", len(stashed), agent_id
        )
        return [
            {
                "text": s["payload"].get("content", ""),
                "stash_reason": s["payload"].get("stash_reason", ""),
                "stashed_at": s["payload"].get("timestamp", ""),
            }
            for s in stashed
        ]

    def reset_agent(self, agent_id: str) -> None:
        """Reset stabilization counter for an agent (e.g., on cell transition)."""
        self._stabilization_counters.pop(agent_id, None)
