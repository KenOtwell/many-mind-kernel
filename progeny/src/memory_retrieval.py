"""
Memory retrieval — dual-vector search with λ(t) blending.

Implements the multi-axis retrieval described in the living doc:
  1. Emotional resonance (mood-congruent, via 9d semagram similarity)
  2. Semantic similarity (text content, via 384d all-MiniLM)
  3. Role referents (payload filter: who was involved?)
  4. Recency (exponential game-time decay)
  5. Sensory anchors (-log frequency weighting for rare features)

λ(t) controls the emotional–residual retrieval balance:
  λ → 1 = emotion-first (episodes, narratives, grudges)
  λ → 0 = residual-first (domain knowledge, combat tactics)

See living doc §Multi-Axis Retrieval and §Buffer-Sequenced Retrieval.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any, Optional

from shared.constants import (
    COLLECTION_LORE,
    COLLECTION_NPC_MEMORIES,
    EMOTIONAL_DIM,
)

from .qdrant_client import MMKQdrantClient

logger = logging.getLogger(__name__)

# Retrieval tuning defaults
DEFAULT_BROAD_LIMIT = 30       # Candidates in initial broad pass
DEFAULT_FINAL_LIMIT = 8        # Final anchors after re-ranking
DEFAULT_RECENCY_HALFLIFE = 50.0  # Game-time units for 50% decay


@dataclass
class RetrievalResult:
    """A single retrieved memory with scoring breakdown."""
    point_id: str
    content: str
    tier: str
    score: float
    emotional_score: float = 0.0
    semantic_score: float = 0.0
    recency_score: float = 0.0
    anchor_boost: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class MemoryBundle:
    """
    Assembled memory context for one agent's prompt block.

    Maps to the canonical JSON schema:
      state_history.recent[]        <- RAW points around anchors
      state_history.summaries[]     <- MOD arc summaries
      state_history.expandable_refs[] <- point IDs for on-demand expansion
    """
    agent_id: str
    recent: list[dict[str, Any]] = field(default_factory=list)
    summaries: list[dict[str, Any]] = field(default_factory=list)
    expandable_refs: list[str] = field(default_factory=list)
    lore_hits: list[dict[str, Any]] = field(default_factory=list)


class MemoryRetriever:
    """
    Retrieves memories from Qdrant using multi-axis scoring.

    The retrieval pipeline:
      1. Broad resonance pass — dual-vector search with λ(t) blending
      2. Referent filtering — boost memories involving scene participants
      3. Re-rank — apply recency decay + sensory anchor boost
      4. Top anchors selected
      5. Arc expansion — find parent arc summaries, get time bounds
      6. Wrapper block retrieval — pull RAW points in arc windows
    """

    def __init__(self, client: MMKQdrantClient):
        self._client = client

    # ------------------------------------------------------------------
    # Main retrieval pipeline
    # ------------------------------------------------------------------

    def retrieve_for_agent(
        self,
        agent_id: str,
        semantic_query: list[float],
        emotional_query: list[float],
        lambda_t: float,
        current_game_ts: float,
        referents: Optional[list[str]] = None,
        broad_limit: int = DEFAULT_BROAD_LIMIT,
        final_limit: int = DEFAULT_FINAL_LIMIT,
        recency_halflife: float = DEFAULT_RECENCY_HALFLIFE,
    ) -> MemoryBundle:
        """
        Full retrieval pipeline for one agent.

        Args:
            agent_id: NPC identifier.
            semantic_query: 384d embedding of current context.
            emotional_query: 9d semagram of current emotional state.
            lambda_t: Emotional–residual balance [0, 1].
            current_game_ts: Current game timestamp for recency scoring.
            referents: NPCs in the current scene (for referent boosting).
            broad_limit: Candidates in initial pass.
            final_limit: Final anchor count.
            recency_halflife: Game-time units for 50% recency decay.

        Returns:
            MemoryBundle ready for prompt assembly.
        """
        # Step 1: Broad dual-vector search
        agent_filter = self._client.filter_by_agent(agent_id)

        emotional_hits = self._client.search_emotional(
            collection=COLLECTION_NPC_MEMORIES,
            query_vector=emotional_query,
            limit=broad_limit,
            filter_conditions=agent_filter,
        )
        semantic_hits = self._client.search_semantic(
            collection=COLLECTION_NPC_MEMORIES,
            query_vector=semantic_query,
            limit=broad_limit,
            filter_conditions=agent_filter,
        )

        # Step 2: Merge and score with λ(t) blending
        candidates = self._merge_and_score(
            emotional_hits, semantic_hits, lambda_t
        )

        # Step 3: Apply recency decay
        self._apply_recency_decay(candidates, current_game_ts, recency_halflife)

        # Step 4: Apply referent boosting
        if referents:
            self._apply_referent_boost(candidates, referents)

        # Step 5: Sort and take top anchors
        candidates.sort(key=lambda r: r.score, reverse=True)
        anchors = candidates[:final_limit]

        # Step 6: Expand anchors to arc context
        bundle = self._expand_to_bundle(agent_id, anchors)

        logger.debug(
            "Retrieved %d anchors for %s (λ=%.2f, %d recent, %d summaries)",
            len(anchors), agent_id, lambda_t,
            len(bundle.recent), len(bundle.summaries),
        )
        return bundle

    # ------------------------------------------------------------------
    # Lore retrieval (semantic only, no emotional axis)
    # ------------------------------------------------------------------

    def retrieve_lore(
        self,
        semantic_query: list[float],
        limit: int = 3,
        score_threshold: float = 0.3,
    ) -> list[dict[str, Any]]:
        """Retrieve relevant lore entries by semantic similarity."""
        return self._client.search_semantic(
            collection=COLLECTION_LORE,
            query_vector=semantic_query,
            limit=limit,
            score_threshold=score_threshold,
        )

    # ------------------------------------------------------------------
    # Internal scoring pipeline
    # ------------------------------------------------------------------

    def _merge_and_score(
        self,
        emotional_hits: list[dict[str, Any]],
        semantic_hits: list[dict[str, Any]],
        lambda_t: float,
    ) -> list[RetrievalResult]:
        """
        Merge emotional and semantic search results with λ(t) blending.

        similarity = λ(t) · emotional_sim + (1 - λ(t)) · semantic_sim

        Deduplicates by point ID, keeping the blended score.
        """
        scores: dict[str, RetrievalResult] = {}

        # Emotional hits
        for hit in emotional_hits:
            pid = hit["id"]
            payload = hit.get("payload", {})
            scores[pid] = RetrievalResult(
                point_id=pid,
                content=payload.get("content", ""),
                tier=payload.get("tier", "RAW"),
                score=lambda_t * hit["score"],
                emotional_score=hit["score"],
                payload=payload,
            )

        # Semantic hits — merge or create
        complement = 1.0 - lambda_t
        for hit in semantic_hits:
            pid = hit["id"]
            payload = hit.get("payload", {})
            if pid in scores:
                scores[pid].semantic_score = hit["score"]
                scores[pid].score += complement * hit["score"]
            else:
                scores[pid] = RetrievalResult(
                    point_id=pid,
                    content=payload.get("content", ""),
                    tier=payload.get("tier", "RAW"),
                    score=complement * hit["score"],
                    semantic_score=hit["score"],
                    payload=payload,
                )

        return list(scores.values())

    def _apply_recency_decay(
        self,
        candidates: list[RetrievalResult],
        current_ts: float,
        halflife: float,
    ) -> None:
        """
        Apply exponential recency decay to candidate scores.

        decay = 0.5 ^ (age / halflife)
        Final score = blended_score * (0.7 + 0.3 * decay)

        The 0.7 floor ensures old but strongly resonant memories
        can still surface — recency is a tiebreaker, not a gate.
        """
        if halflife <= 0:
            return
        ln2_over_halflife = math.log(2) / halflife
        for c in candidates:
            game_ts = c.payload.get("game_ts", 0)
            age = max(current_ts - game_ts, 0)
            decay = math.exp(-ln2_over_halflife * age)
            c.recency_score = decay
            c.score *= (0.7 + 0.3 * decay)

    def _apply_referent_boost(
        self,
        candidates: list[RetrievalResult],
        scene_referents: list[str],
    ) -> None:
        """
        Boost memories involving NPCs present in the current scene.

        A 20% score boost per matching referent, capped at 60%.
        """
        scene_set = frozenset(r.lower() for r in scene_referents)
        for c in candidates:
            memory_referents = c.payload.get("referents", [])
            memory_set = frozenset(r.lower() for r in memory_referents)
            overlap = len(scene_set & memory_set)
            if overlap > 0:
                boost = min(overlap * 0.2, 0.6)
                c.anchor_boost = boost
                c.score *= (1.0 + boost)

    # ------------------------------------------------------------------
    # Arc expansion
    # ------------------------------------------------------------------

    def _expand_to_bundle(
        self,
        agent_id: str,
        anchors: list[RetrievalResult],
    ) -> MemoryBundle:
        """
        Expand anchor points into a full memory bundle.

        For each anchor:
          - If it's a MOD arc summary → extract arc time bounds,
            retrieve RAW points in that window.
          - If it's a RAW point → include directly in recent[].
          - If it's a MAX essence → include as expandable_ref.
        """
        bundle = MemoryBundle(agent_id=agent_id)
        seen_raw_ids: set[str] = set()

        for anchor in anchors:
            tier = anchor.tier

            if tier == "MOD":
                # Arc summary — add to summaries and expand raw points
                bundle.summaries.append({
                    "text": anchor.content,
                    "tier": "MOD",
                    "arc_id": anchor.point_id,
                    "score": anchor.score,
                })
                # Retrieve the underlying RAW points
                raw_ids = anchor.payload.get("raw_point_ids", [])
                unseen = [rid for rid in raw_ids if rid not in seen_raw_ids]
                if unseen:
                    raw_points = self._client.get_points(
                        COLLECTION_NPC_MEMORIES, unseen
                    )
                    for rp in raw_points:
                        seen_raw_ids.add(rp["id"])
                        bundle.recent.append({
                            "text": rp["payload"].get("content", ""),
                            "game_ts": rp["payload"].get("game_ts", 0),
                            "event_type": rp["payload"].get("event_type", ""),
                        })

            elif tier == "RAW":
                if anchor.point_id not in seen_raw_ids:
                    seen_raw_ids.add(anchor.point_id)
                    bundle.recent.append({
                        "text": anchor.content,
                        "game_ts": anchor.payload.get("game_ts", 0),
                        "event_type": anchor.payload.get("event_type", ""),
                    })

            elif tier == "MAX":
                bundle.expandable_refs.append(anchor.point_id)

        # Sort recent by game_ts for chronological order
        bundle.recent.sort(key=lambda r: r.get("game_ts", 0))

        return bundle
