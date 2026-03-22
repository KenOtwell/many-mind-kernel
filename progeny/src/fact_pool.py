"""
ATMS-style fact pool with bitvector knowledge tagging.

Every fact carries an integer bitvector — one bit per named character
(player included at bit 0). Context assembly per agent is a bitmask AND:
only facts where the agent's bit is set appear in their prompt.

One fact pool. N bit-views. No per-agent world model to maintain.

Fact currency: facts are current beliefs, not historical states. When
something changes, the old fact is superseded — agents who haven't
learned the update still hold the stale belief. Staleness is a feature.
No temporal logic engine — the LLM reasons about time from the facts
it sees.

Explicitly separate from memory compression (RAW→MOD→MAX). Compression
is storage efficiency. Fact currency is truth maintenance.

See plan: ATMS Fact Pool + Per-Agent World View.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# Player always occupies bit 0.
PLAYER_BIT_NAME = "Player"
PLAYER_BIT_POSITION = 0


@dataclass
class Fact:
    """A single fact with bitvector knowledge tagging.

    knowledge_bits: int where bit N set = NPC at index N knows this fact.
    superseded_by: if set, a newer fact replaced this one. Agents still
    holding this fact have stale (but locally valid) beliefs.
    """
    fact_id: str
    content: str
    category: str               # "event", "location", "speech", "lore", "quest", "npc_state"
    game_ts: float
    knowledge_bits: int = 0
    source_event_id: Optional[str] = None
    superseded_by: Optional[str] = None

    @property
    def is_superseded(self) -> bool:
        return self.superseded_by is not None

    def knows(self, bit_position: int) -> bool:
        """Check if the agent at bit_position knows this fact."""
        return bool(self.knowledge_bits & (1 << bit_position))

    def add_knower(self, bit_position: int) -> None:
        """Set the knowledge bit for an agent."""
        self.knowledge_bits |= (1 << bit_position)

    def add_knowers(self, bit_positions: list[int]) -> None:
        """Set knowledge bits for multiple agents at once."""
        mask = 0
        for pos in bit_positions:
            mask |= (1 << pos)
        self.knowledge_bits |= mask


class NpcBitIndex:
    """Maps NPC names to stable bit positions.

    Player is always bit 0. New NPCs get the next available bit.
    Positions are stable within a session — an NPC that leaves loaded
    cells keeps its bit so existing facts retain their knower.
    """

    def __init__(self) -> None:
        self._name_to_bit: dict[str, int] = {PLAYER_BIT_NAME: PLAYER_BIT_POSITION}
        self._bit_to_name: dict[int, str] = {PLAYER_BIT_POSITION: PLAYER_BIT_NAME}
        self._next_bit: int = 1  # 0 is reserved for Player

    def get_or_assign(self, name: str) -> int:
        """Get existing bit position or assign the next available one."""
        if name in self._name_to_bit:
            return self._name_to_bit[name]
        bit = self._next_bit
        self._next_bit += 1
        self._name_to_bit[name] = bit
        self._bit_to_name[bit] = name
        return bit

    def get(self, name: str) -> Optional[int]:
        """Get bit position for a name, or None if not registered."""
        return self._name_to_bit.get(name)

    def name_of(self, bit: int) -> Optional[str]:
        """Get name for a bit position, or None."""
        return self._bit_to_name.get(bit)

    def mask_for(self, name: str) -> int:
        """Get the single-bit mask for an NPC. Assigns if new."""
        return 1 << self.get_or_assign(name)

    def mask_for_all(self, names: list[str]) -> int:
        """Get combined mask for multiple NPCs. Assigns any new ones."""
        mask = 0
        for name in names:
            mask |= (1 << self.get_or_assign(name))
        return mask

    @property
    def count(self) -> int:
        """Number of registered NPCs (including Player)."""
        return len(self._name_to_bit)

    def all_names(self) -> list[str]:
        """All registered names in bit order."""
        return [self._bit_to_name[i] for i in range(self._next_bit)]


class FactPool:
    """In-memory fact store with ATMS-style bitvector knowledge tagging.

    Central pool of world facts. Each fact's knowledge_bits track which
    agents know it. Per-agent context assembly is a bitmask filter —
    O(N) scan, O(1) per-fact check.

    Usage:
        pool = FactPool()
        pool.add_fact("Dragon attacked Whiterun", "event", 100.0, ["Player", "Lydia"])
        lydia_facts = pool.query("Lydia")  # only facts Lydia knows
    """

    def __init__(self) -> None:
        self.bit_index = NpcBitIndex()
        self._facts: dict[str, Fact] = {}  # fact_id → Fact
        # Category index for faster filtered queries
        self._by_category: dict[str, list[str]] = {}

    @property
    def count(self) -> int:
        """Total number of facts in the pool."""
        return len(self._facts)

    # ------------------------------------------------------------------
    # Fact creation
    # ------------------------------------------------------------------

    def add_fact(
        self,
        content: str,
        category: str,
        game_ts: float,
        knower_ids: list[str],
        source_event_id: Optional[str] = None,
        fact_id: Optional[str] = None,
    ) -> Fact:
        """Create a new fact known by the listed NPCs.

        Args:
            content:         Human-readable fact text.
            category:        "event", "location", "speech", "lore", "quest", "npc_state".
            game_ts:         Game timestamp when this fact became true.
            knower_ids:      NPC names who know this fact (auto-registered).
            source_event_id: Originating TypedEvent ID, if any.
            fact_id:         Override ID (for testing). Auto-generated if None.

        Returns:
            The created Fact.
        """
        fid = fact_id or str(uuid4())
        bits = self.bit_index.mask_for_all(knower_ids)

        fact = Fact(
            fact_id=fid,
            content=content,
            category=category,
            game_ts=game_ts,
            knowledge_bits=bits,
            source_event_id=source_event_id,
        )
        self._facts[fid] = fact

        if category not in self._by_category:
            self._by_category[category] = []
        self._by_category[category].append(fid)

        return fact

    def add_lore(self, content: str, game_ts: float = 0.0) -> Fact:
        """Add a universal lore fact — all bits set for all registered NPCs.

        Lore is known by everyone. The knowledge_bits are set to cover
        all currently registered NPCs. New NPCs registered later should
        call ensure_lore_bits() to catch up.
        """
        all_names = self.bit_index.all_names()
        return self.add_fact(content, "lore", game_ts, all_names)

    def ensure_lore_bits(self, npc_name: str) -> None:
        """Set the new NPC's bit on all existing lore facts.

        Called when a new NPC is registered to ensure they know
        all universal lore.
        """
        bit = self.bit_index.get_or_assign(npc_name)
        for fid in self._by_category.get("lore", []):
            self._facts[fid].add_knower(bit)

    # ------------------------------------------------------------------
    # Knowledge propagation
    # ------------------------------------------------------------------

    def propagate_presence(
        self,
        fact_id: str,
        present_npc_ids: list[str],
    ) -> None:
        """Set knowledge bits for all NPCs present when a fact occurred.

        Used when an event happens — everyone in the cell learns the fact.
        """
        fact = self._facts.get(fact_id)
        if fact is None:
            return
        bits = self.bit_index.mask_for_all(present_npc_ids)
        fact.knowledge_bits |= bits

    def propagate_speech(
        self,
        fact_ids: list[str],
        listener_ids: list[str],
    ) -> None:
        """Propagate knowledge of specific facts to listeners.

        Used when an NPC tells others something — the listeners learn
        the referenced facts. The speaker already knows them.
        """
        listener_mask = self.bit_index.mask_for_all(listener_ids)
        for fid in fact_ids:
            fact = self._facts.get(fid)
            if fact is not None:
                fact.knowledge_bits |= listener_mask

    def propagate_earshot(
        self,
        fact_id: str,
        earshot_npc_ids: list[str],
    ) -> None:
        """Set knowledge bits for NPCs within hearing distance.

        Separate from presence — earshot is distance-based and may be
        a subset of present NPCs (exterior cells) or all of them (interiors).
        """
        self.propagate_presence(fact_id, earshot_npc_ids)

    # ------------------------------------------------------------------
    # Supersession (fact currency)
    # ------------------------------------------------------------------

    def supersede(
        self,
        old_fact_id: str,
        new_content: str,
        game_ts: float,
        new_knower_ids: list[str],
        source_event_id: Optional[str] = None,
    ) -> Optional[Fact]:
        """Supersede an old fact with a new one.

        The old fact's knowledge_bits are preserved — agents who haven't
        learned the update still hold the stale belief. The new fact
        starts with the listed knowers (typically whoever witnessed the
        change).

        Returns the new Fact, or None if old_fact_id not found.
        """
        old = self._facts.get(old_fact_id)
        if old is None:
            logger.warning("supersede: old fact %s not found", old_fact_id)
            return None

        new_fact = self.add_fact(
            content=new_content,
            category=old.category,
            game_ts=game_ts,
            knower_ids=new_knower_ids,
            source_event_id=source_event_id,
        )
        old.superseded_by = new_fact.fact_id
        return new_fact

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(
        self,
        agent_name: str,
        category: Optional[str] = None,
        include_superseded: bool = False,
        limit: Optional[int] = None,
    ) -> list[Fact]:
        """Return facts known by the named agent.

        Args:
            agent_name:         NPC name (or "Player").
            category:           Filter by category, or None for all.
            include_superseded: If False (default), skip superseded facts
                                UNLESS the agent doesn't know the superseding fact.
            limit:              Max facts to return (most recent first).

        Returns:
            List of Facts, sorted by game_ts descending (newest first).
        """
        bit = self.bit_index.get(agent_name)
        if bit is None:
            return []

        agent_mask = 1 << bit

        # Choose fact ID pool
        if category is not None:
            fact_ids = self._by_category.get(category, [])
        else:
            fact_ids = list(self._facts.keys())

        results: list[Fact] = []
        for fid in fact_ids:
            fact = self._facts[fid]
            # Agent must know this fact
            if not (fact.knowledge_bits & agent_mask):
                continue
            # Supersession filter: skip if superseded AND agent knows the replacement
            if not include_superseded and fact.is_superseded:
                replacement = self._facts.get(fact.superseded_by)
                if replacement is not None and (replacement.knowledge_bits & agent_mask):
                    continue  # Agent knows the newer version — skip stale
            results.append(fact)

        # Sort by recency (newest first)
        results.sort(key=lambda f: f.game_ts, reverse=True)

        if limit is not None:
            results = results[:limit]

        return results

    def query_recent(
        self,
        agent_name: str,
        since_ts: float,
        limit: Optional[int] = None,
    ) -> list[Fact]:
        """Return recent facts known by the agent since a given timestamp."""
        bit = self.bit_index.get(agent_name)
        if bit is None:
            return []

        agent_mask = 1 << bit
        results = [
            f for f in self._facts.values()
            if (f.knowledge_bits & agent_mask)
            and f.game_ts >= since_ts
            and not f.is_superseded
        ]
        results.sort(key=lambda f: f.game_ts, reverse=True)

        if limit is not None:
            results = results[:limit]

        return results

    # ------------------------------------------------------------------
    # Prompt helpers
    # ------------------------------------------------------------------

    def facts_for_prompt(
        self,
        agent_name: str,
        limit: int = 20,
    ) -> list[dict]:
        """Return facts formatted for inclusion in an agent's prompt block.

        Returns list of dicts with 'content', 'category', 'game_ts' —
        ready for JSON serialization into the agent's known_world.
        """
        facts = self.query(agent_name, limit=limit)
        return [
            {
                "content": f.content,
                "category": f.category,
                "game_ts": f.game_ts,
            }
            for f in facts
        ]

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_fact(self, fact_id: str) -> Optional[Fact]:
        """Retrieve a fact by ID."""
        return self._facts.get(fact_id)

    def knowers_of(self, fact_id: str) -> list[str]:
        """Return names of all agents who know a fact."""
        fact = self._facts.get(fact_id)
        if fact is None:
            return []
        names = []
        for name, bit in self.bit_index._name_to_bit.items():
            if fact.knowledge_bits & (1 << bit):
                names.append(name)
        return names
