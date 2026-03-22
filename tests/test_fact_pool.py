"""Tests for progeny.src.fact_pool — ATMS bitvector fact store."""
from __future__ import annotations

from progeny.src.fact_pool import (
    Fact,
    FactPool,
    NpcBitIndex,
    PLAYER_BIT_NAME,
    PLAYER_BIT_POSITION,
)


# ---------------------------------------------------------------------------
# NpcBitIndex
# ---------------------------------------------------------------------------

class TestNpcBitIndex:
    def test_player_is_bit_zero(self):
        idx = NpcBitIndex()
        assert idx.get(PLAYER_BIT_NAME) == PLAYER_BIT_POSITION

    def test_first_npc_gets_bit_one(self):
        idx = NpcBitIndex()
        assert idx.get_or_assign("Lydia") == 1

    def test_second_npc_gets_bit_two(self):
        idx = NpcBitIndex()
        idx.get_or_assign("Lydia")
        assert idx.get_or_assign("Belethor") == 2

    def test_same_npc_returns_same_bit(self):
        idx = NpcBitIndex()
        b1 = idx.get_or_assign("Lydia")
        b2 = idx.get_or_assign("Lydia")
        assert b1 == b2

    def test_get_returns_none_for_unknown(self):
        idx = NpcBitIndex()
        assert idx.get("Nobody") is None

    def test_name_of_roundtrip(self):
        idx = NpcBitIndex()
        bit = idx.get_or_assign("Lydia")
        assert idx.name_of(bit) == "Lydia"

    def test_mask_for_single(self):
        idx = NpcBitIndex()
        idx.get_or_assign("Lydia")  # bit 1
        assert idx.mask_for("Lydia") == 0b10

    def test_mask_for_all(self):
        idx = NpcBitIndex()
        mask = idx.mask_for_all(["Player", "Lydia", "Belethor"])
        # Player=0, Lydia=1, Belethor=2 → bits 0,1,2 → 0b111 = 7
        assert mask == 0b111

    def test_count_includes_player(self):
        idx = NpcBitIndex()
        assert idx.count == 1  # Player
        idx.get_or_assign("Lydia")
        assert idx.count == 2

    def test_all_names_in_order(self):
        idx = NpcBitIndex()
        idx.get_or_assign("Lydia")
        idx.get_or_assign("Belethor")
        assert idx.all_names() == ["Player", "Lydia", "Belethor"]


# ---------------------------------------------------------------------------
# Fact
# ---------------------------------------------------------------------------

class TestFact:
    def test_knows_checks_bit(self):
        f = Fact(fact_id="f1", content="test", category="event", game_ts=1.0,
                 knowledge_bits=0b110)  # bits 1 and 2
        assert f.knows(0) is False  # Player
        assert f.knows(1) is True   # bit 1
        assert f.knows(2) is True   # bit 2

    def test_add_knower(self):
        f = Fact(fact_id="f1", content="test", category="event", game_ts=1.0)
        assert f.knows(3) is False
        f.add_knower(3)
        assert f.knows(3) is True

    def test_add_knowers_batch(self):
        f = Fact(fact_id="f1", content="test", category="event", game_ts=1.0)
        f.add_knowers([1, 3, 5])
        assert f.knows(1) is True
        assert f.knows(2) is False
        assert f.knows(3) is True
        assert f.knows(5) is True

    def test_is_superseded(self):
        f = Fact(fact_id="f1", content="old", category="event", game_ts=1.0)
        assert f.is_superseded is False
        f.superseded_by = "f2"
        assert f.is_superseded is True


# ---------------------------------------------------------------------------
# FactPool — creation and basic queries
# ---------------------------------------------------------------------------

class TestFactPoolCreation:
    def test_add_fact_returns_fact(self):
        pool = FactPool()
        fact = pool.add_fact("Dragon attacks", "event", 100.0, ["Player", "Lydia"])
        assert fact.content == "Dragon attacks"
        assert fact.category == "event"

    def test_player_and_lydia_know_fact(self):
        pool = FactPool()
        fact = pool.add_fact("Dragon attacks", "event", 100.0, ["Player", "Lydia"])
        assert fact.knows(PLAYER_BIT_POSITION)
        lydia_bit = pool.bit_index.get("Lydia")
        assert fact.knows(lydia_bit)

    def test_belethor_does_not_know(self):
        pool = FactPool()
        pool.add_fact("Dragon attacks", "event", 100.0, ["Player", "Lydia"])
        pool.bit_index.get_or_assign("Belethor")  # register but don't tell him
        facts = pool.query("Belethor")
        assert len(facts) == 0

    def test_count(self):
        pool = FactPool()
        pool.add_fact("a", "event", 1.0, ["Player"])
        pool.add_fact("b", "event", 2.0, ["Player"])
        assert pool.count == 2

    def test_custom_fact_id(self):
        pool = FactPool()
        fact = pool.add_fact("test", "event", 1.0, ["Player"], fact_id="custom-id")
        assert fact.fact_id == "custom-id"
        assert pool.get_fact("custom-id") is not None


# ---------------------------------------------------------------------------
# FactPool — queries
# ---------------------------------------------------------------------------

class TestFactPoolQuery:
    def test_query_returns_only_known_facts(self):
        pool = FactPool()
        pool.add_fact("Public event", "event", 100.0, ["Player", "Lydia", "Belethor"])
        pool.add_fact("Lydia's secret", "event", 200.0, ["Lydia"])
        pool.add_fact("Player saw something", "event", 300.0, ["Player"])

        lydia_facts = pool.query("Lydia")
        assert len(lydia_facts) == 2  # public + her secret
        contents = {f.content for f in lydia_facts}
        assert "Public event" in contents
        assert "Lydia's secret" in contents
        assert "Player saw something" not in contents

    def test_query_sorted_by_recency(self):
        pool = FactPool()
        pool.add_fact("old", "event", 100.0, ["Player"])
        pool.add_fact("new", "event", 200.0, ["Player"])
        facts = pool.query("Player")
        assert facts[0].content == "new"
        assert facts[1].content == "old"

    def test_query_with_limit(self):
        pool = FactPool()
        for i in range(10):
            pool.add_fact(f"fact {i}", "event", float(i), ["Player"])
        facts = pool.query("Player", limit=3)
        assert len(facts) == 3
        # Should be the 3 most recent
        assert facts[0].game_ts == 9.0

    def test_query_by_category(self):
        pool = FactPool()
        pool.add_fact("battle", "event", 100.0, ["Player"])
        pool.add_fact("arrived at Whiterun", "location", 200.0, ["Player"])
        pool.add_fact("quest started", "quest", 300.0, ["Player"])

        events = pool.query("Player", category="event")
        assert len(events) == 1
        assert events[0].content == "battle"

    def test_query_unknown_agent_returns_empty(self):
        pool = FactPool()
        pool.add_fact("something", "event", 1.0, ["Player"])
        assert pool.query("Nobody") == []

    def test_query_recent(self):
        pool = FactPool()
        pool.add_fact("old", "event", 100.0, ["Player"])
        pool.add_fact("new", "event", 500.0, ["Player"])
        facts = pool.query_recent("Player", since_ts=200.0)
        assert len(facts) == 1
        assert facts[0].content == "new"


# ---------------------------------------------------------------------------
# FactPool — propagation
# ---------------------------------------------------------------------------

class TestFactPoolPropagation:
    def test_propagate_presence_adds_knowers(self):
        pool = FactPool()
        fact = pool.add_fact("Explosion", "event", 100.0, ["Player"])
        pool.propagate_presence(fact.fact_id, ["Lydia", "Belethor"])
        # Now Lydia and Belethor should know
        assert len(pool.query("Lydia")) == 1
        assert len(pool.query("Belethor")) == 1

    def test_propagate_speech_transfers_knowledge(self):
        pool = FactPool()
        secret = pool.add_fact("Hidden treasure", "event", 100.0, ["Player"])
        # Player tells Lydia
        pool.propagate_speech([secret.fact_id], ["Lydia"])
        assert len(pool.query("Lydia")) == 1
        # Belethor wasn't told
        pool.bit_index.get_or_assign("Belethor")
        assert len(pool.query("Belethor")) == 0

    def test_speech_propagation_chain(self):
        """A tells B, B tells C — witness chain."""
        pool = FactPool()
        fact = pool.add_fact("Secret plan", "event", 100.0, ["Player"])
        # Player tells Lydia
        pool.propagate_speech([fact.fact_id], ["Lydia"])
        # Lydia tells Belethor
        pool.propagate_speech([fact.fact_id], ["Belethor"])
        # All three know
        assert len(pool.query("Player")) == 1
        assert len(pool.query("Lydia")) == 1
        assert len(pool.query("Belethor")) == 1

    def test_propagate_nonexistent_fact_is_noop(self):
        pool = FactPool()
        pool.propagate_presence("nonexistent", ["Lydia"])  # should not raise

    def test_enemy_npc_gets_knowledge(self):
        """Enemy NPCs accumulate knowledge the same way friendlies do."""
        pool = FactPool()
        # Player discusses patrol route in earshot of bandit spy
        fact = pool.add_fact(
            "Patrol route goes through the eastern pass",
            "event", 100.0, ["Player", "Lydia"],
        )
        pool.propagate_earshot(fact.fact_id, ["BanditSpy"])
        # Spy knows, player doesn't know spy knows
        spy_facts = pool.query("BanditSpy")
        assert len(spy_facts) == 1
        assert spy_facts[0].content == "Patrol route goes through the eastern pass"


# ---------------------------------------------------------------------------
# FactPool — supersession (fact currency)
# ---------------------------------------------------------------------------

class TestFactPoolSupersession:
    def test_supersede_creates_new_fact(self):
        pool = FactPool()
        old = pool.add_fact("Camp is empty", "event", 100.0, ["Player", "Lydia"])
        new = pool.supersede(old.fact_id, "Reinforcements arrived", 200.0, ["BanditChief"])
        assert new is not None
        assert new.content == "Reinforcements arrived"
        assert old.superseded_by == new.fact_id

    def test_stale_belief_preserved_for_uninformed_agent(self):
        """Lydia still believes camp is empty because she doesn't know about reinforcements."""
        pool = FactPool()
        old = pool.add_fact("Camp is empty", "event", 100.0, ["Player", "Lydia"])
        pool.supersede(old.fact_id, "Reinforcements arrived", 200.0, ["BanditChief"])

        # Lydia's query: she doesn't know the superseding fact, so old fact surfaces
        lydia_facts = pool.query("Lydia")
        assert len(lydia_facts) == 1
        assert lydia_facts[0].content == "Camp is empty"

    def test_informed_agent_sees_new_fact_not_old(self):
        """BanditChief knows about reinforcements, doesn't see stale 'empty' fact."""
        pool = FactPool()
        old = pool.add_fact("Camp is empty", "event", 100.0, ["BanditChief"])
        pool.supersede(old.fact_id, "Reinforcements arrived", 200.0, ["BanditChief"])

        chief_facts = pool.query("BanditChief")
        assert len(chief_facts) == 1
        assert chief_facts[0].content == "Reinforcements arrived"

    def test_include_superseded_flag(self):
        pool = FactPool()
        old = pool.add_fact("Camp is empty", "event", 100.0, ["BanditChief"])
        pool.supersede(old.fact_id, "Reinforcements arrived", 200.0, ["BanditChief"])

        # With include_superseded, both appear
        all_facts = pool.query("BanditChief", include_superseded=True)
        assert len(all_facts) == 2

    def test_supersede_nonexistent_returns_none(self):
        pool = FactPool()
        result = pool.supersede("nonexistent", "new", 1.0, ["Player"])
        assert result is None

    def test_supersession_chain_preserves_category(self):
        pool = FactPool()
        old = pool.add_fact("Gate is open", "location", 100.0, ["Player"])
        new = pool.supersede(old.fact_id, "Gate is closed", 200.0, ["Player"])
        assert new.category == "location"


# ---------------------------------------------------------------------------
# FactPool — lore
# ---------------------------------------------------------------------------

class TestFactPoolLore:
    def test_add_lore_known_by_all(self):
        pool = FactPool()
        pool.bit_index.get_or_assign("Lydia")
        pool.bit_index.get_or_assign("Belethor")
        pool.add_lore("Skyrim is the homeland of the Nords")
        assert len(pool.query("Player")) == 1
        assert len(pool.query("Lydia")) == 1
        assert len(pool.query("Belethor")) == 1

    def test_ensure_lore_bits_for_new_npc(self):
        pool = FactPool()
        pool.add_lore("The Empire and the Stormcloaks are at war")
        # Register new NPC after lore was added
        pool.bit_index.get_or_assign("Heimskr")
        pool.ensure_lore_bits("Heimskr")
        assert len(pool.query("Heimskr")) == 1


# ---------------------------------------------------------------------------
# FactPool — prompt helpers
# ---------------------------------------------------------------------------

class TestFactsForPrompt:
    def test_returns_dicts(self):
        pool = FactPool()
        pool.add_fact("Dragon attacked", "event", 100.0, ["Player"])
        result = pool.facts_for_prompt("Player")
        assert len(result) == 1
        assert result[0]["content"] == "Dragon attacked"
        assert result[0]["category"] == "event"
        assert result[0]["game_ts"] == 100.0

    def test_respects_limit(self):
        pool = FactPool()
        for i in range(20):
            pool.add_fact(f"fact {i}", "event", float(i), ["Player"])
        result = pool.facts_for_prompt("Player", limit=5)
        assert len(result) == 5

    def test_per_agent_filtering(self):
        """Different agents see different facts in their prompt."""
        pool = FactPool()
        pool.add_fact("Shared event", "event", 100.0, ["Player", "Lydia", "BanditChief"])
        pool.add_fact("Lydia overheard plan", "event", 200.0, ["Lydia"])
        pool.add_fact("Bandit reinforcements", "event", 300.0, ["BanditChief"])

        player_prompt = pool.facts_for_prompt("Player")
        lydia_prompt = pool.facts_for_prompt("Lydia")
        chief_prompt = pool.facts_for_prompt("BanditChief")

        assert len(player_prompt) == 1
        assert len(lydia_prompt) == 2
        assert len(chief_prompt) == 2

        # Player doesn't know about overheard plan OR reinforcements
        player_contents = {f["content"] for f in player_prompt}
        assert "Lydia overheard plan" not in player_contents
        assert "Bandit reinforcements" not in player_contents


# ---------------------------------------------------------------------------
# FactPool — introspection
# ---------------------------------------------------------------------------

class TestFactPoolIntrospection:
    def test_knowers_of(self):
        pool = FactPool()
        fact = pool.add_fact("test", "event", 1.0, ["Player", "Lydia"])
        knowers = pool.knowers_of(fact.fact_id)
        assert "Player" in knowers
        assert "Lydia" in knowers
        assert len(knowers) == 2

    def test_knowers_of_nonexistent(self):
        pool = FactPool()
        assert pool.knowers_of("nope") == []
