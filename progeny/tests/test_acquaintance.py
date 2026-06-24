"""Tests for Phase 6b — acquaintance predicate, stranger detection, ledger.

Pure/synchronous: the 6b predicate is FactPool-level, so no Qdrant or model
load is needed. (6e upgrades acquaintance to a memory-retrieval check.)
"""
from __future__ import annotations

import pytest

from progeny.src import acquaintance as acq
from progeny.src.fact_pool import FactPool


def _pool() -> FactPool:
    return FactPool()


class TestMentions:
    @pytest.mark.parametrize("text,subject,expected", [
        ("Ulfric Stormcloak leads the rebellion", "Ulfric Stormcloak", True),
        ("I heard Ulfric escaped Helgen", "Ulfric Stormcloak", True),   # first-token match
        ("Lydia is my housecarl", "Lydia", True),
        ("Balgruuf rules Whiterun", "Lydia", False),
        ("", "Lydia", False),
    ])
    def test_mentions(self, text, subject, expected):
        assert acq.mentions(text, subject) is expected


class TestAreAcquainted:
    def test_event_fact_makes_acquainted(self):
        fp = _pool()
        fp.add_fact("Lydia fought beside Irileth at the watchtower", "event", 0.0, ["Lydia"])
        assert acq.are_acquainted(fp, "Lydia", "Irileth") is True

    def test_no_belief_is_not_acquainted(self):
        fp = _pool()
        fp.add_fact("Lydia fought beside Irileth", "event", 0.0, ["Lydia"])
        assert acq.are_acquainted(fp, "Lydia", "Ulfric Stormcloak") is False

    def test_unregistered_observer_is_not_acquainted(self):
        fp = _pool()
        assert acq.are_acquainted(fp, "Nobody", "Ulfric") is False

    def test_reputation_lore_counts_as_acquaintance(self):
        fp = _pool()
        fp.add_fact("Ulfric Stormcloak leads the Stormcloak rebellion", "lore", 0.0, ["Commoner"])
        assert acq.has_reputation_lore(fp, "Commoner", "Ulfric Stormcloak") is True
        assert acq.are_acquainted(fp, "Commoner", "Ulfric Stormcloak") is True


class TestStranger:
    def test_stranger_when_empty_recall_and_unacquainted(self):
        fp = _pool()
        assert acq.is_stranger(fp, "Ulfric", "Commoner", recognition_empty=True) is True

    def test_not_stranger_when_recognition_nonempty(self):
        fp = _pool()
        assert acq.is_stranger(fp, "Ulfric", "Commoner", recognition_empty=False) is False

    def test_asymmetry_via_reputation(self):
        fp = _pool()
        # The commoner has heard of Ulfric (holds reputation-lore);
        # Ulfric knows nothing of the commoner.
        fp.add_fact("Ulfric Stormcloak is Jarl of Windhelm", "lore", 0.0, ["Commoner"])
        # Commoner "knows of" Ulfric -> not a stranger even with empty recall.
        assert acq.is_stranger(fp, "Commoner", "Ulfric Stormcloak", recognition_empty=True) is False
        # Ulfric holds nothing about the commoner -> the commoner is a stranger to him.
        assert acq.is_stranger(fp, "Ulfric Stormcloak", "Commoner", recognition_empty=True) is True


class TestLedger:
    def setup_method(self):
        acq.clear()

    def teardown_method(self):
        acq.clear()

    def test_record_and_query(self):
        acq.record_stranger("Ulfric", "Commoner")
        assert acq.is_known_stranger("Ulfric", "Commoner") is True
        assert acq.strangers_of("Ulfric") == frozenset({"Commoner"})

    def test_clear_pair(self):
        acq.record_stranger("Ulfric", "Commoner")
        acq.clear_stranger("Ulfric", "Commoner")
        assert acq.is_known_stranger("Ulfric", "Commoner") is False
        assert acq.strangers_of("Ulfric") == frozenset()

    def test_clear_all(self):
        acq.record_stranger("A", "B")
        acq.record_stranger("C", "D")
        acq.clear()
        assert acq.strangers_of("A") == frozenset()
        assert acq.strangers_of("C") == frozenset()
