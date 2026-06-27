"""
Acquaintance & stranger detection — Phase 6b (Social Bootstrapping).

Acquaintance is justified belief, not a flag (plan d03f97c4). In 6b the
predicate is computed over the `FactPool`: an observer is acquainted with a
subject if the observer holds any belief that mentions the subject —
firsthand event/speech facts, identity facts, or shared reputation-lore.
(Phase 6e upgrades this to a provenance-bearing hearsay-memory retrieval
check; 6b is the FactPool-level first cut that downstream phases build on.)

Strangerness is the detected absence: on a presence change, an existing
agent regards a newcomer as a stranger when recognition retrieval surfaced
no personal memory of them AND the FactPool holds no belief about them
(no identity facts, no reputation-lore). The relationship is asymmetric by
construction — a commoner who has heard of Ulfric holds reputation-lore and
so is not a stranger to him, while Ulfric, holding nothing about the
commoner, regards the commoner as a stranger.

Everything here is a server-side signal/telemetry. Nothing is injected into
the prompt (surfacing discipline); 6c/6d consume the ledger to gate the
valence nudge and the "get acquainted" goal.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from progeny.src.fact_pool import Fact, FactPool

logger = logging.getLogger(__name__)

# Fact categories that count as "holding a belief about" someone. Lore is
# included so reputation ("knowing of" a famous NPC) counts as acquaintance
# even without a personal memory.
ACQUAINTANCE_CATEGORIES: frozenset[str] = frozenset({
    "event", "speech", "identity", "lore", "npc_state",
})


def _match_tokens(name: str) -> list[str]:
    """Lowercased match tokens for a subject name: the full name plus its
    first token (so 'Ulfric Stormcloak' matches content that says 'Ulfric').
    Tokens shorter than 3 chars are dropped to avoid spurious hits.
    """
    n = (name or "").strip().lower()
    if not n:
        return []
    tokens = [n]
    first = n.split()[0]
    if first != n and len(first) >= 3:
        tokens.append(first)
    return [t for t in tokens if len(t) >= 3]


def mentions(text: str, subject: str) -> bool:
    """True if `text` references `subject` (case-insensitive: full name or first token)."""
    if not text:
        return False
    low = text.lower()
    return any(tok in low for tok in _match_tokens(subject))


def known_beliefs_about(fact_pool: "FactPool", observer: str, subject: str) -> list["Fact"]:
    """Facts the observer knows whose content mentions the subject (acquaintance categories)."""
    return [
        f for f in fact_pool.query(observer)
        if f.category in ACQUAINTANCE_CATEGORIES and mentions(f.content, subject)
    ]


def has_reputation_lore(fact_pool: "FactPool", observer: str, subject: str) -> bool:
    """True if the observer holds lore-category belief mentioning the subject."""
    return any(mentions(f.content, subject) for f in fact_pool.query(observer, category="lore"))


def are_acquainted(fact_pool: "FactPool", observer: str, subject: str) -> bool:
    """Observer is acquainted with subject if they hold any belief about them.

    Reputation-lore is one such belief, so "knowing of" a famous NPC counts —
    which is what makes the relation asymmetric.
    """
    return bool(known_beliefs_about(fact_pool, observer, subject))


def is_stranger(
    fact_pool: "FactPool",
    observer: str,
    subject: str,
    *,
    recognition_empty: bool,
) -> bool:
    """A newcomer is a stranger to an existing observer when recognition surfaced
    no personal memory AND the FactPool holds no belief about them.

    `not are_acquainted(...)` subsumes the plan's "no reputation-lore" clause,
    since lore is one of the acquaintance belief categories.
    """
    return recognition_empty and not are_acquainted(fact_pool, observer, subject)


# ---------------------------------------------------------------------------
# Stranger ledger — server-side signal feeding 6c (valence) and 6d (goal).
# Process-lifetime, keyed observer -> set of newcomers currently seen as
# strangers. Never injected into the prompt.
# ---------------------------------------------------------------------------

_strangers: dict[str, set[str]] = {}


def record_stranger(observer: str, subject: str) -> None:
    """Mark that `observer` currently regards `subject` as a stranger."""
    _strangers.setdefault(observer, set()).add(subject)


def clear_stranger(observer: str, subject: str) -> None:
    """Drop the stranger mark once acquaintance is established."""
    seen = _strangers.get(observer)
    if seen:
        seen.discard(subject)
        if not seen:
            _strangers.pop(observer, None)


def is_known_stranger(observer: str, subject: str) -> bool:
    """True if `observer` currently regards `subject` as a stranger."""
    return subject in _strangers.get(observer, frozenset())


def strangers_of(observer: str) -> frozenset[str]:
    """All newcomers `observer` currently regards as strangers."""
    return frozenset(_strangers.get(observer, frozenset()))


def clear() -> None:
    """Drop all stranger marks (e.g., on session wipe)."""
    _strangers.clear()
