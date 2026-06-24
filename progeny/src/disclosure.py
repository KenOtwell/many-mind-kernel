"""
Disclosure to hearsay propagation — Phase 6e (Social Bootstrapping).

When an NPC speaks to someone it has not met, the exchange functions as an
introduction and closes the acquaintance loop. Propagation is **reciprocal** —
it leaves a memory in both minds (plan d03f97c4):

  * the listener gains a provenance-bearing **hearsay memory** about the
    speaker ("{speaker} introduced themselves — a blacksmith from Riften"),
    written under the listener's own ``agent_id`` with the speaker as referent
    and how-it-was-learned recorded;
  * the speaker gains a complementary **disclosure ("telling") memory** ("I
    introduced myself to {listener}; now they know who I am"), under the
    speaker's own ``agent_id`` with the listener as referent. This is the
    meta-belief that lets an NPC answer "How do you know John knows x?" with
    "I told him" — and it is why *both* parties recognize each other next time,
    not just the listener.

A thin symbolic **identity fact** is also written to the ``FactPool`` (known by
both parties) so the 6b ``are_acquainted`` predicate returns true without a
retrieval round-trip, and the stranger ledger is cleared in both directions so
the 6d get-acquainted goal resolves.

Idempotence: deterministic point/fact IDs — the hearsay on (listener, speaker),
the telling on (speaker, listener), the fact on (speaker, listener) — so a
retelling upserts the same points and reinforces rather than duplicates.

Model-free: the caller passes the identity semantic vector (embedded once per
speaker) and the per-agent reaction vectors, so this module needs no embedding
model and is fully unit-testable.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from uuid import NAMESPACE_DNS, uuid5

from progeny.src import acquaintance

if TYPE_CHECKING:
    from progeny.src.fact_pool import FactPool
    from progeny.src.identity_kernel import IdentityKernel
    from progeny.src.memory_writer import MemoryWriter

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Content builders (pure)
# ---------------------------------------------------------------------------

def identity_descriptor(kernel: "IdentityKernel") -> str:
    """A short public descriptor from the kernel (occupation + origin).

    Public-only (6a partition); empty when the NPC has no seeded profile, in
    which case the encounter still de-strangers the pair via a generic line.
    """
    parts: list[str] = []
    occupation = kernel.occupation()
    if occupation:
        parts.append(f"a {occupation}")
    origin = str(kernel.public.get("origin", "")).strip()
    if origin:
        parts.append(f"from {origin}")
    return ", ".join(parts)


def hearsay_content(speaker: str, descriptor: str) -> str:
    """The listener's memory of being introduced to the speaker."""
    if descriptor:
        return f"{speaker} introduced themselves — {descriptor}."
    return f"{speaker} introduced themselves."


def telling_content(listener: str) -> str:
    """The speaker's reciprocal memory that the listener now knows them."""
    return f"I introduced myself to {listener}; now {listener} knows who I am."


def fact_content(speaker: str, descriptor: str) -> str:
    """The symbolic identity belief (mentions the speaker, for are_acquainted)."""
    if descriptor:
        return f"{speaker} is {descriptor}."
    return f"{speaker} introduced themselves."


# ---------------------------------------------------------------------------
# Deterministic IDs (idempotent reinforcement)
# ---------------------------------------------------------------------------

def hearsay_id(listener: str, speaker: str) -> str:
    return str(uuid5(NAMESPACE_DNS, f"mmk:hearsay:{listener}:{speaker}:identity"))


def telling_id(speaker: str, listener: str) -> str:
    return str(uuid5(NAMESPACE_DNS, f"mmk:telling:{speaker}:{listener}:identity"))


def identity_fact_id(speaker: str, listener: str) -> str:
    return str(uuid5(NAMESPACE_DNS, f"mmk:identity_fact:{speaker}:{listener}"))


# ---------------------------------------------------------------------------
# Propagation
# ---------------------------------------------------------------------------

async def propagate_introduction(
    *,
    writer: "MemoryWriter",
    fact_pool: "FactPool | None",
    speaker: str,
    listener: str,
    speaker_kernel: "IdentityKernel",
    identity_semantic_vec: list[float],
    listener_reaction: list[float],
    speaker_reaction: list[float],
    game_ts: float,
) -> bool:
    """Propagate a speaker's introduction to one listener, reciprocally.

    Writes the listener's hearsay memory and the speaker's telling memory
    (deterministic IDs, provenance payloads), records a symbolic identity fact
    known by both, and clears the stranger ledger in both directions. Returns
    True once propagated.
    """
    descriptor = identity_descriptor(speaker_kernel)

    # Listener: a provenance-bearing hearsay memory about the speaker.
    await writer.write_raw_event(
        agent_id=listener,
        content=hearsay_content(speaker, descriptor),
        semantic_vector=identity_semantic_vec,
        emotional_vector=listener_reaction,
        game_ts=game_ts,
        event_type="hearsay",
        referents=[speaker],
        extra_payload={"provenance": "told", "source": speaker, "subject": speaker},
        point_id=hearsay_id(listener, speaker),
    )

    # Speaker: the reciprocal telling memory (the listener now knows them).
    await writer.write_raw_event(
        agent_id=speaker,
        content=telling_content(listener),
        semantic_vector=identity_semantic_vec,
        emotional_vector=speaker_reaction,
        game_ts=game_ts,
        event_type="disclosure",
        referents=[listener],
        extra_payload={"provenance": "disclosed", "told": listener, "subject": speaker},
        point_id=telling_id(speaker, listener),
    )

    # Symbolic identity fact (known by both) so are_acquainted is true without
    # a retrieval round-trip. Deterministic fact_id keeps it idempotent.
    if fact_pool is not None:
        fact_pool.add_fact(
            content=fact_content(speaker, descriptor),
            category="identity",
            game_ts=game_ts,
            knower_ids=[speaker, listener],
            fact_id=identity_fact_id(speaker, listener),
        )

    # Close the loop both ways — the exchange leaves a memory in both minds.
    acquaintance.clear_stranger(listener, speaker)
    acquaintance.clear_stranger(speaker, listener)

    logger.debug("Disclosure: %s -> %s (descriptor=%r)", speaker, listener, descriptor)
    return True
