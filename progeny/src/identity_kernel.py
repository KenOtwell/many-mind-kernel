"""
Identity kernel — per-NPC first-person identity loaded from seed profiles.

Phase 6a (Social Bootstrapping). On first sight of an NPC, Progeny reads the
NPC's seeded profile from `skyrim_npc_profiles` (see
`scripts/import_seed_data.py`) and parses it into an `IdentityKernel`: a
compact, partitioned view of who the NPC is.

Partition semantics:
  - The agent's OWN Tier-0/1 block may carry its full self-concept (public
    plus a little private self-knowledge). That block is private Layer-2, seen
    only by that agent's own mind (`self_clause`).
  - The PUBLIC half is the disclosure surface: what an NPC volunteers when
    introducing itself, and what may propagate to others as hearsay/lore in
    6e (`public_disclosure`). The PRIVATE half (full desire list, secrets,
    sensitive relationships, full backstory) never auto-propagates.

The kernel is also the class-signal source (occupation + tags) for the
valence-conditioned approach in 6c (`class_signal`).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)

# Personality-JSON fields that may be volunteered/propagated. Anything not
# listed is treated as private by default (privacy-first).
PUBLIC_PERSONALITY_FIELDS: frozenset[str] = frozenset({
    "name", "race", "gender", "occupation", "origin",
    "communicationStyle", "corePersonalityTraits",
})

_SLUG_DISALLOWED = re.compile(r"[^a-z0-9_'-]+")
_MULTI_UNDERSCORE = re.compile(r"_+")


def agent_id_to_slug(agent_id: str) -> str:
    """Normalize a runtime NPC name/id to the seed-data slug.

    Mirrors the slugs produced for `skyrim_npc_profiles`: lowercase, spaces to
    '_', hyphens/apostrophes preserved, disallowed characters dropped.
      'Ulfric Stormcloak'      -> 'ulfric_stormcloak'
      'Balgruuf the Greater'   -> 'balgruuf_the_greater'
      'Hjornskar Head-Smasher' -> 'hjornskar_head-smasher'
      'Balagog gro-Nolob'      -> 'balagog_gro-nolob'
      'Lydia'                  -> 'lydia'
    Returns '' when nothing usable remains (caller treats as no-profile).
    """
    s = (agent_id or "").strip().lower().replace(" ", "_")
    s = _SLUG_DISALLOWED.sub("", s)
    s = _MULTI_UNDERSCORE.sub("_", s).strip("_-")
    return s


@dataclass
class IdentityKernel:
    """Parsed, partitioned identity for one NPC."""
    agent_id: str
    slug: str
    public: dict[str, Any] = field(default_factory=dict)
    private: dict[str, Any] = field(default_factory=dict)
    bio_text: str = ""
    tags: list[str] = field(default_factory=list)
    voice_type: str = ""
    has_personality: bool = False

    def occupation(self) -> str:
        return str(self.public.get("occupation", "")).strip()

    def class_signal(self) -> str:
        """Short text signature of the NPC's kind (occupation + tags).

        Used by the 6c class probe so a percept query can resonate with
        memories about *others of that kind* before any shared history.
        """
        parts: list[str] = []
        occ = self.occupation()
        if occ:
            parts.append(occ)
        parts.extend(t for t in self.tags if t)
        return ", ".join(parts)

    def _tone(self) -> str:
        style = self.public.get("communicationStyle")
        if isinstance(style, dict):
            return str(style.get("tone", "")).strip()
        return str(style or "").strip()

    def _identity_head(self) -> dict[str, Any]:
        """Shared compact head (name/race/gender/occupation/origin + traits/tone)."""
        clause: dict[str, Any] = {}
        for key in ("name", "race", "gender", "occupation", "origin"):
            val = self.public.get(key)
            if val:
                clause[key] = val
        traits = self.public.get("corePersonalityTraits")
        if isinstance(traits, list) and traits:
            clause["traits"] = traits[:5]
        tone = self._tone()
        if tone:
            clause["tone"] = tone
        return clause

    def self_clause(self) -> dict[str, Any]:
        """Compact self-concept for the agent's OWN Tier-0/1 block.

        Private Layer-2 (only this agent's mind sees it), so it may include a
        little self-knowledge (a leading desire/values) beyond the public set.
        Token-lean: parsed fields, never the raw profile JSON.
        """
        clause = self._identity_head()
        desires = self.private.get("desires")
        if isinstance(desires, list) and desires:
            clause["desire"] = desires[0]
        values = self.private.get("coreValuesBeliefs")
        if isinstance(values, list) and values:
            clause["values"] = values[:3]
        return clause

    def public_disclosure(self) -> dict[str, Any]:
        """Disclosure-eligible fields (6e) — the public half only.

        The basis for hearsay/lore propagation when this NPC introduces
        itself. Never includes private self-knowledge.
        """
        return self._identity_head()


def parse_kernel(agent_id: str, payload: dict[str, Any]) -> IdentityKernel:
    """Parse a skyrim_npc_profiles payload into an IdentityKernel.

    Splits the personality JSON into public (disclosure-eligible) and private
    (self-only) halves by field name. Template-only NPCs (no personality JSON)
    yield a kernel with empty public/private and `has_personality=False` but
    still carry bio_text/tags/voice_type.
    """
    slug = str(payload.get("slug") or agent_id_to_slug(agent_id))
    personality = payload.get("personality")
    public: dict[str, Any] = {}
    private: dict[str, Any] = {}
    if isinstance(personality, dict):
        for key, val in personality.items():
            if key in PUBLIC_PERSONALITY_FIELDS:
                public[key] = val
            else:
                private[key] = val
    return IdentityKernel(
        agent_id=agent_id,
        slug=slug,
        public=public,
        private=private,
        bio_text=str(payload.get("bio_text") or ""),
        tags=list(payload.get("tags") or []),
        voice_type=str(payload.get("voice_type") or ""),
        has_personality=bool(payload.get("has_personality")),
    )


# ---------------------------------------------------------------------------
# Process-lifetime cache (sibling to _harmonic_state, _fact_pool, _goal_pool)
# ---------------------------------------------------------------------------

_cache: dict[str, IdentityKernel] = {}


def get(agent_id: str) -> IdentityKernel | None:
    """Return the cached kernel for an agent, or None if not loaded."""
    return _cache.get(agent_id)


def has(agent_id: str) -> bool:
    """True if an identity kernel is already cached for this agent."""
    return agent_id in _cache


def put(kernel: IdentityKernel) -> None:
    """Cache a kernel under its agent_id."""
    _cache[kernel.agent_id] = kernel


def clear() -> None:
    """Drop all cached kernels (e.g., on session wipe)."""
    _cache.clear()
