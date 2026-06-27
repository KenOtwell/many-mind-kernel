"""Tests for Phase 6a — identity kernel bootstrap.

Covers slug normalization, profile parsing + public/private partition,
`read_profile` against in-memory Qdrant, and the Tier-0/1-only injection of
the compact identity clause into the agent block.

Hermetic: in-memory Qdrant + hand-made payloads. No model download.
"""
from __future__ import annotations

import pytest
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import PointStruct

import progeny.src.qdrant_client as client_mod
from progeny.src.qdrant_client import _npc_profile_point_id, ensure_collections, read_profile
from progeny.src import identity_kernel as ik
from progeny.src.prompt_formatter import _build_agent_block
from progeny.src.agent_scheduler import ScheduledAgent
from progeny.src.event_accumulator import TurnContext
from shared.constants import COLLECTION_NPC_PROFILES, SEMANTIC_DIM


LYDIA_PERSONALITY = {
    "name": "Lydia",
    "race": "Nord",
    "gender": "female",
    "occupation": "Housecarl",
    "origin": "Whiterun",
    "desires": ["To serve and protect the Dragonborn"],
    "needsRequests": ["To follow the Dragonborn on adventures"],
    "backgroundSummary": "Born and raised in Whiterun, sworn to her Thane.",
    "coreValuesBeliefs": ["Honor above all", "Loyalty to hold"],
    "communicationStyle": {"tone": "formal", "mannerisms": "uses honorifics"},
    "corePersonalityTraits": ["Loyal", "Dutiful", "Brave"],
    "relationships": [{"name": "Balgruuf the Greater", "description": "liege lord"}],
}


@pytest.fixture
async def qdrant():
    mem = AsyncQdrantClient(location=":memory:")
    client_mod.configure(mem)
    await ensure_collections()
    yield mem
    client_mod.configure(None)


async def _seed_profile(
    slug: str,
    personality: dict | None = None,
    bio: str = "",
    tags: list[str] | None = None,
    voice: str = "",
    has_personality: bool = True,
) -> None:
    payload: dict = {
        "slug": slug,
        "bio_text": bio,
        "tags": tags or [],
        "voice_type": voice,
        "has_personality": has_personality,
    }
    if personality is not None:
        payload["personality"] = personality
    await client_mod.get_client().upsert(
        collection_name=COLLECTION_NPC_PROFILES,
        points=[PointStruct(
            id=_npc_profile_point_id(slug),
            vector={"semantic": [0.1] * SEMANTIC_DIM},
            payload=payload,
        )],
    )


def _ctx() -> TurnContext:
    return TurnContext(
        player_input="",
        agent_buffers={},
        active_npc_ids=["Lydia"],
        world_events=[],
        session_events=[],
    )


# ---------------------------------------------------------------------------
# Slug normalization
# ---------------------------------------------------------------------------

class TestSlug:
    @pytest.mark.parametrize("name,expected", [
        ("Ulfric Stormcloak", "ulfric_stormcloak"),
        ("Balgruuf the Greater", "balgruuf_the_greater"),
        ("Hjornskar Head-Smasher", "hjornskar_head-smasher"),
        ("Balagog gro-Nolob", "balagog_gro-nolob"),
        ("Lydia", "lydia"),
        ("  Lydia  ", "lydia"),
    ])
    def test_known_slugs(self, name, expected):
        assert ik.agent_id_to_slug(name) == expected

    def test_empty_is_empty(self):
        assert ik.agent_id_to_slug("") == ""
        assert ik.agent_id_to_slug("   ") == ""


# ---------------------------------------------------------------------------
# parse_kernel + partition
# ---------------------------------------------------------------------------

class TestParseKernel:
    def test_public_private_partition(self):
        k = ik.parse_kernel("Lydia", {"slug": "lydia", "personality": LYDIA_PERSONALITY})
        # Public half: identity-ish fields.
        assert k.public.get("occupation") == "Housecarl"
        assert "corePersonalityTraits" in k.public
        # Private half: self-knowledge that must not auto-propagate.
        assert "desires" in k.private
        assert "backgroundSummary" in k.private
        assert "desires" not in k.public

    def test_self_clause_includes_self_knowledge(self):
        k = ik.parse_kernel("Lydia", {"slug": "lydia", "personality": LYDIA_PERSONALITY})
        sc = k.self_clause()
        assert sc["name"] == "Lydia"
        assert sc["occupation"] == "Housecarl"
        assert sc["tone"] == "formal"
        assert sc["desire"] == "To serve and protect the Dragonborn"
        assert sc["values"] == ["Honor above all", "Loyalty to hold"]

    def test_public_disclosure_excludes_private(self):
        k = ik.parse_kernel("Lydia", {"slug": "lydia", "personality": LYDIA_PERSONALITY})
        pd = k.public_disclosure()
        assert pd["occupation"] == "Housecarl"
        assert "desire" not in pd
        assert "values" not in pd
        assert "backgroundSummary" not in pd

    def test_class_signal_occupation_plus_tags(self):
        k = ik.parse_kernel(
            "Lydia",
            {"slug": "lydia", "tags": ["nord", "warrior"], "personality": LYDIA_PERSONALITY},
        )
        assert k.class_signal() == "Housecarl, nord, warrior"

    def test_template_only_no_personality(self):
        k = ik.parse_kernel(
            "Some Guard",
            {"slug": "some_guard", "bio_text": "A city guard.", "tags": ["guard"],
             "voice_type": "sk_male", "has_personality": False, "personality": None},
        )
        assert k.has_personality is False
        assert k.public == {} and k.private == {}
        assert k.self_clause() == {}
        # Class signal still works from tags alone (no occupation).
        assert k.class_signal() == "guard"


# ---------------------------------------------------------------------------
# read_profile (in-memory Qdrant)
# ---------------------------------------------------------------------------

class TestReadProfile:
    async def test_roundtrip(self, qdrant):
        await _seed_profile("lydia", personality=LYDIA_PERSONALITY, bio="Housecarl of Whiterun.")
        payload = await read_profile("lydia")
        assert payload is not None
        assert payload["slug"] == "lydia"
        assert payload["personality"]["occupation"] == "Housecarl"

    async def test_missing_returns_none(self, qdrant):
        assert await read_profile("definitely_not_seeded") is None

    async def test_empty_slug_returns_none(self, qdrant):
        assert await read_profile("") is None

    async def test_parse_from_read(self, qdrant):
        await _seed_profile("lydia", personality=LYDIA_PERSONALITY)
        payload = await read_profile("lydia")
        kernel = ik.parse_kernel("Lydia", payload)
        assert kernel.occupation() == "Housecarl"
        assert "desire" in kernel.self_clause()


# ---------------------------------------------------------------------------
# Tier-gated agent-block injection
# ---------------------------------------------------------------------------

class TestAgentBlockInjection:
    CLAUSE = {"name": "Lydia", "occupation": "Housecarl", "origin": "Whiterun"}

    def _block(self, tier: int, clause: dict | None):
        return _build_agent_block(
            ScheduledAgent(agent_id="Lydia", tier=tier, ticks_since_last_action=0),
            _ctx(),
            ["Lydia"],
            identity_clause=clause,
        )

    def test_identity_present_tier0(self):
        block = self._block(0, self.CLAUSE)
        assert block.get("identity") == self.CLAUSE

    def test_identity_present_tier1(self):
        block = self._block(1, self.CLAUSE)
        assert block.get("identity") == self.CLAUSE

    def test_identity_absent_tier2(self):
        block = self._block(2, self.CLAUSE)
        assert "identity" not in block

    def test_identity_absent_tier3(self):
        block = self._block(3, self.CLAUSE)
        assert "identity" not in block

    def test_no_clause_no_field(self):
        block = self._block(0, None)
        assert "identity" not in block
