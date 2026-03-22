"""
Round-trip integration tests: SKSE event → Falcon → stub Progeny → CHIM wire.

Uses FastAPI TestClient with stub_progeny mounted in-process (no network).
"""
import asyncio
import base64
import pytest
from unittest.mock import AsyncMock, patch

from httpx import AsyncClient, ASGITransport

from falcon.api.server import app as falcon_app
from scripts.stub_progeny import app as stub_app
from shared.schemas import TickPackage, TurnResponse, AckResponse, AgentResponse, ExtractionLevel
from tests.fixtures.factories import (
    WIRE_INPUTTEXT, WIRE_INFO, WIRE_REQUEST, WIRE_GOODNIGHT, WIRE_CHATNF,
    WIRE_MALFORMED_EMPTY, WIRE_DATA_WITH_PIPES,
    make_turn_package,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def anyio_backend():
    return "asyncio"


# ---------------------------------------------------------------------------
# Wire protocol round-trip (Falcon only, no Progeny)
# ---------------------------------------------------------------------------

class TestFalconEndpoint:
    """Test the /comm.php endpoint directly (without Progeny)."""

    @pytest.mark.anyio
    async def test_request_poll_returns_empty(self):
        """SKSE request poll with empty queue returns empty body."""
        async with AsyncClient(
            transport=ASGITransport(app=falcon_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/comm.php", content=WIRE_REQUEST)
            assert resp.status_code == 200
            assert resp.text == ""

    @pytest.mark.anyio
    async def test_malformed_event_returns_empty(self):
        async with AsyncClient(
            transport=ASGITransport(app=falcon_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/comm.php", content=WIRE_MALFORMED_EMPTY)
            assert resp.status_code == 200
            assert resp.text == ""

    @pytest.mark.anyio
    async def test_chatnf_returns_empty(self):
        async with AsyncClient(
            transport=ASGITransport(app=falcon_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/comm.php", content=WIRE_CHATNF)
            assert resp.status_code == 200
            assert resp.text == ""

    @pytest.mark.anyio
    async def test_goodnight_returns_empty(self):
        async with AsyncClient(
            transport=ASGITransport(app=falcon_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/comm.php", content=WIRE_GOODNIGHT)
            assert resp.status_code == 200
            assert resp.text == ""

    @pytest.mark.anyio
    async def test_health_endpoint(self):
        async with AsyncClient(
            transport=ASGITransport(app=falcon_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/health")
            assert resp.status_code == 200
            data = resp.json()
            assert data["service"] == "falcon"
            assert "queue_depth" in data
            assert "active_npcs" in data
            assert "tick_interval_s" in data

    @pytest.mark.anyio
    async def test_base64_query_string_decoding(self):
        """SKSE DLL sends base64-encoded data in the query string."""
        wire_data = "request|1710624010|54330.0|"
        b64_data = base64.b64encode(wire_data.encode()).decode()
        async with AsyncClient(
            transport=ASGITransport(app=falcon_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(f"/comm.php?DATA={b64_data}&profile=default")
            assert resp.status_code == 200
            assert resp.text == ""  # request poll with empty queue

    @pytest.mark.anyio
    async def test_base64_session_event(self):
        """Base64-encoded session events should be handled correctly."""
        wire_data = "goodnight|1710624012|54332.0|"
        b64_data = base64.b64encode(wire_data.encode()).decode()
        async with AsyncClient(
            transport=ASGITransport(app=falcon_app),
            base_url="http://test",
        ) as client:
            resp = await client.post(f"/comm.php?DATA={b64_data}")
            assert resp.status_code == 200
            assert resp.text == ""

    @pytest.mark.anyio
    async def test_raw_body_fallback_still_works(self):
        """Existing raw POST body path should still work for testing."""
        async with AsyncClient(
            transport=ASGITransport(app=falcon_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/comm.php", content=WIRE_REQUEST)
            assert resp.status_code == 200
            assert resp.text == ""


# ---------------------------------------------------------------------------
# Stub Progeny direct tests
# ---------------------------------------------------------------------------

class TestStubProgeny:
    """Test the stub Progeny service directly."""

    @pytest.mark.anyio
    async def test_turn_trigger_returns_responses(self):
        """Stub should return canned responses for turn triggers."""
        package = make_turn_package()
        async with AsyncClient(
            transport=ASGITransport(app=stub_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/ingest", json=package.model_dump(mode="json"))
            assert resp.status_code == 200
            data = resp.json()
            turn = TurnResponse.model_validate(data)
            assert len(turn.responses) == 1
            assert turn.responses[0].agent_id == "Lydia"
            assert turn.responses[0].utterance is not None
            assert turn.model_used == "stub-canned"

    @pytest.mark.anyio
    async def test_data_event_returns_ack(self):
        """Stub should return AckResponse for non-turn events."""
        from tests.fixtures.factories import make_data_package
        package = make_data_package()
        async with AsyncClient(
            transport=ASGITransport(app=stub_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/ingest", json=package.model_dump(mode="json"))
            assert resp.status_code == 200
            ack = AckResponse.model_validate(resp.json())
            assert ack.status == "accumulated"

    @pytest.mark.anyio
    async def test_multi_npc_returns_all(self):
        """Stub should return one response per NPC."""
        from tests.fixtures.factories import make_multi_npc_package
        package = make_multi_npc_package()
        async with AsyncClient(
            transport=ASGITransport(app=stub_app),
            base_url="http://test",
        ) as client:
            resp = await client.post("/ingest", json=package.model_dump(mode="json"))
            turn = TurnResponse.model_validate(resp.json())
            agent_ids = {r.agent_id for r in turn.responses}
            assert "Lydia" in agent_ids
            assert "Belethor" in agent_ids
            assert "Ysolda" in agent_ids
            assert "Heimskr" in agent_ids

    @pytest.mark.anyio
    async def test_stub_health(self):
        async with AsyncClient(
            transport=ASGITransport(app=stub_app),
            base_url="http://test",
        ) as client:
            resp = await client.get("/health")
            assert resp.json()["service"] == "stub_progeny"


# ---------------------------------------------------------------------------
# Schema validation tests
# ---------------------------------------------------------------------------

class TestSchemaValidation:
    """Test that Pydantic models validate correctly."""

    def test_tick_package_serializes(self):
        """TickPackage should round-trip through JSON."""
        package = make_turn_package()
        json_data = package.model_dump(mode="json")
        restored = TickPackage.model_validate(json_data)
        assert restored.events[0].event_type == "inputtext"
        assert restored.has_turn_trigger is True
        assert len(restored.active_npc_ids) == 1

    def test_actor_value_clamping(self):
        """Actor values outside range should fail validation."""
        from pydantic import ValidationError
        from shared.schemas import ActorValueDeltas
        with pytest.raises(ValidationError):
            ActorValueDeltas(Aggression=5)  # Max is 3

    def test_emotional_state_requires_9d(self):
        """EmotionalState vectors must be exactly 9 dimensions."""
        from pydantic import ValidationError
        from shared.schemas import EmotionalState
        with pytest.raises(ValidationError):
            EmotionalState(
                base_vector=[0.1, 0.2],  # Too short
                delta=[0.0] * 9,
            )
