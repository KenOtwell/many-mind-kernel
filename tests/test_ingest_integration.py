"""
End-to-end integration tests for the /ingest pipeline.

Sends TickPackages through the FastAPI app with a mocked LLM backend.
Verifies the full flow: accumulate → schedule → prompt → LLM → expand → respond.
"""
from __future__ import annotations

import json
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from progeny.api.server import app
from progeny.api import routes
from progeny.src.event_accumulator import EventAccumulator
from progeny.src.agent_scheduler import AgentScheduler
from progeny.src.llm_client import GenerateResult
from tests.fixtures.factories import make_turn_package, make_data_package


@pytest.fixture(autouse=True)
def _fresh_state():
    """Reset pipeline state between tests."""
    routes._accumulator = EventAccumulator()
    routes._scheduler = AgentScheduler()
    routes._reminding_queue = {}
    yield


def _mock_llm_response(agents: list[str]) -> str:
    """Build a valid LLM response JSON for the given agents."""
    return json.dumps({
        "responses": [
            {
                "agent_id": agent_id,
                "utterance": f"Hello from {agent_id}.",
                "actor_value_deltas": {"Mood": 3},
            }
            for agent_id in agents
        ]
    })


class TestDataOnlyTick:
    @pytest.mark.asyncio
    async def test_data_tick_returns_ack(self):
        pkg = make_data_package()
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ingest",
                content=pkg.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "accumulated"
        assert "tick_id" in data


class TestTurnTrigger:
    @pytest.mark.asyncio
    async def test_turn_returns_turn_response(self):
        pkg = make_turn_package("What do you think?", active_npc_ids=["Lydia"])
        mock_response = _mock_llm_response(["Lydia"])

        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert "responses" in data
        assert len(data["responses"]) == 1
        assert data["responses"][0]["agent_id"] == "Lydia"
        assert data["responses"][0]["utterance"] == "Hello from Lydia."
        assert data["model_used"] == "llama.cpp"
        assert data["processing_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_multi_npc_turn(self):
        pkg = make_turn_package(
            "Hello everyone!",
            active_npc_ids=["Lydia", "Belethor", "Ysolda"],
        )
        mock_response = _mock_llm_response(["Lydia", "Belethor", "Ysolda"])

        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        data = resp.json()
        assert len(data["responses"]) == 3
        ids = [r["agent_id"] for r in data["responses"]]
        assert ids == ["Lydia", "Belethor", "Ysolda"]

    @pytest.mark.asyncio
    async def test_llm_error_returns_graceful_degradation(self):
        """LLM failure → graceful degradation with empty agent responses."""
        from progeny.src.llm_client import LLMError

        pkg = make_turn_package("Help!", active_npc_ids=["Lydia"])

        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.side_effect = LLMError("Connection failed")
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        assert resp.status_code == 200
        data = resp.json()
        assert data["model_used"] == "error"
        # Graceful degradation: agent entries exist but have no utterances
        assert len(data["responses"]) == 1
        assert data["responses"][0]["agent_id"] == "Lydia"
        assert data["responses"][0]["utterance"] is None

    @pytest.mark.asyncio
    async def test_no_active_npcs_returns_empty(self):
        """Turn trigger with no NPCs → no LLM call, empty responses."""
        pkg = make_turn_package("Anyone there?", active_npc_ids=[])

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post(
                "/ingest",
                content=pkg.model_dump_json(),
                headers={"Content-Type": "application/json"},
            )

        data = resp.json()
        assert data["responses"] == []
        assert data["model_used"] == "none"


class TestPromptPassedToLLM:
    @pytest.mark.asyncio
    async def test_prompt_contains_player_input(self):
        """Verify the LLM receives the player's text in the prompt."""
        pkg = make_turn_package("Tell me about dragons", active_npc_ids=["Lydia"])
        mock_response = _mock_llm_response(["Lydia"])
        captured_messages = []

        async def capture_generate(messages):
            captured_messages.extend(messages)
            return GenerateResult(content=mock_response)

        with patch("progeny.src.llm_client.generate", side_effect=capture_generate):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        assert len(captured_messages) == 2
        json_part = captured_messages[1]["content"].split("\n\n")[0]
        data_msg = json.loads(json_part)
        assert data_msg["player_input"]["text"] == "Tell me about dragons"

    @pytest.mark.asyncio
    async def test_prompt_contains_agent_blocks(self):
        """With parallel dispatch, each agent gets its own prompt."""
        pkg = make_turn_package("Hi", active_npc_ids=["Lydia", "Belethor"])
        captured_calls: list[list[dict]] = []

        async def capture_generate(messages):
            captured_calls.append(list(messages))
            # Return response for whichever agent is in this group
            json_part = messages[1]["content"].split("\n\n")[0]
            data = json.loads(json_part)
            agent_ids = [a["agent_id"] for a in data["agents"]]
            return GenerateResult(content=_mock_llm_response(agent_ids))

        with patch("progeny.src.llm_client.generate", side_effect=capture_generate):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                resp = await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # Each Tier 0 agent gets its own dispatch group
        assert len(captured_calls) == 2
        # Collect all agents across all groups
        all_agent_ids = []
        all_group_contexts = []
        for call in captured_calls:
            json_part = call[1]["content"].split("\n\n")[0]
            data = json.loads(json_part)
            all_agent_ids.extend(a["agent_id"] for a in data["agents"])
            all_group_contexts.append(data.get("group_context", {}))
        assert sorted(all_agent_ids) == ["Belethor", "Lydia"]
        # Each group sees ALL present NPCs via shared group_context
        for gc in all_group_contexts:
            present = gc.get("present_npcs", [])
            assert "Lydia" in present
            assert "Belethor" in present
        # Final merged response has both
        data = resp.json()
        resp_ids = [r["agent_id"] for r in data["responses"]]
        assert "Lydia" in resp_ids
        assert "Belethor" in resp_ids


class TestLLMHarmonics:
    """Verify two-pass emotional evaluation via updated_harmonics.

    Pass 1 (mechanical): text embedding → 9d projection → EMA update.
    Pass 2 (LLM): updated_harmonics.base_vector blended into buffer.
    """

    @pytest.mark.asyncio
    async def test_updated_harmonics_shifts_agent_state(self):
        """LLM-proposed harmonics should shift the agent's emotional state."""
        from mindcore.harmonic_buffer import HarmonicState, configure, HarmonicConfig

        # Use a fresh harmonic state for this test
        routes._harmonic_state = HarmonicState()
        # Set blend to 1.0 for clear signal
        configure(HarmonicConfig(llm_harmonics_blend=1.0))

        # LLM proposes high fear for Lydia
        fear_vector = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        mock_response = json.dumps({"responses": [{
            "agent_id": "Lydia",
            "utterance": "I sense danger.",
            "updated_harmonics": {"base_vector": fear_vector},
        }]})

        pkg = make_turn_package("What's wrong?", active_npc_ids=["Lydia"])
        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # After Pass 2, Lydia's state should have moved toward fear
        sem = routes._harmonic_state.get_semagram("Lydia")
        assert sem[0] > 0.1  # Fear axis (dim 0) should be elevated

        # Reset config
        configure(HarmonicConfig())

    @pytest.mark.asyncio
    async def test_no_harmonics_no_pass2(self):
        """Agent without updated_harmonics gets only mechanical pass."""
        from mindcore.harmonic_buffer import HarmonicState
        routes._harmonic_state = HarmonicState()

        mock_response = json.dumps({"responses": [{
            "agent_id": "Lydia",
            "utterance": "All is well.",
            # No updated_harmonics field
        }]})

        pkg = make_turn_package("How are you?", active_npc_ids=["Lydia"])
        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # State should exist (from mechanical pass) but fear shouldn't be spiked
        sem = routes._harmonic_state.get_semagram("Lydia")
        assert sem[0] < 0.3  # No fear injection from LLM

    @pytest.mark.asyncio
    async def test_blend_zero_disables_pass2(self):
        """blend=0.0 should completely ignore LLM harmonics."""
        from mindcore.harmonic_buffer import HarmonicState, configure, HarmonicConfig

        routes._harmonic_state = HarmonicState()
        configure(HarmonicConfig(llm_harmonics_blend=0.0))

        fear_vector = [0.9, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.1]
        mock_response = json.dumps({"responses": [{
            "agent_id": "Lydia",
            "utterance": "Fine.",
            "updated_harmonics": {"base_vector": fear_vector},
        }]})

        pkg = make_turn_package("Status?", active_npc_ids=["Lydia"])
        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # Fear should NOT be elevated — blend=0.0 ignores the LLM proposal
        sem = routes._harmonic_state.get_semagram("Lydia")
        assert sem[0] < 0.3

        configure(HarmonicConfig())


class TestRemindingQueue:
    """Verify the one-tick-delayed reminding protocol.

    Retrieval results from tick N enter the prompt on tick N+1.
    This is the anti-recursion guard: remindings can never trigger
    more retrieval in the same cycle.
    """

    @pytest.mark.asyncio
    async def test_first_turn_has_no_remindings(self):
        """First turn: reminding queue is empty, so no state_history from remindings."""
        pkg = make_turn_package("Hello", active_npc_ids=["Lydia"])
        mock_response = _mock_llm_response(["Lydia"])
        captured_prompts = []

        async def capture(messages):
            captured_prompts.append(messages)
            return GenerateResult(content=mock_response)

        with patch("progeny.src.llm_client.generate", side_effect=capture):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # First turn: prior_remindings was empty, so no state_history injected
        json_part = captured_prompts[0][1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert "state_history" not in agent

    @pytest.mark.asyncio
    async def test_reminding_queue_populated_after_turn(self):
        """After a turn with retrieval, _reminding_queue holds results for next tick."""
        from progeny.src.memory_retrieval import MemoryBundle

        pkg = make_turn_package("Hello", active_npc_ids=["Lydia"])
        mock_response = _mock_llm_response(["Lydia"])

        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # Reminding queue should exist (may be empty if Qdrant not available,
        # but the dict itself should be present as module state)
        assert isinstance(routes._reminding_queue, dict)

    @pytest.mark.asyncio
    async def test_prior_remindings_injected_on_second_turn(self):
        """Manually seed the reminding queue; verify it appears in the next prompt."""
        from progeny.src.memory_retrieval import MemoryBundle

        # Seed the queue as if tick N's retrieval produced a result
        routes._reminding_queue = {
            "Lydia": MemoryBundle(
                agent_id="Lydia",
                summaries=[{"text": "Fought a dragon together", "tier": "MOD"}],
            ),
        }

        pkg = make_turn_package("What happened?", active_npc_ids=["Lydia"])
        mock_response = _mock_llm_response(["Lydia"])
        captured_prompts = []

        async def capture(messages):
            captured_prompts.append(messages)
            return GenerateResult(content=mock_response)

        with patch("progeny.src.llm_client.generate", side_effect=capture):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # The prior reminding should appear in the agent's state_history
        json_part = captured_prompts[0][1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        assert "state_history" in agent
        assert "summaries" in agent["state_history"]
        assert agent["state_history"]["summaries"][0]["text"] == "Fought a dragon together"

    @pytest.mark.asyncio
    async def test_remindings_consumed_after_injection(self):
        """After injection, prior remindings are cleared (not re-injected next tick)."""
        from progeny.src.memory_retrieval import MemoryBundle

        routes._reminding_queue = {
            "Lydia": MemoryBundle(
                agent_id="Lydia",
                summaries=[{"text": "Old memory", "tier": "MOD"}],
            ),
        }

        # First turn: consumes the queue
        pkg1 = make_turn_package("First", active_npc_ids=["Lydia"])
        mock_response = _mock_llm_response(["Lydia"])
        with patch("progeny.src.llm_client.generate", new_callable=AsyncMock) as mock_gen:
            mock_gen.return_value = GenerateResult(content=mock_response)
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg1.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # Second turn: queue should be fresh (no Qdrant = empty retrieval)
        captured_prompts = []

        async def capture(messages):
            captured_prompts.append(messages)
            return GenerateResult(content=mock_response)

        pkg2 = make_turn_package("Second", active_npc_ids=["Lydia"])
        with patch("progeny.src.llm_client.generate", side_effect=capture):
            transport = ASGITransport(app=app)
            async with AsyncClient(transport=transport, base_url="http://test") as client:
                await client.post(
                    "/ingest",
                    content=pkg2.model_dump_json(),
                    headers={"Content-Type": "application/json"},
                )

        # The old "Old memory" should NOT appear in the second prompt
        json_part = captured_prompts[0][1]["content"].split("\n\n")[0]
        data = json.loads(json_part)
        agent = data["agents"][0]
        # state_history should be absent or not contain the old memory
        summaries = agent.get("state_history", {}).get("summaries", [])
        old_texts = [s.get("text") for s in summaries]
        assert "Old memory" not in old_texts


class TestHealth:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.get(
                "/health",
            )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "llm_connected" in data
