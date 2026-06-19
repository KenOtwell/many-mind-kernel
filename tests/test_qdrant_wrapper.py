"""Tests for shared.qdrant_wrapper — the enrichment gate.

Uses a mocked AsyncQdrantClient (no live Qdrant needed). The embedding
model and emotional bases are real — tests validate the full enrichment
pipeline from text to stored vectors.
"""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from mindcore import qdrant_wrapper
from shared.constants import COLLECTION_NPC_MEMORIES, EMOTIONAL_DIM, SEMANTIC_DIM


@pytest.fixture
def mock_client():
    """An AsyncQdrantClient mock that records upsert/retrieve calls."""
    client = AsyncMock()
    # upsert succeeds silently
    client.upsert = AsyncMock()
    # retrieve returns empty by default (overridden per-test)
    client.retrieve = AsyncMock(return_value=[])
    return client


@pytest.fixture(autouse=True)
def _ensure_models_loaded():
    """Load real embedding model + emotional bases for the test session."""
    from mindcore import embedding, emotional
    embedding.load_model()
    emotional.load_bases()


# ---------------------------------------------------------------------------
# ingest()
# ---------------------------------------------------------------------------

class TestIngest:
    @pytest.mark.asyncio
    async def test_returns_point_id(self, mock_client):
        """ingest() should return a UUID string on success."""
        key = await qdrant_wrapper.ingest(
            client=mock_client,
            text="I am sworn to carry your burdens.",
            collection=COLLECTION_NPC_MEMORIES,
            agent_id="Lydia",
            game_ts=100.0,
        )
        assert key is not None
        assert isinstance(key, str)
        assert len(key) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_writes_dual_vectors(self, mock_client):
        """The upserted point should have semantic (384d) + emotional (9d) vectors."""
        await qdrant_wrapper.ingest(
            client=mock_client,
            text="Something's not right. Stay behind me.",
            collection=COLLECTION_NPC_MEMORIES,
            agent_id="Lydia",
            game_ts=100.0,
            event_type="speech",
        )
        mock_client.upsert.assert_called_once()
        call_kwargs = mock_client.upsert.call_args
        points = call_kwargs.kwargs.get("points") or call_kwargs[1].get("points")
        assert len(points) == 1

        point = points[0]
        semantic_vec = point.vector["semantic"]
        emotional_vec = point.vector["emotional"]
        assert len(semantic_vec) == SEMANTIC_DIM
        assert len(emotional_vec) == EMOTIONAL_DIM

    @pytest.mark.asyncio
    async def test_payload_contains_text_and_metadata(self, mock_client):
        """Stored payload should contain original text plus metadata."""
        await qdrant_wrapper.ingest(
            client=mock_client,
            text="Do come back.",
            collection=COLLECTION_NPC_MEMORIES,
            agent_id="Belethor",
            game_ts=200.0,
            event_type="dialogue",
            extra_payload={"location": "Whiterun"},
        )
        point = mock_client.upsert.call_args.kwargs["points"][0]
        payload = point.payload
        assert payload["text"] == "Do come back."
        assert payload["agent_id"] == "Belethor"
        assert payload["event_type"] == "dialogue"
        assert payload["game_ts"] == 200.0
        assert payload["location"] == "Whiterun"
        assert "wall_ts" in payload

    @pytest.mark.asyncio
    async def test_empty_text_returns_none(self, mock_client):
        """Empty or whitespace-only text should return None without writing."""
        for text in ["", "   ", None]:
            key = await qdrant_wrapper.ingest(
                client=mock_client,
                text=text or "",
                collection=COLLECTION_NPC_MEMORIES,
                agent_id="Nobody",
                game_ts=0.0,
            )
            assert key is None
        mock_client.upsert.assert_not_called()

    @pytest.mark.asyncio
    async def test_qdrant_failure_returns_none(self, mock_client):
        """If Qdrant upsert fails, ingest() returns None (no crash)."""
        mock_client.upsert.side_effect = Exception("connection refused")
        key = await qdrant_wrapper.ingest(
            client=mock_client,
            text="This should fail gracefully.",
            collection=COLLECTION_NPC_MEMORIES,
            agent_id="Test",
            game_ts=0.0,
        )
        assert key is None

    @pytest.mark.asyncio
    async def test_emotional_vector_is_valid_semagram(self, mock_client):
        """The emotional vector should be a valid 9d semagram with non-negative residual."""
        await qdrant_wrapper.ingest(
            client=mock_client,
            text="I'm terrified of that dragon!",
            collection=COLLECTION_NPC_MEMORIES,
            agent_id="Guard",
            game_ts=50.0,
        )
        point = mock_client.upsert.call_args.kwargs["points"][0]
        emotional_vec = point.vector["emotional"]
        # Residual (axis 8) should be non-negative
        assert emotional_vec[8] >= 0.0
        # Fear (axis 0) should be notably positive for "terrified"
        assert emotional_vec[0] > 0.1


# ---------------------------------------------------------------------------
# read_text()
# ---------------------------------------------------------------------------

class TestReadText:
    @pytest.mark.asyncio
    async def test_returns_stored_text(self, mock_client):
        """read_text() should return the text from the point payload."""
        mock_point = MagicMock()
        mock_point.payload = {"text": "Stay behind me.", "agent_id": "Lydia"}
        mock_client.retrieve = AsyncMock(return_value=[mock_point])

        text = await qdrant_wrapper.read_text(
            mock_client, COLLECTION_NPC_MEMORIES, "some-uuid",
        )
        assert text == "Stay behind me."

    @pytest.mark.asyncio
    async def test_returns_none_for_missing_point(self, mock_client):
        """read_text() returns None if the point doesn't exist."""
        mock_client.retrieve = AsyncMock(return_value=[])
        text = await qdrant_wrapper.read_text(
            mock_client, COLLECTION_NPC_MEMORIES, "nonexistent-uuid",
        )
        assert text is None

    @pytest.mark.asyncio
    async def test_returns_none_on_qdrant_error(self, mock_client):
        """read_text() returns None if Qdrant is unreachable."""
        mock_client.retrieve = AsyncMock(side_effect=Exception("timeout"))
        text = await qdrant_wrapper.read_text(
            mock_client, COLLECTION_NPC_MEMORIES, "some-uuid",
        )
        assert text is None
