"""
Progeny FastAPI application.

The growing mind — ALL cognitive work happens here.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from progeny.api.routes import router
from progeny.src import llm_client
from progeny.src import embedding
from progeny.src import emotional_projection
from progeny.src import client as qdrant_client
from progeny.src.prompt_formatter import SYSTEM_PROMPT
from shared.config import settings

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)


async def _warm_kv_cache() -> None:
    """
    Prime the LLM server's KV cache with the system prompt.

    On the first real request, the system prompt prefix will already be
    cached — skipping ~200 tokens of prompt eval. Only works when the
    llama-server uses a single slot (-np 1) so subsequent requests hit
    the same cache.
    """
    if not await llm_client.health_check():
        logger.warning("LLM server not reachable — skipping cache warm")
        return

    try:
        # Send a minimal request with just the system prompt.
        # The LLM will cache the system prompt tokens in the KV cache.
        # max_tokens=1 ensures we don't waste generation time.
        result = await llm_client.generate([
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Respond with: {}"},
        ])
        logger.info(
            "KV cache warmed: %d prompt tokens cached (%.0fms)",
            result.prompt_tokens, result.prompt_ms,
        )
    except Exception as exc:
        logger.warning("Cache warm failed (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info(
        "Progeny starting — model profile: %s (temp=%.1f, strict_alt=%s)",
        settings.model.name, settings.model.temperature,
        settings.model.strict_alternation,
    )
    # Load emotional intelligence pipeline (CPU, ~2s)
    embedding.load_model()
    emotional_projection.load_bases()
    logger.info("Emotional pipeline ready")
    # Connect to Qdrant and create collections if needed
    await qdrant_client.init()
    await qdrant_client.ensure_collections()
    qdrant_ok = await qdrant_client.health_check()
    logger.info("Qdrant: %s", "connected" if qdrant_ok else "unreachable (will retry per-op)")
    await _warm_kv_cache()
    yield
    logger.info("Progeny shutting down")


app = FastAPI(
    title="Progeny — Many-Mind Kernel",
    description="Stateful mind engine for Skyrim NPC companions.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
