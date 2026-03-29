"""
Falcon FastAPI application — THE server that replaces HerikaServer.

SKSE plugin (AIAgent.dll) POSTs here. We parse, process, forward to Progeny,
format the response, and return CHIM wire protocol.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from qdrant_client import AsyncQdrantClient

from falcon.api.routes import (
    router, startup as routes_startup, shutdown as routes_shutdown,
    _handle_turn_response,
)
from falcon.src import progeny_protocol
from shared import embedding, emotional
from shared.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("falcon")


# Module-level Qdrant client — initialized in lifespan, used by routes.
qdrant_client: AsyncQdrantClient | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    global qdrant_client
    logger.info("Falcon starting — listening on %s:%d%s",
                settings.falcon.host, settings.falcon.port, settings.falcon.comm_path)
    logger.info("Progeny WebSocket: %s", settings.progeny.ws_url)
    # Load enrichment pipeline (CPU, ~2s total)
    embedding.load_model()
    emotional.load_bases()
    logger.info("Enrichment pipeline ready (embedding + emotional projection)")
    # Connect to local Qdrant
    qdrant_client = AsyncQdrantClient(
        host=settings.qdrant.host, port=settings.qdrant.rest_port,
    )
    logger.info("Qdrant client initialized → %s:%d",
                settings.qdrant.host, settings.qdrant.rest_port)
    await routes_startup()
    # Connect to Progeny via WebSocket (auto-reconnects on failure)
    await progeny_protocol.connect(on_turn_response=_handle_turn_response)
    yield
    await routes_shutdown()
    await progeny_protocol.close()
    if qdrant_client:
        await qdrant_client.close()
    logger.info("Falcon shut down")


app = FastAPI(
    title="Falcon — Many-Mind Kernel",
    description="SKSE-compatible HerikaServer replacement",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
