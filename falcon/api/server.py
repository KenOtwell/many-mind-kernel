"""
Falcon FastAPI application — THE server that replaces HerikaServer.

SKSE plugin (AIAgent.dll) POSTs here. We parse, process, forward to Progeny,
format the response, and return CHIM wire protocol.
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from falcon.api.routes import router, startup as routes_startup, shutdown as routes_shutdown
from falcon.src.progeny_protocol import close as close_progeny_client
from shared.config import settings

logging.basicConfig(
    level=getattr(logging, settings.log_level, logging.INFO),
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
)
logger = logging.getLogger("falcon")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup/shutdown lifecycle."""
    logger.info("Falcon starting — listening on %s:%d%s",
                settings.falcon.host, settings.falcon.port, settings.falcon.comm_path)
    logger.info("Progeny endpoint: %s/ingest", settings.progeny.base_url)
    await routes_startup()
    yield
    await routes_shutdown()
    await close_progeny_client()
    logger.info("Falcon shut down")


app = FastAPI(
    title="Falcon — Many-Mind Kernel",
    description="SKSE-compatible HerikaServer replacement",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)
