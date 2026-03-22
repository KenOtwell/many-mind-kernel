"""
Stub Progeny service for Falcon development.

Returns canned TurnResponses without any cognitive processing.
Run: uvicorn scripts.stub_progeny:app --port 8001
"""
from __future__ import annotations

import logging
import time
from uuid import uuid4

from fastapi import FastAPI

from shared.schemas import (
    TickPackage, TurnResponse, AckResponse,
    AgentResponse, ActorValueDeltas, ActionCommand,
    ExtractionLevel,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [stub] %(message)s")
logger = logging.getLogger("stub_progeny")

app = FastAPI(title="Stub Progeny", version="0.0.1")


# ---------------------------------------------------------------------------
# Canned NPC responses — enough personality to eyeball-test the pipeline
# ---------------------------------------------------------------------------

def _canned(agent_id: str) -> AgentResponse:
    """Generate a canned response for a known or unknown NPC."""
    responses = {
        "Lydia": AgentResponse(
            agent_id="Lydia",
            utterance="I am sworn to carry your burdens.",
            actor_value_deltas=ActorValueDeltas(Confidence=3, Mood=0, Assistance=2),
            actions=[ActionCommand(command="Follow", target="Player")],
            extraction_level=ExtractionLevel.STRICT,
        ),
        "Belethor": AgentResponse(
            agent_id="Belethor",
            utterance="Do come back.",
            actor_value_deltas=ActorValueDeltas(Mood=3),
            extraction_level=ExtractionLevel.STRICT,
        ),
        "Ysolda": AgentResponse(
            agent_id="Ysolda",
            utterance="I've been thinking about trading with the Khajiit caravans.",
            actor_value_deltas=ActorValueDeltas(Mood=3, Confidence=2),
            extraction_level=ExtractionLevel.STRICT,
        ),
        "Heimskr": AgentResponse(
            agent_id="Heimskr",
            actor_value_deltas=ActorValueDeltas(Mood=1),
            extraction_level=ExtractionLevel.STRICT,
        ),
        "Adrianne": AgentResponse(
            agent_id="Adrianne",
            utterance="Looking to protect yourself, or deal some damage?",
            actor_value_deltas=ActorValueDeltas(Mood=3, Confidence=2),
            extraction_level=ExtractionLevel.STRICT,
        ),
    }
    if agent_id in responses:
        return responses[agent_id]
    # Unknown NPC — minimal stub
    return AgentResponse(
        agent_id=agent_id,
        utterance="...",
        actor_value_deltas=ActorValueDeltas(Mood=0),
        extraction_level=ExtractionLevel.STRICT,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.post("/ingest")
async def ingest(package: TickPackage) -> TurnResponse | AckResponse:
    """Mock Progeny ingest — accumulate or return canned turn response."""
    if not package.has_turn_trigger:
        logger.info("Accumulated tick (%d events)", len(package.events))
        return AckResponse(tick_id=package.tick_id)

    start = time.monotonic()
    logger.info("Turn trigger: %d NPCs, %d events",
                len(package.active_npc_ids), len(package.events))

    # Build canned responses for each active NPC; fall back if registry is empty
    npc_ids = package.active_npc_ids or ["Unknown"]
    responses = [_canned(npc_id) for npc_id in npc_ids]

    elapsed_ms = int((time.monotonic() - start) * 1000)
    logger.info("Returning %d canned responses (%dms)", len(responses), elapsed_ms)

    return TurnResponse(
        tick_id=package.tick_id,
        turn_id=uuid4(),
        responses=responses,
        processing_time_ms=elapsed_ms,
        model_used="stub-canned",
    )


@app.get("/health")
async def health():
    return {"status": "ok", "service": "stub_progeny", "model": "canned"}
