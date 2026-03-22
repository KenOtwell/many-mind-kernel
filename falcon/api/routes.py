"""
Falcon API routes.

POST /comm.php — SKSE compatibility endpoint (the main interface)
GET  /health   — service health check
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import base64

from fastapi import APIRouter, Request, Response

from falcon.src.wire_protocol import parse_event, format_turn_response
from falcon.src.progeny_protocol import send_package
from falcon.src.event_parsers import parse_typed_data
from falcon.src.tick_accumulator import TickAccumulator
from shared.schemas import TypedEvent, TickPackage, TurnResponse
from shared.config import settings

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# SKSE polls with 'request' events. We queue Progeny results here.
# Lost on restart (acceptable — next turn regenerates).
_response_queue: deque[str] = deque()

# Tick accumulator: initialised in startup(), torn down in shutdown().
_tick_accumulator: Optional[TickAccumulator] = None


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------

async def startup() -> None:
    """Initialise the tick accumulator (called from lifespan in server.py)."""
    global _tick_accumulator
    _tick_accumulator = TickAccumulator(
        interval_seconds=settings.falcon.tick_interval_seconds,
        on_package=_process_tick,
    )
    await _tick_accumulator.start()


async def shutdown() -> None:
    """Shut down the tick accumulator (called from lifespan in server.py)."""
    if _tick_accumulator is not None:
        await _tick_accumulator.stop()


# ---------------------------------------------------------------------------
# SKSE compatibility endpoint
# ---------------------------------------------------------------------------

@router.post("/comm.php")
@router.post("/{path:path}")  # Catch configurable AIAgent.ini paths
async def comm_endpoint(request: Request) -> Response:
    """
    SKSE compatibility endpoint.

    Accepts base64-encoded pipe-delimited wire format in query string
    (?DATA=<base64>&profile=<name>), returns CHIM wire format.
    This is the entry point that replaces HerikaServer's comm.php.
    """
    raw_body = await _decode_skse_request(request)
    parsed = parse_event(raw_body)

    if parsed is None:
        return _empty()

    logger.debug("SKSE event: type=%s game_ts=%.1f trigger=%s local=%s",
                 parsed.event_type, parsed.game_ts,
                 parsed.is_turn_trigger, parsed.is_local)

    # --- SKSE polling for response ---
    if parsed.event_type == "request":
        return _dequeue_response()

    # --- Chat target not found (Falcon-local) ---
    if parsed.event_type == "chatnf":
        logger.warning("SKSE: NPC not found — %s", parsed.data[:80])
        return _empty()

    # --- Just-say: queue verbatim wire output for SKSE polling ---
    if parsed.event_type == "just_say":
        _response_queue.append(parsed.data)
        return _empty()

    # --- Session lifecycle ---
    if parsed.is_session:
        if parsed.event_type in ("init", "wipe", "playerdied"):
            if _tick_accumulator is not None:
                await _tick_accumulator.clear_npcs()
            logger.info("Session reset (%s) — NPC registry cleared", parsed.event_type)
        else:
            logger.info("Session signal: %s", parsed.event_type)
        return _empty()

    # --- All other events: decode structure, push to tick accumulator ---
    if _tick_accumulator is not None:
        event = TypedEvent(
            event_type=parsed.event_type,
            local_ts=datetime.now(timezone.utc).isoformat(),
            game_ts=parsed.game_ts,
            raw_data=parsed.data,
            parsed_data=parse_typed_data(parsed.event_type, parsed.data),
            is_turn_trigger=parsed.is_turn_trigger,
        )
        await _tick_accumulator.push(event)
    else:
        logger.warning("TickAccumulator not initialised — dropping %s event",
                       parsed.event_type)

    return _empty()


# ---------------------------------------------------------------------------
# SKSE request decoding
# ---------------------------------------------------------------------------

async def _decode_skse_request(request: Request) -> str:
    """
    Decode SKSE wire data from the HTTP request.

    The DLL sends data as base64 in the query string:
        GET/POST ...?DATA=<base64>&profile=<name>

    Falls back to raw POST body for testing and forward compatibility.
    """
    # Primary: base64-encoded DATA in query string (what the DLL sends)
    data_param = request.query_params.get("DATA") or request.query_params.get("data")
    if data_param:
        try:
            decoded = base64.b64decode(data_param).decode("utf-8", errors="replace")
            logger.debug("Decoded base64 DATA from query string (%d bytes)", len(decoded))
            return decoded
        except Exception as exc:
            logger.warning("Failed to base64-decode DATA param: %s", exc)

    # Fallback: check if the full query string is DATA=<base64> without proper parsing
    # (some HTTP clients mangle query params with + chars in base64)
    raw_qs = str(request.url.query)
    if raw_qs.upper().startswith("DATA="):
        # Strip "DATA=" prefix, take up to first & if present
        b64_part = raw_qs[5:]
        amp_pos = b64_part.find("&")
        if amp_pos >= 0:
            b64_part = b64_part[:amp_pos]
        try:
            decoded = base64.b64decode(b64_part).decode("utf-8", errors="replace")
            logger.debug("Decoded base64 from raw query string (%d bytes)", len(decoded))
            return decoded
        except Exception as exc:
            logger.warning("Failed to base64-decode raw query string: %s", exc)

    # Fallback: raw POST body (for testing and forward compatibility)
    body = (await request.body()).decode("utf-8", errors="replace")
    if body:
        logger.debug("Using raw POST body (%d bytes)", len(body))
    return body


def _get_profile(request: Request) -> str:
    """Extract the CHIM profile name from the request query string."""
    return request.query_params.get("profile", "")


# ---------------------------------------------------------------------------
# Tick processing
# ---------------------------------------------------------------------------

async def _process_tick(package: TickPackage) -> None:
    """Send TickPackage to Progeny and queue any TurnResponse for SKSE polling."""
    try:
        result = await send_package(package)

        if result and isinstance(result, TurnResponse) and result.responses:
            wire_output = format_turn_response(
                [r.model_dump() for r in result.responses]
            )
            if wire_output:
                _response_queue.append(wire_output)
                logger.info("Turn response queued (%d agents, %d chars)",
                            len(result.responses), len(wire_output))
        else:
            logger.debug("No turn response from Progeny (or empty)")

    except Exception:
        logger.exception("Failed to process tick package")


# ---------------------------------------------------------------------------
# Response queue management
# ---------------------------------------------------------------------------

def _dequeue_response() -> Response:
    """Dequeue the next pending response for SKSE, or return empty."""
    if _response_queue:
        wire_text = _response_queue.popleft()
        logger.debug("Dequeuing response (%d chars, %d remaining)",
                     len(wire_text), len(_response_queue))
        return Response(content=wire_text, media_type="text/plain")
    return _empty()


def _empty() -> Response:
    """Return empty response (SKSE interprets as 'nothing to do')."""
    return Response(content="", media_type="text/plain")


# ---------------------------------------------------------------------------
# Health / debug
# ---------------------------------------------------------------------------

@router.get("/health")
async def health():
    """Health check — Falcon service status."""
    active_npcs = _tick_accumulator.get_active_npc_count() if _tick_accumulator else 0
    return {
        "status": "ok",
        "service": "falcon",
        "progeny_url": settings.progeny.base_url,
        "queue_depth": len(_response_queue),
        "active_npcs": active_npcs,
        "tick_interval_s": settings.falcon.tick_interval_seconds,
    }
