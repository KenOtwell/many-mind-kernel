"""
Falcon API routes.

GET|POST /comm.php — SKSE compatibility endpoint (the main interface)
GET      /health   — service health check
"""
from __future__ import annotations

import logging
from collections import deque
from datetime import datetime, timezone
from typing import Optional

import base64

from fastapi import APIRouter, Request, Response

from falcon.src.wire_protocol import parse_event, format_turn_response
from falcon.src import progeny_protocol
from falcon.src.event_parsers import parse_typed_data
from falcon.src.tick_accumulator import TickAccumulator
from shared.schemas import TypedEvent, TickPackage, TurnResponse
from shared.constants import COLLECTION_NPC_MEMORIES
from shared.config import settings
from shared import qdrant_wrapper

logger = logging.getLogger(__name__)

router = APIRouter()


# ---------------------------------------------------------------------------
# Module-level state
# ---------------------------------------------------------------------------

# SKSE polls with 'request' events. We queue Progeny results here.
# Lost on restart (acceptable — next turn regenerates).
# Bounded: if SKSE stops polling, don't let memory grow unbounded.
_response_queue: deque[str] = deque(maxlen=64)

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

@router.get("/health")
async def health():
    """Health check — Falcon service status. Registered before catch-all."""
    active_npcs = _tick_accumulator.get_active_npc_count() if _tick_accumulator else 0
    return {
        "status": "ok",
        "service": "falcon",
        "progeny_ws": settings.progeny.ws_url,
        "ws_connected": progeny_protocol._ws is not None,
        "queue_depth": len(_response_queue),
        "active_npcs": active_npcs,
        "tick_interval_s": settings.falcon.tick_interval_seconds,
    }


@router.api_route("/comm.php", methods=["GET", "POST"])
async def comm_endpoint(request: Request) -> Response:
    """
    SKSE compatibility endpoint.

    The DLL sends base64-encoded data as GET query params:
        GET /comm.php?DATA=<base64>&profile=<name>
    Also accepts POST for forward compatibility and testing.
    """
    return await _handle_skse_request(request)


@router.api_route("/{path:path}", methods=["GET", "POST"])
async def comm_endpoint_catchall(request: Request) -> Response:
    """Catch-all for all other CHIM DLL paths (gamedata.php, streamv2.php, etc.)."""
    return await _handle_skse_request(request)


async def _handle_skse_request(request: Request) -> Response:
    """Shared SKSE request handler for all comm endpoints."""
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
        # Forward session events to tick accumulator so Progeny gets
        # visibility into session boundaries (rollback, diary, Dragon Break).
        if _tick_accumulator is not None:
            event = TypedEvent(
                event_type=parsed.event_type,
                local_ts=datetime.now(timezone.utc).isoformat(),
                game_ts=parsed.game_ts,
                raw_data=parsed.data,
                parsed_data=None,
                is_turn_trigger=False,
            )
            await _tick_accumulator.push(event)
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



# ---------------------------------------------------------------------------
# Tick processing
# ---------------------------------------------------------------------------

async def _process_tick(package: TickPackage) -> None:
    """Send TickPackage to Progeny via WebSocket. Non-blocking, fire-and-forget.

    Response handling happens asynchronously in _handle_turn_response(),
    called by the WebSocket receive loop when Progeny delivers results.
    The tick loop never stalls.
    """
    try:
        await progeny_protocol.send_tick(package)
    except Exception:
        logger.exception("Failed to send tick package")


async def _handle_turn_response(turn: TurnResponse) -> None:
    """Callback for WebSocket receive loop: process a completed TurnResponse.

    Resolves utterance_key references to inline text via Qdrant, formats
    to CHIM wire protocol, and enqueues for SKSE polling.
    """
    if not turn.responses:
        logger.debug("Empty turn response from Progeny")
        return

    try:
        await _resolve_utterance_keys(turn)

        wire_output = format_turn_response(
            [r.model_dump() for r in turn.responses]
        )
        if wire_output:
            _response_queue.append(wire_output)
            logger.info("Turn response queued (%d agents, %d chars)",
                        len(turn.responses), len(wire_output))
    except Exception:
        logger.exception("Failed to handle turn response")


async def _resolve_utterance_keys(turn: TurnResponse) -> None:
    """Resolve utterance_key → inline utterance text via Qdrant read.

    For each AgentResponse that has utterance_key but no utterance,
    read the text from Qdrant and set it inline. Wire formatting
    only sees utterance text — keys are transparent at this layer.
    """
    from falcon.api.server import qdrant_client
    if qdrant_client is None:
        return

    for resp in turn.responses:
        if resp.utterance_key and not resp.utterance:
            text = await qdrant_wrapper.read_text(
                qdrant_client, COLLECTION_NPC_MEMORIES, resp.utterance_key,
            )
            if text:
                resp.utterance = text
            else:
                logger.warning(
                    "Failed to resolve utterance_key %s for %s",
                    resp.utterance_key, resp.agent_id,
                )


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


