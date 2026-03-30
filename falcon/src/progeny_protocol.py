"""
Falcon ↔ Progeny WebSocket client.

Persistent bidirectional channel: Falcon sends tick frames, Progeny
sends back ack/turn_response frames asynchronously. The tick loop
never blocks on Progeny — send_tick() is fire-and-forget.

Auto-reconnects with exponential backoff on disconnect. NPCs continue
on engine AI while disconnected (graceful degradation).
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Awaitable, Callable, Optional

import websockets
from websockets.asyncio.client import ClientConnection

from shared.schemas import TickPackage, TurnResponse
from shared.config import settings

logger = logging.getLogger(__name__)

# Module-level state
_ws: Optional[ClientConnection] = None
_receive_task: Optional[asyncio.Task] = None
_reconnect_task: Optional[asyncio.Task] = None
_on_turn_response: Optional[Callable[[TurnResponse], Awaitable[None]]] = None
_closing = False

# Reconnect parameters
_BASE_BACKOFF = 1.0
_MAX_BACKOFF = 30.0


async def connect(
    on_turn_response: Callable[[TurnResponse], Awaitable[None]],
) -> None:
    """Establish WebSocket to Progeny and start the receive loop.

    Args:
        on_turn_response: async callback invoked when Progeny delivers a
            completed TurnResponse. Falcon's routes.py provides this to
            resolve utterance keys, format wire output, and enqueue for SKSE.
    """
    global _on_turn_response, _closing
    _on_turn_response = on_turn_response
    _closing = False
    await _establish_connection()


async def _establish_connection() -> None:
    """Open the WebSocket and start the background receive loop."""
    global _ws, _receive_task

    url = settings.progeny.ws_url
    try:
        _ws = await websockets.connect(url, ping_interval=30, ping_timeout=60)
        logger.info("Connected to Progeny WebSocket at %s", url)
        _receive_task = asyncio.create_task(
            _receive_loop(), name="progeny_ws_receive"
        )
    except (OSError, websockets.exceptions.WebSocketException) as exc:
        logger.warning("Progeny WebSocket connect failed (%s) — will retry", exc)
        _ws = None
        _schedule_reconnect()


async def send_tick(package: TickPackage) -> None:
    """Send a tick frame to Progeny. Non-blocking, fire-and-forget.

    If the WebSocket is not connected, the tick is silently dropped
    (graceful degradation — NPCs continue on engine AI).
    """
    if _ws is None:
        logger.debug("Progeny not connected — dropping tick")
        return

    frame = json.dumps({
        "type": "tick",
        "data": package.model_dump(mode="json"),
    })
    try:
        await _ws.send(frame)
    except websockets.exceptions.ConnectionClosed:
        logger.warning("Progeny WebSocket closed during send — reconnecting")
        _schedule_reconnect()
    except Exception:
        logger.exception("Failed to send tick frame")


async def _receive_loop() -> None:
    """Background task: read frames from Progeny and dispatch."""
    global _ws
    assert _ws is not None

    try:
        async for raw in _ws:
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError:
                logger.warning("Invalid JSON from Progeny: %s", raw[:120])
                continue

            frame_type = frame.get("type")

            if frame_type == "turn_response":
                turn = TurnResponse.model_validate(frame["data"])
                if _on_turn_response:
                    try:
                        await _on_turn_response(turn)
                    except Exception:
                        logger.exception("Error in turn_response callback")

            elif frame_type == "ack":
                logger.debug(
                    "Progeny ack: tick %s",
                    frame.get("data", {}).get("tick_id"),
                )

            elif frame_type == "heartbeat":
                pass  # keepalive handled by websockets ping/pong

            else:
                logger.warning("Unknown frame type from Progeny: %s", frame_type)

    except websockets.exceptions.ConnectionClosed:
        logger.warning("Progeny WebSocket closed — reconnecting")
    except asyncio.CancelledError:
        return
    except Exception:
        logger.exception("Receive loop error")
    finally:
        _ws = None
        if not _closing:
            _schedule_reconnect()


def _schedule_reconnect() -> None:
    """Schedule a background reconnection attempt with exponential backoff."""
    global _reconnect_task
    if _closing:
        return
    if _reconnect_task and not _reconnect_task.done():
        return  # reconnect already scheduled
    _reconnect_task = asyncio.create_task(
        _reconnect_with_backoff(), name="progeny_ws_reconnect"
    )


async def _reconnect_with_backoff() -> None:
    """Attempt to reconnect with exponential backoff."""
    backoff = _BASE_BACKOFF
    while not _closing:
        logger.info("Reconnecting to Progeny in %.1fs...", backoff)
        await asyncio.sleep(backoff)
        try:
            await _establish_connection()
            if _ws is not None:
                return  # success
        except Exception:
            logger.debug("Reconnect attempt failed")
        backoff = min(backoff * 2, _MAX_BACKOFF)


async def close() -> None:
    """Shut down the WebSocket connection cleanly."""
    global _ws, _receive_task, _reconnect_task, _closing
    _closing = True

    if _reconnect_task and not _reconnect_task.done():
        _reconnect_task.cancel()
        try:
            await _reconnect_task
        except asyncio.CancelledError:
            pass

    if _receive_task and not _receive_task.done():
        _receive_task.cancel()
        try:
            await _receive_task
        except asyncio.CancelledError:
            pass

    if _ws is not None:
        await _ws.close()
        _ws = None

    logger.info("Progeny WebSocket client closed")
