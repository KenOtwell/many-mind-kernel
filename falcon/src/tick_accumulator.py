"""
Falcon tick-based event accumulator.

Buffers incoming TypedEvents and fires a TickPackage to a callback
on a configurable interval. Background asyncio task; HTTP handlers push
events via push(), which is awaited directly (fast lock + append).

active_npc_ids is derived from addnpc events and cleared on world-reset
session signals (init / wipe / playerdied).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Awaitable, Callable, Optional

from shared.schemas import TickPackage, TypedEvent

logger = logging.getLogger(__name__)


class TickAccumulator:
    """
    Accumulates TypedEvents and fires TickPackages on a tick cadence.

    start() / stop() are called from the FastAPI lifespan (server.py).
    push() is called from HTTP handler coroutines — awaiting it is safe and
    fast (just a lock + list append).
    clear_npcs() is called on session-reset events (init / wipe / playerdied).
    get_active_npc_count() is sync best-effort for the health endpoint (no lock).
    """

    def __init__(
        self,
        interval_seconds: float,
        on_package: Callable[[TickPackage], Awaitable[None]],
    ) -> None:
        self._interval = interval_seconds
        self._on_package = on_package
        self._lock = asyncio.Lock()
        self._buffer: list[TypedEvent] = []
        self._active_npc_ids: set[str] = set()
        self._task: Optional[asyncio.Task] = None
        self._last_tick_time: float = 0.0

    async def start(self) -> None:
        """Start the background tick loop task."""
        self._last_tick_time = time.monotonic()
        self._task = asyncio.create_task(self._tick_loop(), name="tick_accumulator")
        logger.info("TickAccumulator started (interval=%.1fs)", self._interval)

    async def stop(self) -> None:
        """Cancel the background tick task cleanly."""
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TickAccumulator stopped")

    async def push(self, event: TypedEvent) -> None:
        """
        Push a TypedEvent into the buffer.

        Updates active_npc_ids when an addnpc event arrives with a parsed
        NPC name.  Uses prefix matching (startswith) to mirror HerikaServer's
        strpos-based dispatch — the DLL may send addnpc variants.
        """
        async with self._lock:
            self._buffer.append(event)
            if event.event_type.startswith("addnpc") and event.parsed_data:
                npc_name = event.parsed_data.get("name")
                if npc_name:
                    self._active_npc_ids.add(npc_name)
                    logger.info("NPC registered: %s (active=%d)",
                                npc_name, len(self._active_npc_ids))
                else:
                    logger.warning("addnpc event has parsed_data but no 'name' key: %s",
                                   event.parsed_data)
            elif event.event_type.startswith("addnpc"):
                logger.warning("addnpc event arrived with parsed_data=None — "
                               "parser may have failed. raw=%.120s",
                               event.raw_data)

    async def clear_npcs(self) -> None:
        """Clear the NPC registry (called on init / wipe / playerdied)."""
        async with self._lock:
            prev_count = len(self._active_npc_ids)
            prev_names = list(self._active_npc_ids) if prev_count <= 10 else []
            self._active_npc_ids.clear()
        logger.info("NPC registry cleared (was %d NPCs: %s)", prev_count, prev_names)

    def get_active_npc_count(self) -> int:
        """Best-effort sync count for the health endpoint (no lock taken)."""
        return len(self._active_npc_ids)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _tick_loop(self) -> None:
        """Background loop: sleep interval then fire a tick, forever."""
        while True:
            await asyncio.sleep(self._interval)
            await self._fire_tick()

    async def _fire_tick(self) -> None:
        """Atomically snapshot the buffer and ship a TickPackage if non-empty."""
        now = time.monotonic()

        async with self._lock:
            if not self._buffer:
                return  # Skip empty ticks
            events = self._buffer[:]
            self._buffer.clear()
            active_ids = list(self._active_npc_ids)

        elapsed_ms = int((now - self._last_tick_time) * 1000)
        self._last_tick_time = now

        package = TickPackage(
            events=events,
            tick_interval_ms=elapsed_ms,
            active_npc_ids=active_ids,
        )

        logger.debug(
            "Tick fired: %d events, npcs=%d",
            len(events), len(active_ids),
        )

        try:
            await self._on_package(package)
        except Exception:
            logger.exception("Error in tick package callback")
