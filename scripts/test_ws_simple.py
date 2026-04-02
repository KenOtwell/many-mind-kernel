"""Minimal WebSocket test: data-only tick, no LLM involved."""
import asyncio
import json
from uuid import uuid4
import websockets

async def test():
    url = "ws://192.168.0.220:8001/ws"
    ws = await websockets.connect(url, ping_interval=30, ping_timeout=60)
    print(f"Connected to {url}")

    # Data-only tick — no player input, should get an ack back instantly
    frame = json.dumps({"type": "tick", "data": {
        "events": [{
            "event_type": "addnpc",
            "local_ts": "2026-03-30T16:00:00",
            "game_ts": 13333334.0,
            "raw_data": "Lydia@Lydia@Female@Nord@000A2C94",
            "parsed_data": None,
        }],
        "tick_interval_ms": 2000,
        "active_npc_ids": ["Lydia"],
        "tick_id": str(uuid4()),
    }})

    await ws.send(frame)
    print("Sent data-only tick. Waiting for ack...")

    try:
        resp = await asyncio.wait_for(ws.recv(), timeout=10)
        data = json.loads(resp)
        print(f"Response type: {data.get('type')}")
        print(f"Data: {json.dumps(data.get('data', {}), indent=2)[:300]}")
    except asyncio.TimeoutError:
        print("TIMEOUT")
    except websockets.exceptions.ConnectionClosedError as e:
        print(f"CONNECTION CLOSED: {e}")

    await ws.close()
    print("Done.")

asyncio.run(test())
