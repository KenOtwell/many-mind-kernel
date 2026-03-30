"""Direct WebSocket round-trip test: bypass Falcon, talk to Progeny."""
import asyncio
import json
import websockets

async def test():
    url = "ws://192.168.0.220:8001/ws"
    try:
        ws = await websockets.connect(url, ping_interval=30, ping_timeout=60)
        print(f"Connected to {url}")
    except Exception as e:
        print(f"Connect failed: {e}")
        return

    frame = json.dumps({"type": "tick", "data": {
        "events": [{
            "event_type": "inputtext",
            "local_ts": "2026-03-30T16:00:00",
            "game_ts": 13333334.0,
            "raw_data": "Hello Lydia, what do you think about dragons?",
            "parsed_data": None,
            "is_turn_trigger": True,
        }],
        "has_turn_trigger": True,
        "tick_interval_ms": 2000,
        "active_npc_ids": ["Lydia"],
        "tick_id": "ws-test-001",
    }})

    await ws.send(frame)
    print("Sent tick with turn trigger. Waiting up to 30s...")

    try:
        resp = await asyncio.wait_for(ws.recv(), timeout=30)
        data = json.loads(resp)
        print(f"Response type: {data.get('type')}")
        if data.get("type") == "turn_response":
            tr = data["data"]
            print(f"Model: {tr.get('model_used')}, time: {tr.get('processing_time_ms')}ms")
            for r in tr.get("responses", []):
                utt = r.get("utterance") or "(no utterance)"
                print(f"  {r.get('agent_id')}: {utt[:120]}")
        else:
            print(f"Full: {json.dumps(data, indent=2)[:500]}")
    except asyncio.TimeoutError:
        print("TIMEOUT - no response in 30s")

    await ws.close()
    print("Done.")

asyncio.run(test())
