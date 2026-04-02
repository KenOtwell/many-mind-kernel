# MMK Startup & Debugging Guide

*For session continuity — what next-you needs to know.*

## Service Launch Order

1. **Qdrant** (Gaming PC)
   ```powershell
   Start-Process -FilePath "C:\Tools\qdrant\qdrant.exe" -WorkingDirectory "C:\Tools\qdrant"
   ```
   Verify: `Invoke-RestMethod http://127.0.0.1:6333/collections`

2. **Progeny** (Beelink at 192.168.0.220)
   - SSH to Beelink, start Progeny service (port 8001)
   - Verify: `Invoke-RestMethod http://192.168.0.220:8001/health`

3. **Falcon** (Gaming PC)
   ```powershell
   $env:PROGENY_HOST = "192.168.0.220"
   $env:PROGENY_PORT = "8001"
   Start-Process python -ArgumentList "-m","uvicorn","falcon.api.server:app","--host","0.0.0.0","--port","8000","--app-dir","C:\Users\Ken\Projects\many-mind-kernel" -WindowStyle Normal
   ```
   Wait ~12 seconds for model loading, then verify:
   ```powershell
   Invoke-RestMethod http://127.0.0.1:8000/health
   ```
   Check: `ws_connected: True`, `active_npcs` (0 until Skyrim loads)

4. **Skyrim** — Launch from MO2 at `C:\Modlists\PandasSovngarde`
   - AIAgent + Papyrus MessageBox mods must be enabled
   - `AIAgent.ini` at `mods\AIAgent\SKSE\Plugins\AIAgent.ini` → `SERVER=127.0.0.1 PORT=8000 PATH=/comm.php POLINT=1`
   - After loading a save, `active_npcs` should climb in Falcon health

## Health Checks

```powershell
# All three at once:
"Falcon:";  Invoke-RestMethod http://127.0.0.1:8000/health -ErrorAction SilentlyContinue
"Progeny:"; Invoke-RestMethod http://192.168.0.220:8001/health -ErrorAction SilentlyContinue
"Qdrant:";  Invoke-RestMethod http://127.0.0.1:6333/collections -ErrorAction SilentlyContinue | Select-Object status
```

Falcon health fields:
- `ws_connected` — WebSocket to Progeny alive?
- `active_npcs` — NPCs registered via `addnpc` events
- `queue_depth` — responses waiting for SKSE to poll

## Test Scripts

```powershell
# Direct WebSocket round-trip (bypasses Falcon, talks to Progeny directly):
python scripts/test_ws_roundtrip.py

# Simple data-only tick (no LLM):
python scripts/test_ws_simple.py

# Manual player input injection (through Falcon):
$data = "inputtext|$(Get-Date -Format 'yyyyMMddHHmmss')|13333334|Hello there"
$b64 = [Convert]::ToBase64String([System.Text.Encoding]::UTF8.GetBytes($data))
Invoke-WebRequest -Uri "http://127.0.0.1:8000/comm.php?DATA=$b64" -Method Get
```

## Known Working

- **Full WebSocket round-trip**: `test_ws_roundtrip.py` → Progeny → LLM (13s) → turn_response with NPC dialogue ✅
- **SKSE event parsing**: All event types parse correctly (inputtext/inputtext_s=player input, request=local, addnpc/info*/etc.=accumulate) ✅
- **GET+POST routing**: SKSE DLL sends GET with base64 query params, Falcon accepts both ✅
- **WebSocket non-blocking**: Progeny processes turns via `asyncio.create_task()`, receive loop stays alive ✅
- **tick_id**: Must be UUID format (uuid4), not plain string ✅

## Current Bug: `active_npcs` stays 0

**Symptom**: After save reload, Falcon health shows `active_npcs: 0` even though SKSE is sending events.

**The NPC registration path**:
1. SKSE sends `addnpc` event as GET with base64 data
2. `routes.py` → `_handle_skse_request()` → `parse_event()` → `parse_typed_data("addnpc", data)`
3. `tick_accumulator.push()` checks `event.event_type == "addnpc" and event.parsed_data`
4. If `parsed_data` has a `"name"` key, NPC is added to `_active_npc_ids`

**Likely failure points**:
- `parse_event()` may not recognize the raw wire data as `addnpc` (event type detection)
- The SKSE DLL may send `addnpc` with a different prefix or encoding than expected
- `parse_typed_data()` may fail on real SKSE `addnpc` payloads (43+ fields, different from test data)
- `parsed_data` may be `None` if parsing throws an exception, so the NPC never registers
- `init` event may fire AFTER `addnpc` events, clearing the registry

**Debug approach**:
1. Add logging to `_handle_skse_request()` to print raw event type + first 100 chars of data for every event
2. Add logging to `tick_accumulator.push()` when `addnpc` arrives — log whether `parsed_data` is None
3. Check if `init` fires after `addnpc` (ordering issue)
4. Compare real SKSE `addnpc` base64 payload against test data

**Quick debug command** (add temporarily to `_handle_skse_request` after parse_event):
```python
if parsed.event_type == "addnpc":
    logger.warning("ADDNPC raw: %s", parsed.data[:200])
    pd = parse_typed_data("addnpc", parsed.data)
    logger.warning("ADDNPC parsed: name=%s, result_type=%s", pd.get("name") if pd else "NONE", type(pd))
```

## Uncommitted Changes

- `falcon/api/routes.py` — added `ws_connected` to health endpoint (diagnostic, should commit)

## Architecture Quick Reference

```
SKSE DLL (GET /comm.php?DATA=<base64>) → Falcon (parse, accumulate, tick) 
    → WebSocket ws://progeny:8001/ws → Progeny (ingest, LLM, respond)
    → WebSocket turn_response → Falcon (_handle_turn_response → wire format → queue)
    → SKSE polls request → dequeue → NPC speaks
```

- Falcon: `C:\Users\Ken\Projects\many-mind-kernel\falcon\`
- Progeny: runs on Beelink 192.168.0.220
- Qdrant: `C:\Tools\qdrant\qdrant.exe` (ports 6333/6334)
- MO2: `C:\Modlists\PandasSovngarde`
- CHIM mod: `C:\Modlists\PandasSovngarde\mods\AIAgent`
- Papyrus compiler: see `docs/The_Many_Mind_Kernel_Living_Doc.md` → Papyrus Build Toolchain section
- AIAgent.log: `%USERPROFILE%\Documents\My Games\Skyrim VR\SKSE\AIAgent.log`
