# Many-Mind Kernel

Skyrim AI Companion — full HerikaServer replacement using a Falcon/Progeny two-service architecture.

**Falcon** (Gaming PC) — stateless current-tick relay: SKSE I/O, embedding, emotional delta, memory retrieval, RAW Qdrant writes.

**Progeny** (Beelink 395AI) — stateful mind engine: event accumulation, harmonic buffers, Many-Mind scheduling, LLM interaction, MOD/MAX Qdrant writes.

They communicate via a single API contract (`POST /ingest`). See `falcon/REQUIREMENTS.md` and `progeny/REQUIREMENTS.md`.

## Quick Start

```bash
pip install -e ".[dev]"

# Run stub Progeny (mock, no Beelink needed)
uvicorn scripts.stub_progeny:app --port 8001

# Run Falcon
uvicorn falcon.api.server:app --port 8000

# Run tests
pytest
```

## Architecture

See `The_Many_Mind_Kernel_Living_Doc.md` for the full design.

Co-Authored-By: Oz <oz-agent@warp.dev>
