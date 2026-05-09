"""
xVASynth TTS integration for Falcon.

Converts NPC dialogue text to WAV audio via xVASynth's HTTP API.
The generated WAV is saved to the soundcache directory where CHIM's
SKSE plugin expects to find it for in-game playback.

xVASynth v3 API:
  POST /loadModel  - load a voice model (once per voice type)
  POST /synthesize - generate WAV from text

Voice model mapping: NPC voice type -> xVASynth voiceId.
The SKSE plugin sends voice type info via addnpc events. For now,
we use a default voice (MaleNord) and build the mapping over time.
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# xVASynth server config
XVASYNTH_HOST = os.environ.get("XVASYNTH_HOST", "127.0.0.1")
XVASYNTH_PORT = int(os.environ.get("XVASYNTH_PORT", "8008"))
XVASYNTH_URL = f"http://{XVASYNTH_HOST}:{XVASYNTH_PORT}"

# Voice models directory
XVASYNTH_MODELS_DIR = os.environ.get(
    "XVASYNTH_MODELS_DIR",
    r"C:\Users\Ken\Projects\xVASynth v3.0.0\v3.0.0\resources\app\models\skyrim",
)

# Soundcache: where CHIM's SKSE plugin looks for generated WAV files
SOUNDCACHE_DIR = os.environ.get(
    "SOUNDCACHE_DIR",
    r"C:\Users\Ken\Projects\xVASynth v3.0.0\v3.0.0\resources\app\soundcache",
)

# Default voice for NPCs without a specific mapping
DEFAULT_VOICE_ID = "sk_malenord"

# Track which model is currently loaded (xVASynth loads one at a time)
_current_model: Optional[str] = None

# NPC name -> voiceId mapping (populated from addnpc data or manual config)
# TODO: Build this from addnpc voice type data when available
_npc_voice_map: dict[str, str] = {}


def _model_path_no_ext(voice_id: str) -> str:
    """Return the model path WITHOUT extension.

    xVASynth's /loadModel handler appends '.pt' to whatever string is sent
    (server.py: `models_manager.load_model(modelType, ckpt+".pt", ...)`),
    and the JSON sidecar is read off the same stem. Sending the .json path
    here was the bug that produced silently-broken model loads.
    """
    return str(Path(XVASYNTH_MODELS_DIR) / voice_id)


def _model_json_path(voice_id: str) -> str:
    """Path to the model's JSON sidecar (used only to confirm the voice exists)."""
    return str(Path(XVASYNTH_MODELS_DIR) / f"{voice_id}.json")


def get_voice_id(npc_name: str) -> str:
    """Get the xVASynth voiceId for an NPC.

    Checks the NPC voice map first, then falls back to default.
    Voice IDs are like 'sk_malenord', 'sk_femalecommoner', 'sk_serana'.
    """
    return _npc_voice_map.get(npc_name, DEFAULT_VOICE_ID)


def set_npc_voice(npc_name: str, voice_id: str) -> None:
    """Register a voice mapping for an NPC."""
    _npc_voice_map[npc_name] = voice_id
    logger.info("Voice mapping: %s -> %s", npc_name, voice_id)


async def load_model(voice_id: str) -> bool:
    """Load a voice model in xVASynth. Returns True on success.

    Sends the model path WITHOUT extension; xVASynth appends .pt for the
    weights and reads the .json sidecar from the same stem. The sidecar is
    where `self.base_emb` (used by the synth path) gets populated, so a
    correct path here is what makes synthesis produce real audio rather
    than noise.

    No separate vocoder load is needed for xVAPitch — the acoustic model
    bundles its own vocoder, and the UI confirms this by sending
    `vocoder: "n/a"` for xVAPitch in /synthesize calls.

    Logs every transition (including same-model no-ops) at INFO so we can
    detect parallel callers (e.g. CHIM SKVA Synth) thrashing the single
    loaded-model slot.
    """
    global _current_model

    if _current_model == voice_id:
        logger.debug("xVASynth model already loaded: %s (no-op)", voice_id)
        return True

    previous = _current_model
    bare_path = _model_path_no_ext(voice_id)
    pt_path = f"{bare_path}.pt"
    if not Path(pt_path).exists():
        logger.error("Voice model weights not found: %s", pt_path)
        return False
    if not Path(_model_json_path(voice_id)).exists():
        logger.error("Voice model JSON sidecar not found: %s",
                     _model_json_path(voice_id))
        return False

    data = {
        "model": bare_path,           # xVASynth appends ".pt" itself
        "modelType": "xVAPitch",
        "base_lang": "en",
        "pluginsContext": "{}",
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(f"{XVASYNTH_URL}/loadModel", json=data)
            resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        _current_model = voice_id
        logger.info("xVASynth model load: %s (was: %s) in %.0fms",
                    voice_id, previous, elapsed_ms)
        return True
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error("xVASynth model load FAILED: %s (was: %s) after %.0fms: %s",
                     voice_id, previous, elapsed_ms, exc)
        return False


def _snapshot_soundcache_for(agent_prefix: str) -> list[str]:
    """Return existing WAV filenames in soundcache matching `<agent_prefix>_*`.

    Used to detect parallel writers (e.g. CHIM SKVA Synth) dropping audio
    for the same NPC. An unexpected pre-existing file is the signal.
    """
    cache_dir = Path(SOUNDCACHE_DIR)
    if not cache_dir.exists():
        return []
    prefix = f"{agent_prefix}_"
    try:
        return sorted(
            p.name for p in cache_dir.iterdir()
            if p.is_file() and p.suffix.lower() == ".wav" and p.name.startswith(prefix)
        )
    except OSError as exc:
        logger.debug("Soundcache snapshot failed: %s", exc)
        return []


async def synthesize(
    npc_name: str,
    text: str,
    output_filename: str,
) -> Optional[str]:
    """Synthesize speech for an NPC and save to soundcache.

    Args:
        npc_name: NPC name (for voice model lookup).
        text: Dialogue text to synthesize.
        output_filename: WAV filename (without path).

    Returns:
        Full path to the generated WAV file, or None on failure.
    """
    if not text or not text.strip():
        return None

    voice_id = get_voice_id(npc_name)
    chars = len(text)

    # Pre-write soundcache snapshot. If a WAV for this NPC already exists
    # before we generate one, somebody else (likely CHIM SKVA Synth) wrote it.
    pre_snapshot = _snapshot_soundcache_for(npc_name)
    if pre_snapshot:
        logger.info("TTS pre-write snapshot for %s: %d existing file(s) %s",
                    npc_name, len(pre_snapshot), pre_snapshot[:5])

    # Load model if needed (and the .json sidecar that sets self.base_emb)
    if not await load_model(voice_id):
        return None

    # Ensure soundcache directory exists
    os.makedirs(SOUNDCACHE_DIR, exist_ok=True)
    save_path = str(Path(SOUNDCACHE_DIR) / output_filename)

    # Mirror xVASynth UI's /synthesize payload (script.js:769-785).
    # For xVAPitch:
    #   * vocoder is "n/a" — the acoustic model bundles its own.
    #   * base_emb is empty — the model uses self.base_emb populated from the
    #     .json sidecar during /loadModel. Sending a non-empty value would
    #     override it (used only for style-embedding selection in the UI).
    data = {
        "modelType": "xVAPitch",
        "sequence": f" {text.strip()} ",  # pad with spaces to avoid cutoffs
        "pace": 1.0,
        "outfile": save_path,
        "vocoder": "n/a",
        "base_lang": "en",
        "base_emb": "",
        "useSR": 0,
        "useCleanup": 0,
        "pluginsContext": json.dumps({}),
    }

    t0 = time.perf_counter()
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{XVASYNTH_URL}/synthesize", json=data)
            resp.raise_for_status()
        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Confirm the file actually landed and capture its size.
        out_path = Path(save_path)
        if out_path.exists():
            size_bytes = out_path.stat().st_size
            logger.info(
                "TTS synth done: %s voice=%s chars=%d -> %s (%d bytes, %.0fms)",
                npc_name, voice_id, chars, output_filename, size_bytes, elapsed_ms,
            )
        else:
            logger.warning(
                "TTS synth claimed success but file missing: %s voice=%s chars=%d -> %s (%.0fms)",
                npc_name, voice_id, chars, save_path, elapsed_ms,
            )
        return save_path
    except Exception as exc:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "TTS synth FAILED: %s voice=%s chars=%d after %.0fms: %s",
            npc_name, voice_id, chars, elapsed_ms, exc,
        )
        return None


async def synthesize_responses(
    responses: list[dict],
) -> dict[str, Optional[str]]:
    """Synthesize audio for all agent responses that have utterances.

    Args:
        responses: List of AgentResponse dicts.

    Returns:
        Dict mapping agent_id -> WAV file path (or None if failed).
    """
    results: dict[str, Optional[str]] = {}

    for resp in responses:
        agent_id = resp.get("agent_id", "Unknown")
        utterance = resp.get("utterance")

        if not utterance:
            continue

        # Generate a unique filename
        import hashlib
        text_hash = hashlib.md5(utterance.encode()).hexdigest()[:8]
        filename = f"{agent_id}_{text_hash}.wav"

        wav_path = await synthesize(agent_id, utterance, filename)
        results[agent_id] = wav_path

    return results


async def health_check() -> bool:
    """Check if xVASynth is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{XVASYNTH_URL}/api/health")
            return resp.status_code == 200
    except Exception:
        return False
