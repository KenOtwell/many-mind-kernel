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
import wave
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

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

# Soundcache: diagnostic copies of generated WAVs (not played in-game). The
# in-game playback path is `AIAGENT_VOICE_DIR / <voicetype> / <voicetype>.wav`
# (see voice_placeholder_path); soundcache survives as a debug artifact so the
# files can be opened in a media player to verify synth quality offline.
SOUNDCACHE_DIR = os.environ.get(
    "SOUNDCACHE_DIR",
    r"C:\Users\Ken\Projects\xVASynth v3.0.0\v3.0.0\resources\app\soundcache",
)

# AIAgent voice replacement directory — the path SKSE actually reads from
# when AIAgent.esp dialogue triggers. CHIM's SKVA Synth plugin POSTs vanilla
# voice samples to /vsx.php at session init; we save each to
# `<AIAGENT_VOICE_DIR>/<voicetype>/<voicetype>.wav` so the file exists for
# SKSE to load. On each LLM turn we overwrite that same file with the
# synthesized audio. Same filename per voicetype, content updated per turn.
AIAGENT_VOICE_DIR = os.environ.get(
    "AIAGENT_VOICE_DIR",
    r"C:\Modlists\PandasSovngarde\mods\AIAgent\Sound\Voice\AIAgent.esp",
)

# Default voice for NPCs without a specific mapping
DEFAULT_VOICE_ID = "sk_malenord"

# Track which model is currently loaded (xVASynth loads one at a time)
_current_model: Optional[str] = None

# NPC name -> voiceId mapping (populated from addnpc data or manual config)
# TODO: Build this from addnpc voice type data when available
_npc_voice_map: dict[str, str] = {}


# Skyrim's voice engine reads voice files at this sample rate. xVASynth
# outputs 22050 Hz, so synthesized WAVs must be upsampled before SKSE will
# accept them. Confirmed by inspection: every shipped AIAgent voice
# placeholder is 44.1 kHz mono 16-bit PCM.
SKSE_VOICE_RATE_HZ = 44100


def _resample_wav_inplace(path: str, target_rate: int = SKSE_VOICE_RATE_HZ) -> None:
    """Resample a WAV file in place to `target_rate` if it's not already there.

    Linear interpolation via numpy. Adequate for voice; trades a touch of
    high-frequency aliasing for stdlib-only operation (no scipy dependency).
    Assumes 16-bit PCM mono input — the format xVASynth always produces.
    """
    try:
        with wave.open(path, "rb") as wf:
            src_rate = wf.getframerate()
            if src_rate == target_rate:
                return  # already correct, nothing to do
            nchan = wf.getnchannels()
            sampw = wf.getsampwidth()
            nframes = wf.getnframes()
            raw = wf.readframes(nframes)
    except (wave.Error, FileNotFoundError) as exc:
        logger.warning("resample: cannot open %s: %s", path, exc)
        return

    if sampw != 2:
        logger.warning(
            "resample: unexpected sample width %d at %s — skipping",
            sampw, path,
        )
        return

    samples = np.frombuffer(raw, dtype=np.int16)
    if nchan > 1:
        samples = samples.reshape(-1, nchan)

    ratio = target_rate / src_rate
    in_len = samples.shape[0]
    out_len = int(round(in_len * ratio))
    x_new = np.arange(out_len, dtype=np.float64) / ratio
    x_old = np.arange(in_len, dtype=np.float64)

    if samples.ndim == 1:
        ups = np.interp(x_new, x_old, samples).astype(np.int16)
    else:
        ups = np.empty((out_len, nchan), dtype=np.int16)
        for c in range(nchan):
            ups[:, c] = np.interp(x_new, x_old, samples[:, c]).astype(np.int16)

    with wave.open(path, "wb") as wf:
        wf.setnchannels(nchan)
        wf.setsampwidth(sampw)
        wf.setframerate(target_rate)
        wf.writeframes(ups.tobytes())


def voice_placeholder_path(voicetype: str) -> str:
    """Return the path SKSE reads when AIAgent.esp dialogue triggers.

    Format: `<AIAGENT_VOICE_DIR>/<voicetype>/<voicetype>.wav`. CHIM ships a
    handful of these placeholders for vanilla voicetypes; the rest are
    created on the fly when SKVA Synth POSTs the vanilla sample to /vsx.php
    at session init. Falcon overwrites the same file with synthesized audio
    on each LLM turn for the NPC of that voicetype.
    """
    return str(Path(AIAGENT_VOICE_DIR) / voicetype / f"{voicetype}.wav")


def extract_voicetype_from_oname(oname: str) -> Optional[str]:
    """Pull the voicetype out of a Bethesda voice path.

    Examples:
      `Sound\\Voice\\Skyrim.esm\\maleeventoned\\<line>.fuz` -> "maleeventoned"
      `Sound/Voice/Skyrim.esm/malecommoner/<line>.fuz`      -> "malecommoner"

    The voicetype is always the parent directory of the audio filename in
    Bethesda's layout, so we take the second-to-last path segment.
    """
    if not oname:
        return None
    parts = [p for p in oname.replace("/", "\\").split("\\") if p]
    if len(parts) < 2:
        return None
    return parts[-2]


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


async def synthesize(
    npc_name: str,
    text: str,
    output_path: str,
) -> Optional[str]:
    """Synthesize speech for an NPC and save to the given path.

    Args:
        npc_name: NPC name (for voice model lookup).
        text: Dialogue text to synthesize.
        output_path: Absolute path where the WAV should be written. Caller
            decides whether this is the SKSE placeholder path (in-game
            playback) or a soundcache diagnostic path (offline only).

    Returns:
        Absolute path to the generated WAV file, or None on failure.
    """
    if not text or not text.strip():
        return None

    voice_id = get_voice_id(npc_name)
    chars = len(text)

    # Load model if needed (and the .json sidecar that sets self.base_emb)
    if not await load_model(voice_id):
        return None

    # Ensure the output directory exists (handles both AIAgent.esp/<voicetype>/
    # and soundcache/ targets without special-casing)
    save_path = output_path
    parent = os.path.dirname(save_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

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
        if not out_path.exists():
            logger.warning(
                "TTS synth claimed success but file missing: %s voice=%s chars=%d -> %s (%.0fms)",
                npc_name, voice_id, chars, save_path, elapsed_ms,
            )
            return save_path

        # Upsample 22050 -> 44100 Hz in place so SKSE's voice engine accepts
        # the file. This is a real prerequisite for in-game playback, not a
        # quality knob — Skyrim rejects voice files at non-standard rates.
        t_rs = time.perf_counter()
        _resample_wav_inplace(save_path)
        resample_ms = (time.perf_counter() - t_rs) * 1000

        size_bytes = out_path.stat().st_size
        logger.info(
            "TTS synth done: %s voice=%s chars=%d -> %s (%d bytes, synth=%.0fms resample=%.0fms)",
            npc_name, voice_id, chars, save_path, size_bytes, elapsed_ms, resample_ms,
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
    voicetypes: Optional[dict[str, str]] = None,
) -> dict[str, Optional[str]]:
    """Synthesize audio for all agent responses that have utterances.

    Writes WAVs to soundcache for offline verification only. The previous
    behavior (writing to AIAgent.esp/<voicetype>/<voicetype>.wav) broke
    CHIM's own canned dialogue audio without enabling LLM dialogue audio,
    and Procmon confirmed SKSE does not read that path per dialogue line.
    A working in-game delivery path is still TBD.

    Args:
        responses: List of AgentResponse dicts.
        voicetypes: Optional mapping of agent_id -> Bethesda voicetype
            (e.g. "maleeventoned"). Sourced from tick_accumulator's
            registry, which is populated by SKVA's /vsx.php POSTs.

    Returns:
        Dict mapping agent_id -> output WAV path (or None if synth failed).
    """
    voicetypes = voicetypes or {}
    results: dict[str, Optional[str]] = {}

    for resp in responses:
        agent_id = resp.get("agent_id", "Unknown")
        utterance = resp.get("utterance")

        if not utterance:
            continue

        # DISABLED: writing to AIAgent.esp/<voicetype>/<voicetype>.wav broke
        # CHIM's own canned dialogue audio (NPC mouths moved, no sound).
        # Procmon also confirmed SKSE does NOT read this path per dialogue
        # line. Until we find the real delivery mechanism, everything goes
        # to soundcache for offline verification only.
        #
        # The voicetype lookup still happens — used in the filename so the
        # diagnostic WAVs are grouped by voicetype for easier inspection.
        voicetype = voicetypes.get(agent_id) or "unknown"
        import hashlib
        text_hash = hashlib.md5(utterance.encode()).hexdigest()[:8]
        output_path = str(
            Path(SOUNDCACHE_DIR) / f"{agent_id}_{voicetype}_{text_hash}.wav"
        )

        wav_path = await synthesize(agent_id, utterance, output_path)
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
