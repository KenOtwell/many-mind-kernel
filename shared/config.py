"""
Configuration for the Many-Mind Kernel.

Settings for both Falcon and Progeny services.
Override via environment variables (FALCON_PORT=8080, etc.)
or by modifying the defaults below.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


def _env_bool(key: str, default: bool) -> bool:
    val = os.environ.get(key)
    if val is None:
        return default
    return val.lower() in ("1", "true", "yes")


@dataclass
class QdrantConfig:
    """Qdrant connection — both services connect to the Gaming PC instance."""
    host: str = field(default_factory=lambda: _env("QDRANT_HOST", "127.0.0.1"))
    rest_port: int = field(default_factory=lambda: _env_int("QDRANT_REST_PORT", 6333))
    grpc_port: int = field(default_factory=lambda: _env_int("QDRANT_GRPC_PORT", 6334))


@dataclass
class FalconConfig:
    """Falcon service settings (Gaming PC)."""
    host: str = field(default_factory=lambda: _env("FALCON_HOST", "0.0.0.0"))
    port: int = field(default_factory=lambda: _env_int("FALCON_PORT", 8000))
    comm_path: str = field(default_factory=lambda: _env("FALCON_COMM_PATH", "/comm.php"))
    # Tick cadence: Falcon ships an event package to Progeny every N seconds.
    tick_interval_seconds: float = field(
        default_factory=lambda: _env_float("FALCON_TICK_INTERVAL", 2.0)
    )


@dataclass
class ProgenyConfig:
    """Progeny service settings (Beelink or stub)."""
    host: str = field(default_factory=lambda: _env("PROGENY_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("PROGENY_PORT", 8001))
    timeout_seconds: float = field(default_factory=lambda: _env_float("PROGENY_TIMEOUT", 30.0))
    retry_attempts: int = field(default_factory=lambda: _env_int("PROGENY_RETRIES", 2))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def ws_url(self) -> str:
        """WebSocket URL for persistent Falcon↔Progeny channel."""
        return f"ws://{self.host}:{self.port}/ws"


@dataclass
class EmbeddingConfig:
    """Embedding model settings. Used by both Falcon and Progeny via shared modules."""
    model_name: str = "all-MiniLM-L6-v2"
    device: str = "cpu"
    semantic_dim: int = 384
    emotional_dim: int = 9


@dataclass
class SchedulerConfig:
    """Many-Mind Scheduling thresholds and cadences."""
    # Distance tiers (game units, approximate meters)
    tier0_distance: float = field(default_factory=lambda: _env_float("SCHED_TIER0_DIST", 5.0))
    tier1_distance: float = field(default_factory=lambda: _env_float("SCHED_TIER1_DIST", 20.0))
    tier2_distance: float = field(default_factory=lambda: _env_float("SCHED_TIER2_DIST", 50.0))

    # Harmonic cadence (include agent every N turns at their tier)
    tier0_cadence: int = 1    # every prompt
    tier1_cadence: int = 2    # every 2nd
    tier2_cadence: int = 4    # every 4th
    tier3_cadence: int = 16   # every 16th

    max_agents_per_prompt: int = 16
    max_parallel_slots: int = field(
        default_factory=lambda: _env_int("SCHED_MAX_PARALLEL_SLOTS", 4)
    )
    collaboration_floor_tier: int = 1  # min tier for collaborating NPCs


@dataclass
class ModelProfile:
    """
    LLM model profile — captures model-specific quirks and recommended settings.

    BYOM users set LLM_PROFILE to a known model family (mistral-nemo, qwen2,
    llama3, generic) or override individual settings via env vars. Progeny
    adapts prompt format, sampling, and response parsing accordingly.
    """
    # Sampling
    temperature: float = field(default_factory=lambda: _env_float("LLM_TEMPERATURE", 0.7))
    top_p: float = field(default_factory=lambda: _env_float("LLM_TOP_P", 0.9))
    repeat_penalty: float = field(default_factory=lambda: _env_float("LLM_REPEAT_PENALTY", 1.1))

    # Chat template constraints
    strict_alternation: bool = field(
        default_factory=lambda: _env_bool("LLM_STRICT_ALTERNATION", False)
    )
    """If True, chat messages must alternate user/assistant. Consecutive
    same-role messages are merged. Required for Mistral-family models."""

    # JSON output support
    supports_json_mode: bool = field(
        default_factory=lambda: _env_bool("LLM_JSON_MODE", True)
    )
    """Whether the backend supports response_format: json_object.
    If False, Progeny relies on prompt instructions + repair pass only."""

    # Display name (for logging / health endpoint)
    name: str = field(default_factory=lambda: _env("LLM_PROFILE", "generic"))


# ---------------------------------------------------------------------------
# Built-in profiles for known model families
# ---------------------------------------------------------------------------

MODEL_PROFILES: dict[str, dict] = {
    "mistral-nemo": {
        "temperature": 0.3,
        "top_p": 0.9,
        "repeat_penalty": 1.0,
        "strict_alternation": True,
        "supports_json_mode": True,
        "name": "mistral-nemo",
    },
    "qwen2": {
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "strict_alternation": False,
        "supports_json_mode": True,
        "name": "qwen2",
    },
    "llama3": {
        "temperature": 0.6,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "strict_alternation": True,
        "supports_json_mode": True,
        "name": "llama3",
    },
    "dolphin": {
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "strict_alternation": False,  # ChatML-based, flexible
        "supports_json_mode": True,
        "name": "dolphin",
    },
    "generic": {
        "temperature": 0.7,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
        "strict_alternation": False,
        "supports_json_mode": True,
        "name": "generic",
    },
}


def load_model_profile() -> ModelProfile:
    """
    Load model profile from LLM_PROFILE env var, then overlay any
    individual env var overrides on top.

    Priority: individual env vars > profile defaults > generic defaults.
    """
    profile_name = _env("LLM_PROFILE", "generic")
    base = MODEL_PROFILES.get(profile_name, MODEL_PROFILES["generic"]).copy()

    # Env var overrides win over profile defaults.
    # Only override if the env var is explicitly set.
    if os.environ.get("LLM_TEMPERATURE"):
        base["temperature"] = _env_float("LLM_TEMPERATURE", base["temperature"])
    if os.environ.get("LLM_TOP_P"):
        base["top_p"] = _env_float("LLM_TOP_P", base["top_p"])
    if os.environ.get("LLM_REPEAT_PENALTY"):
        base["repeat_penalty"] = _env_float("LLM_REPEAT_PENALTY", base["repeat_penalty"])
    if os.environ.get("LLM_STRICT_ALTERNATION"):
        base["strict_alternation"] = _env_bool("LLM_STRICT_ALTERNATION", base["strict_alternation"])
    if os.environ.get("LLM_JSON_MODE"):
        base["supports_json_mode"] = _env_bool("LLM_JSON_MODE", base["supports_json_mode"])

    return ModelProfile(**base)


@dataclass
class Settings:
    """Top-level settings container."""
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    falcon: FalconConfig = field(default_factory=FalconConfig)
    progeny: ProgenyConfig = field(default_factory=ProgenyConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    model: ModelProfile = field(default_factory=load_model_profile)

    debug: bool = field(default_factory=lambda: _env_bool("MMK_DEBUG", False))
    log_level: str = field(default_factory=lambda: _env("MMK_LOG_LEVEL", "INFO"))


# Singleton — import this
settings = Settings()
