"""
LLM client for Progeny.

Thin adapter targeting llama.cpp server's OpenAI-compatible API at
/v1/chat/completions. The same interface works with Ollama, vLLM, or
any OpenAI-compatible endpoint — just change host/port.

Each call is stateless and self-contained. The prompt carries all
context. No server-side conversation history. No session state.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field

import httpx

from shared.config import settings

logger = logging.getLogger(__name__)


def _env(key: str, default: str) -> str:
    return os.environ.get(key, default)


def _env_int(key: str, default: int) -> int:
    return int(os.environ.get(key, str(default)))


def _env_float(key: str, default: float) -> float:
    return float(os.environ.get(key, str(default)))


@dataclass
class LLMConfig:
    """LLM backend configuration.

    Connection settings (host, port, timeout) live here.
    Sampling settings (temperature, top_p) come from the model profile
    in shared.config.settings.model — set via LLM_PROFILE env var.
    """
    host: str = field(default_factory=lambda: _env("LLM_HOST", "127.0.0.1"))
    port: int = field(default_factory=lambda: _env_int("LLM_PORT", 8080))
    timeout_seconds: float = field(default_factory=lambda: _env_float("LLM_TIMEOUT", 180.0))
    retry_attempts: int = field(default_factory=lambda: _env_int("LLM_RETRIES", 1))

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @property
    def chat_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"


# Module-level default config
_config = LLMConfig()


def configure(config: LLMConfig) -> None:
    """Override the module-level LLM config."""
    global _config
    _config = config


@dataclass
class GenerateResult:
    """LLM response content + backend timing metadata.

    token_logprobs: per-token log probabilities from the LLM generation.
    Each entry is a dict with 'token' (str) and 'logprob' (float).
    None if the backend doesn't support logprobs or the request failed.
    Used by the uncertainty module for per-agent certainty estimation.
    """
    content: str
    prompt_tokens: int = 0
    prompt_ms: float = 0.0
    generated_tokens: int = 0
    generation_ms: float = 0.0
    cache_tokens: int = 0
    token_logprobs: list[dict] | None = None


async def generate(messages: list[dict[str, str]]) -> GenerateResult:
    """
    Send a chat completion request to the LLM server.

    Args:
        messages: Chat completion messages array (system + user messages).

    Returns:
        GenerateResult with response text and timing metadata.

    Raises:
        LLMError: If the LLM is unreachable or returns an error after retries.
    """
    profile = settings.model
    payload: dict = {
        "messages": messages,
        "temperature": profile.temperature,
        "top_p": profile.top_p,
        "stream": False,
        # Request pre-sampling token log probabilities for uncertainty
        # estimation. llama.cpp returns log(softmax(logits)) per token —
        # the model's genuine confidence before sampling distorts it.
        "logprobs": True,
        "top_logprobs": 3,
    }
    if profile.supports_json_mode:
        payload["response_format"] = {"type": "json_object"}

    last_error: Exception | None = None
    attempts = 1 + _config.retry_attempts

    for attempt in range(attempts):
        try:
            async with httpx.AsyncClient(timeout=_config.timeout_seconds) as client:
                response = await client.post(_config.chat_url, json=payload)
                response.raise_for_status()
                data = response.json()

                # Extract content from OpenAI-compatible response
                choices = data.get("choices", [])
                if choices:
                    content = choices[0].get("message", {}).get("content", "")
                    if content:
                        return _build_result(content, data)

                logger.warning("LLM returned empty content on attempt %d", attempt + 1)
                last_error = LLMError("Empty response content")

        except httpx.TimeoutException as exc:
            logger.warning("LLM timeout on attempt %d: %s", attempt + 1, exc)
            last_error = LLMError(f"Timeout: {exc}")
        except httpx.HTTPStatusError as exc:
            logger.warning("LLM HTTP error on attempt %d: %s", attempt + 1, exc)
            last_error = LLMError(f"HTTP {exc.response.status_code}: {exc}")
        except httpx.ConnectError as exc:
            logger.warning("LLM connection error on attempt %d: %s", attempt + 1, exc)
            last_error = LLMError(f"Connection failed: {exc}")
        except Exception as exc:
            logger.error("Unexpected LLM error on attempt %d: %s", attempt + 1, exc)
            last_error = LLMError(f"Unexpected: {exc}")

    raise last_error or LLMError("All attempts failed")


def _build_result(content: str, data: dict) -> GenerateResult:
    """Extract timing metadata and token logprobs from llama.cpp response."""
    # llama.cpp includes timings at top level; OpenAI-compat has usage
    timings = data.get("timings", {})
    usage = data.get("usage", {})

    # Extract per-token logprobs from OAI-compatible response.
    # Format: choices[0].logprobs.content = [{token, logprob, top_logprobs}, ...]
    # Graceful fallback: None if backend doesn't support logprobs.
    token_logprobs = None
    choices = data.get("choices", [])
    if choices:
        logprobs_obj = choices[0].get("logprobs")
        if isinstance(logprobs_obj, dict):
            raw_content = logprobs_obj.get("content")
            if isinstance(raw_content, list):
                token_logprobs = [
                    {"token": entry.get("token", ""), "logprob": entry.get("logprob", 0.0)}
                    for entry in raw_content
                    if isinstance(entry, dict)
                ]

    return GenerateResult(
        content=content,
        prompt_tokens=timings.get("prompt_n", usage.get("prompt_tokens", 0)),
        prompt_ms=timings.get("prompt_ms", 0.0),
        generated_tokens=timings.get("predicted_n", usage.get("completion_tokens", 0)),
        generation_ms=timings.get("predicted_ms", 0.0),
        cache_tokens=timings.get("cache_n", 0),
        token_logprobs=token_logprobs,
    )


async def health_check() -> bool:
    """Check if the LLM server is reachable."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{_config.base_url}/health")
            return response.status_code == 200
    except Exception:
        return False


class LLMError(Exception):
    """Raised when the LLM backend is unreachable or returns an error."""
    pass
