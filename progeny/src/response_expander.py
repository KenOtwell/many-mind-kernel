"""
Response expander for Progeny.

Extractor, not validator. Parses LLM JSON responses into AgentResponse
objects with a graceful degradation cascade:
  1. Strict JSON parse — full structured response
  2. Plaintext fallback — entire response becomes utterance for first agent

Includes a repair pass (strip markdown fences, trailing commas) before
strict parse. Field-level regex extraction planned for Phase 2.

Degradation priority: utterance > actions > new_memories > updated_harmonics.
History reflects reality — the agent's history is built from what was
actually extracted, not what was requested.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from shared.constants import ACTOR_VALUE_RANGES, COMMAND_VOCABULARY
from shared.schemas import (
    AgentResponse,
    ActionCommand,
    ActorValueDeltas,
    ExtractionLevel,
    NewMemory,
    UpdatedHarmonics,
)

logger = logging.getLogger(__name__)


def expand_response(
    raw_text: str,
    expected_agent_ids: list[str],
) -> list[AgentResponse]:
    """
    Extract structured AgentResponses from raw LLM output.

    Tries strict JSON parse first; falls back to plaintext.
    Returns one AgentResponse per expected agent (may be empty for some).
    """
    # Repair pass: strip markdown fences, trailing commas, etc.
    cleaned = _repair_llm_output(raw_text)

    # Try strict JSON parse
    responses = _try_strict_parse(cleaned, expected_agent_ids)
    if responses is not None:
        return responses

    # Plaintext fallback — entire text becomes utterance for first agent
    logger.warning("Strict JSON parse failed; falling back to plaintext")
    return _plaintext_fallback(raw_text, expected_agent_ids)


# ---------------------------------------------------------------------------
# LLM output repair — normalize common model quirks before JSON parse
# ---------------------------------------------------------------------------

# Matches ```json ... ```, ```JSON ... ```, or bare ``` ... ```
_FENCE_RE = re.compile(
    r"```(?:json|JSON)?\s*\n?(.*?)\n?\s*```",
    re.DOTALL,
)

# Trailing commas before } or ] — common LLM mistake
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")

# Line comments (// ...) inside JSON — Mistral and other models do this
_LINE_COMMENT_RE = re.compile(r"\s*//[^\n]*")

# <think>...</think> reasoning blocks — reasoning models (Ornith, Qwen3,
# DeepSeek-R1) emit these before the JSON answer when served via llama.cpp.
# llama.cpp's JSON grammar mode suppresses them in most cases, but this
# strip runs first as defense-in-depth for any that leak through.
_THINK_BLOCK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _repair_llm_output(raw: str) -> str:
    """
    Normalize common LLM output quirks before JSON parse.

    Handles:
    - <think>...</think> reasoning blocks (Ornith, Qwen3, DeepSeek-R1)
    - Markdown code fences (```json ... ```)
    - Trailing commas before } or ]
    - Leading/trailing whitespace
    """
    text = raw.strip()

    # Strip reasoning blocks before any other repair — they can contain
    # arbitrary text including braces that would confuse later passes.
    text = _THINK_BLOCK_RE.sub("", text).strip()

    # Strip markdown code fences — extract inner content
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()

    # Strip // line comments (LLMs like Mistral add these to JSON)
    text = _LINE_COMMENT_RE.sub("", text)

    # Fix trailing commas
    text = _TRAILING_COMMA_RE.sub(r"\1", text)

    return text


def _try_strict_parse(
    raw_text: str,
    expected_agent_ids: list[str],
) -> list[AgentResponse] | None:
    """
    Attempt strict JSON parse of the LLM response.

    Expected format: {"responses": [{"agent_id": ..., ...}, ...]}
    Returns None if parse fails.
    """
    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError:
        return None

    if not isinstance(data, dict):
        return None

    raw_responses = data.get("responses")
    if not isinstance(raw_responses, list):
        return None

    # Index by agent_id for lookup
    response_map: dict[str, dict] = {}
    for entry in raw_responses:
        if isinstance(entry, dict) and "agent_id" in entry:
            response_map[entry["agent_id"]] = entry

    # Build AgentResponse for each expected agent in order
    results = []
    for agent_id in expected_agent_ids:
        entry = response_map.get(agent_id)
        if entry:
            results.append(_parse_agent_entry(entry))
        else:
            # LLM skipped this agent — graceful degradation
            results.append(AgentResponse(
                agent_id=agent_id,
                extraction_level=ExtractionLevel.STRICT,
            ))

    return results


def _parse_agent_entry(entry: dict[str, Any]) -> AgentResponse:
    """Parse a single agent response entry from the LLM JSON."""
    agent_id = entry.get("agent_id", "unknown")

    # Utterance — LLM may use "utterance" (Tier 0) or "brief_utterance" (lower tiers)
    utterance = entry.get("utterance") or entry.get("brief_utterance")
    if utterance is not None:
        utterance = str(utterance).strip() or None

    # Actor value deltas — validate and clamp
    avd = _parse_actor_value_deltas(entry.get("actor_value_deltas"))

    # Actions — validate against command vocabulary
    actions = _parse_actions(entry.get("actions", []))

    # Updated harmonics
    harmonics = _parse_harmonics(entry.get("updated_harmonics"))

    # New memories
    memories = _parse_memories(entry.get("new_memories", []))

    return AgentResponse(
        agent_id=agent_id,
        utterance=utterance,
        actor_value_deltas=avd,
        actions=actions,
        updated_harmonics=harmonics,
        new_memories=memories,
        extraction_level=ExtractionLevel.STRICT,
    )


def _parse_actor_value_deltas(raw: Any) -> ActorValueDeltas | None:
    """Validate and clamp actor value deltas to valid ranges."""
    if not isinstance(raw, dict):
        return None

    clamped: dict[str, int | None] = {}
    for name, (lo, hi, _desc) in ACTOR_VALUE_RANGES.items():
        val = raw.get(name)
        if val is not None:
            try:
                val = int(val)
                val = max(lo, min(hi, val))  # Clamp to valid range
                clamped[name] = val
            except (ValueError, TypeError):
                continue

    if not clamped:
        return None

    return ActorValueDeltas(**clamped)


def _parse_actions(raw: Any) -> list[ActionCommand]:
    """Validate actions against the 43-command vocabulary. Strip unknown."""
    if not isinstance(raw, list):
        return []

    actions = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        command = item.get("command", "")
        if command not in COMMAND_VOCABULARY:
            logger.debug("Stripping unknown command: %s", command)
            continue
        actions.append(ActionCommand(
            command=command,
            target=item.get("target"),
            item=item.get("item"),
        ))

    return actions


def _parse_harmonics(raw: Any) -> UpdatedHarmonics | None:
    """Parse LLM-proposed emotional state update."""
    if not isinstance(raw, dict):
        return None
    base_vector = raw.get("base_vector")
    if not isinstance(base_vector, list) or len(base_vector) != 9:
        return None
    try:
        return UpdatedHarmonics(base_vector=[float(v) for v in base_vector])
    except (ValueError, TypeError):
        return None


def _parse_memories(raw: Any) -> list[NewMemory]:
    """Parse new memories the LLM wants to store."""
    if not isinstance(raw, list):
        return []
    memories = []
    for item in raw:
        if isinstance(item, dict) and "text" in item:
            text = str(item["text"]).strip()
            if text:
                memories.append(NewMemory(text=text))
    return memories


def _plaintext_fallback(
    raw_text: str,
    expected_agent_ids: list[str],
) -> list[AgentResponse]:
    """
    Last resort: entire LLM output becomes the utterance for the first agent.

    No actions, no harmonics, no memories. The agent spoke but didn't act.
    """
    results = []
    text = raw_text.strip()

    for i, agent_id in enumerate(expected_agent_ids):
        if i == 0 and text:
            results.append(AgentResponse(
                agent_id=agent_id,
                utterance=text,
                extraction_level=ExtractionLevel.PLAINTEXT,
            ))
        else:
            results.append(AgentResponse(
                agent_id=agent_id,
                extraction_level=ExtractionLevel.PLAINTEXT,
            ))

    return results
