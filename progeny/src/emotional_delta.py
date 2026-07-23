"""
Bidirectional emotional delta pipeline for Progeny.

Text → embedding → 9d semagram → harmonic buffer update → EmotionalDelta.

The same pipeline runs in both directions:
  Inbound  — game events and player speech shift agent emotional state.
  Outbound — the agent's own utterances shift its state further.

This is the heart: a single coherent path where all text, regardless of
source or direction, passes through embed → project → update. The agent
cannot distinguish "I felt this" from "I said this." Both shift the
harmonics. Both become memory keys.

Public API:
  process_text(agent_id, text, harmonic_state)        → EmotionalDelta
  process_texts(pairs, harmonic_state)                → dict[str, EmotionalDelta]
  process_inbound(turn_context, harmonic_state)       → dict[str, EmotionalDelta]
  process_outbound(responses, harmonic_state)         → dict[str, EmotionalDelta]
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import TYPE_CHECKING

import numpy as np

from mindcore import embedding
from mindcore import emotional as emotional_projection
from mindcore.harmonic_buffer import EmotionalDelta, HarmonicState
from mindcore.event_log import get_event_log

if TYPE_CHECKING:
    from progeny.src.event_accumulator import TurnContext
    from shared.schemas import AgentResponse

logger = logging.getLogger(__name__)


def process_text(
    agent_id: str,
    text: str,
    harmonic_state: HarmonicState,
) -> EmotionalDelta:
    """Single text → EmotionalDelta for one agent.

    The fundamental operation: one piece of text shifts one agent's
    emotional state. Returns the delta signal for prompt building and
    scheduling decisions.

    Args:
        agent_id:       The agent whose harmonic buffer to update.
        text:           The text to embed and project.
        harmonic_state: The live state container to update in-place.

    Returns:
        EmotionalDelta with semagram, curvature, snap, and λ.
    """
    emb = embedding.embed_one(text)
    sem = emotional_projection.project(emb)
    now = time.monotonic()
    delta = harmonic_state.update(agent_id, sem, now=now)
    # Event log: capture the exact input sem folded into the buffer, the raw
    # text anchor, the derived signals, and the exact decay time `now` so the
    # fold is deterministic (replay + nostalgia/drift trace).
    get_event_log().log_delta(agent_id, [text], sem, delta, now=now)
    return delta


def process_texts(
    pairs: list[tuple[str, str]],
    harmonic_state: HarmonicState,
) -> dict[str, EmotionalDelta]:
    """Batch: process multiple (agent_id, text) pairs in one embed pass.

    All texts are embedded together for efficiency. When the same agent
    appears multiple times (e.g. heard player input AND has speech events),
    their semagrams are averaged before the single harmonic update — the
    tick's full emotional gestalt lands as one coherent event.

    Args:
        pairs:          List of (agent_id, text) tuples. Ordering within
                        each agent's group does not matter; they are averaged.
        harmonic_state: The live state container to update in-place.

    Returns:
        Dict mapping agent_id → EmotionalDelta (one update per agent).
    """
    if not pairs:
        return {}

    texts = [text for _, text in pairs]
    all_embs = embedding.embed(texts)                        # (N, 384)
    all_sems = emotional_projection.project_batch(all_embs)  # (N, 9)

    # Group semagrams by agent — one averaged update per agent per tick.
    # Texts are grouped in parallel so the event log records the raw anchors
    # alongside the exact averaged sem folded into the buffer.
    agent_sem_lists: dict[str, list[np.ndarray]] = defaultdict(list)
    agent_text_lists: dict[str, list[str]] = defaultdict(list)
    for (agent_id, text), sem in zip(pairs, all_sems):
        agent_sem_lists[agent_id].append(sem)
        agent_text_lists[agent_id].append(text)

    results: dict[str, EmotionalDelta] = {}
    event_log = get_event_log()
    # One timestamp for the whole batch — all agents updated this tick share the
    # same decay instant, and it is logged so replay folds deterministically.
    now = time.monotonic()
    for agent_id, sems in agent_sem_lists.items():
        mean_sem = np.mean(sems, axis=0).tolist()
        delta = harmonic_state.update(agent_id, mean_sem, now=now)
        results[agent_id] = delta
        event_log.log_delta(
            agent_id, agent_text_lists[agent_id], mean_sem, delta, now=now)

    return results


def process_inbound(
    turn_context: "TurnContext",
    harmonic_state: HarmonicState,
) -> dict[str, EmotionalDelta]:
    """Process all inbound text from a TurnContext.

    Two sources contribute this tick:
      - Player input: heard by every active agent. Each agent's emotional
        state shifts from what the player said.
      - NPC _speech events: attributed to the speaking agent. The act of
        speaking shifts the speaker's own state.

    All texts are batched for a single embed pass. Per-agent averages land
    as single harmonic updates.

    Args:
        turn_context:   The accumulated context for this turn.
        harmonic_state: The live state container to update in-place.

    Returns:
        Dict mapping agent_id → EmotionalDelta for each updated agent.
    """
    pairs: list[tuple[str, str]] = []

    # Player input is heard by all active agents
    if turn_context.player_input:
        for agent_id in turn_context.active_npc_ids:
            pairs.append((agent_id, turn_context.player_input))

    # NPC _speech events affect the speaking agent's own state.
    # Use the parsed speech text — embedding the full JSON (audios path,
    # companions list, etc.) pollutes the semagram with non-utterance noise.
    # Falls back to raw_data only if the parser failed to extract a clean
    # speech field, which should be vanishingly rare for valid _speech events.
    for agent_id, buf in turn_context.agent_buffers.items():
        for event in buf.events:
            if event.event_type != "_speech":
                continue
            speech = ""
            if event.parsed_data:
                speech = (event.parsed_data.get("speech") or "").strip()
            if not speech:
                speech = (event.raw_data or "").strip()
            if speech:
                pairs.append((agent_id, speech))

    if not pairs:
        return {}

    logger.debug(
        "Inbound pipeline: %d text-agent pairs across %d agents",
        len(pairs), len({a for a, _ in pairs}),
    )
    return process_texts(pairs, harmonic_state)


def process_outbound(
    responses: "list[AgentResponse]",
    harmonic_state: HarmonicState,
) -> dict[str, EmotionalDelta]:
    """Process LLM-generated utterances — emotional adoption.

    The agent's own words shift its state exactly as incoming events do.
    This closes the bidirectional loop: what an agent says becomes part of
    what it is. No distinction between "I felt this" and "I said this" —
    both update the harmonics, both become memory keys.

    Args:
        responses:      Parsed AgentResponse objects from this turn.
        harmonic_state: The live state container to update in-place.

    Returns:
        Dict mapping agent_id → EmotionalDelta for each agent that spoke.
    """
    pairs = [
        (resp.agent_id, resp.utterance)
        for resp in responses
        if resp.utterance
    ]
    if not pairs:
        return {}

    logger.debug("Outbound pipeline: %d utterances", len(pairs))
    return process_texts(pairs, harmonic_state)
