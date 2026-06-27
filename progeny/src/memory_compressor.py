"""
Memory compressor for Progeny — extractive tiered compression.

Pure functions, no LLM dependency. Compresses dialogue history entries
through three tiers: verbatim → compressed → keywords → evict.

Compression is structural/extractive: speaker + first sentence + deltas +
actions collapse to a one-liner. Keywords distill entities, emotions, and
action verbs into pipe-delimited tags for future rehydration.

The keyword interpretation runs over the FULL original content at compress
time (``compress_entry``), and the salient tags it finds are appended to the
one-liner. Without that, first-sentence truncation would discard entities /
emotions / actions from later sentences before the keyword tier (which
distills from the one-liner, not the original) ever saw them.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from progeny.src.event_accumulator import TieredMemory

# Tier capacity limits
VERBATIM_MAX = 8
COMPRESSED_MAX = 10
KEYWORDS_MAX = 10

# Patterns for keyword extraction
_EMOTION_WORDS = frozenset({
    "anger", "angry", "fear", "afraid", "happy", "sad", "surprised",
    "disgusted", "calm", "hostile", "protective", "defiant", "apologetic",
    "nervous", "brave", "cowardly", "aggressive", "friendly", "worried",
    "suspicious", "grateful", "resentful", "proud", "ashamed", "excited",
})

_ACTION_VERBS = frozenset({
    "attack", "defend", "flee", "follow", "give", "take", "steal",
    "brawl", "surrender", "sheathe", "cast", "inspect", "search",
    "trade", "threaten", "warn", "heal", "block", "dodge", "charge",
})

# Matches capitalized words including CamelCase (Lydia, BanneredMare, WhiterunExterior)
_ENTITY_RE = re.compile(r"\b[A-Z][a-zA-Z]{2,}\b")

# Matches bracketed annotations like [Anger/Aggressive -> SheatheWeapon]
_BRACKET_RE = re.compile(r"\[([^\]]+)\]")


def _extract_tags(text: str) -> set[str]:
    """Extract structured tags from text: entities, entity:emotion pairings,
    standalone emotions, action verbs, and bracket annotations.

    Shared by ``compress_entry`` (run over the FULL original content) and
    ``distill_keywords`` (run over a compressed one-liner), so the same
    interpretation can be applied at either tier.
    """
    tokens: set[str] = set()
    text_lower = text.lower()

    # Entities — capitalized words (names, places), paired with any emotion
    # present in the text.
    entities = _ENTITY_RE.findall(text)
    for ent in entities:
        paired = False
        for emo in _EMOTION_WORDS:
            if emo in text_lower:
                tokens.add(f"{ent}:{emo}")
                paired = True
        if not paired:
            tokens.add(ent)

    # Standalone emotion words not already captured via an entity pairing.
    for emo in _EMOTION_WORDS:
        if emo in text_lower and not any(f":{emo}" in t for t in tokens):
            tokens.add(emo)

    # Action verbs.
    for verb in _ACTION_VERBS:
        if verb in text_lower:
            tokens.add(verb)

    # Bracket annotations as raw tags.
    for bracket_content in _BRACKET_RE.findall(text):
        tokens.add(bracket_content.strip())

    return tokens


def compress_entry(entry: dict) -> str:
    """Compress a verbatim dialogue entry into a one-line summary.

    Keeps the speaker, the first sentence, and any bracketed annotations, then
    appends salient tags distilled from the FULL content — entities, emotions,
    and actions that appear past the first sentence. Without this, first-sentence
    truncation would drop those tokens before the keyword tier ever saw them
    (the keyword tier distills from this line, not the original).

    Args:
        entry: Dict with 'role' and 'content' keys.

    Returns:
        One-line compressed summary string.
    """
    role = entry.get("role", "unknown")
    content = entry.get("content", "")

    speaker = "Player" if role == "user" else "NPC"
    first_sentence = _extract_first_sentence(content)

    brackets = _BRACKET_RE.findall(content)
    annotation = f" [{'; '.join(brackets)}]" if brackets else ""

    line = f"{speaker}: {first_sentence}{annotation}"

    # Retain salient tokens from the full content that the first-sentence cut
    # would otherwise discard (only those not already represented in the line).
    residual = _extract_tags(content) - _extract_tags(line)
    if residual:
        line += " | " + " | ".join(sorted(residual))

    return line


def distill_keywords(compressed: str) -> str:
    """Distill a compressed entry into pipe-delimited keyword tags.

    Extracts entity names, emotion words, and action verbs present in the
    compressed string (which already carries the full-content salient tags that
    ``compress_entry`` appended). Returns them as pipe-separated tags.

    Args:
        compressed: One-line compressed summary string.

    Returns:
        Pipe-delimited keyword string, e.g. "Lydia:protective | brawl | BanneredMare"
    """
    tokens = _extract_tags(compressed)

    if not tokens:
        # Fallback: first three significant words.
        words = [w for w in compressed.split() if len(w) > 3]
        tokens.update(words[:3])

    return " | ".join(sorted(tokens))


def slide_window(tiered: TieredMemory) -> None:
    """Slide the memory window in-place when verbatim exceeds capacity.

    Oldest verbatim entries compress and shift to compressed tier.
    Oldest compressed entries distill and shift to keywords tier.
    Oldest keywords evict (Phase 2: persist to Qdrant).

    Operates in-place on the TieredMemory instance.
    """
    # Shift overflow from verbatim → compressed
    while len(tiered.verbatim) > VERBATIM_MAX:
        oldest = tiered.verbatim.pop(0)
        tiered.compressed.append(compress_entry(oldest))

    # Shift overflow from compressed → keywords
    while len(tiered.compressed) > COMPRESSED_MAX:
        oldest = tiered.compressed.pop(0)
        tiered.keywords.append(distill_keywords(oldest))

    # Evict overflow from keywords
    while len(tiered.keywords) > KEYWORDS_MAX:
        tiered.keywords.pop(0)


def _extract_first_sentence(text: str) -> str:
    """Extract the first sentence from text, up to ~80 chars.

    Splits on sentence-ending punctuation (.!?) or takes the first
    80 characters if no sentence boundary is found.
    """
    # Match up to first sentence-ending punctuation
    match = re.match(r"^(.+?[.!?])\s", text)
    if match:
        sentence = match.group(1)
        return sentence[:80] if len(sentence) > 80 else sentence

    # No sentence boundary — truncate
    if len(text) > 80:
        return text[:77] + "..."
    return text
