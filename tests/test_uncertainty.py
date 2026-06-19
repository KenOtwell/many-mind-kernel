"""Tests for mindcore.uncertainty — per-agent certainty from token logprobs."""
from __future__ import annotations

import math

import pytest

from mindcore.uncertainty import (
    compute_certainty,
    _segment_certainty,
    _segment_by_agent,
    _is_structural,
)


# ---------------------------------------------------------------------------
# Helper: build mock token logprobs resembling llama.cpp output
# ---------------------------------------------------------------------------

def _tok(token: str, logprob: float) -> dict:
    """Convenience: build a single token logprob entry."""
    return {"token": token, "logprob": logprob}


def _build_single_agent_response(agent_id: str, logprobs: list[float]) -> list[dict]:
    """Build a realistic JSON token sequence for one agent.

    Simulates: {"responses": [{"agent_id": "Name", "utterance": "words"}]}
    with the given logprob values spread across semantic tokens.
    """
    # Structural tokens (high confidence, ~0 logprob)
    tokens = [
        _tok("{", -0.01),
        _tok('"responses"', -0.02),
        _tok(": [", -0.01),
        _tok("{", -0.01),
        _tok('"agent_id"', -0.01),
        _tok(": ", -0.01),
        _tok(f'"{agent_id}"', -0.02),
        _tok(", ", -0.01),
        _tok('"utterance"', -0.02),
        _tok(": ", -0.01),
        _tok('"', -0.01),
    ]
    # Semantic tokens with provided logprobs
    for i, lp in enumerate(logprobs):
        tokens.append(_tok(f"word{i}", lp))
        if i < len(logprobs) - 1:
            tokens.append(_tok(" ", -0.01))
    # Closing structural tokens
    tokens.extend([
        _tok('"', -0.01),
        _tok("}", -0.01),
        _tok("]", -0.01),
        _tok("}", -0.01),
    ])
    return tokens


def _build_multi_agent_response(
    agents: list[tuple[str, list[float]]],
) -> list[dict]:
    """Build a realistic JSON token sequence for multiple agents.

    Args:
        agents: List of (agent_id, logprob_values) tuples.
    """
    tokens = [
        _tok("{", -0.01),
        _tok('"responses"', -0.02),
        _tok(": [", -0.01),
    ]
    for idx, (agent_id, logprobs) in enumerate(agents):
        if idx > 0:
            tokens.append(_tok(", ", -0.01))
        tokens.extend([
            _tok("{", -0.01),
            _tok('"agent_id"', -0.01),
            _tok(": ", -0.01),
            _tok(f'"{agent_id}"', -0.02),
            _tok(", ", -0.01),
            _tok('"utterance"', -0.02),
            _tok(": ", -0.01),
            _tok('"', -0.01),
        ])
        for i, lp in enumerate(logprobs):
            tokens.append(_tok(f"word{i}", lp))
            if i < len(logprobs) - 1:
                tokens.append(_tok(" ", -0.01))
        tokens.extend([
            _tok('"', -0.01),
            _tok("}", -0.01),
        ])
    tokens.extend([
        _tok("]", -0.01),
        _tok("}", -0.01),
    ])
    return tokens


# ---------------------------------------------------------------------------
# _segment_certainty
# ---------------------------------------------------------------------------

class TestStructuralFiltering:
    """Structural token filter correctly identifies grammar-forced tokens."""

    def test_json_punctuation_is_structural(self):
        for tok in ["{", "}", "[", "]", ",", ":", '"', " "]:
            assert _is_structural(tok), f"{tok!r} should be structural"

    def test_field_names_are_structural(self):
        for tok in ['"agent_id"', '"utterance"', '"actor_value_deltas"']:
            assert _is_structural(tok), f"{tok!r} should be structural"

    def test_semantic_words_not_structural(self):
        for tok in ["hello", "sword", "afraid", "3", "Lydia"]:
            assert not _is_structural(tok), f"{tok!r} should not be structural"

    def test_agent_name_not_structural(self):
        # Agent names in quotes are NOT field names — they're the model's choice
        assert not _is_structural('"Lydia"')
        assert not _is_structural('"Nazeem"')


class TestSegmentCertainty:
    """Tests use semantic (non-structural) token strings so the filter keeps them."""

    def test_empty_tokens(self):
        assert _segment_certainty([]) == 1.0

    def test_perfect_certainty(self):
        """All logprobs 0 → certainty = exp(0) = 1.0."""
        tokens = [_tok("hello", 0.0), _tok("world", 0.0)]
        assert _segment_certainty(tokens) == pytest.approx(1.0)

    def test_moderate_certainty(self):
        """Mean logprob = -0.5 → certainty ≈ 0.607."""
        tokens = [_tok("afraid", -0.5), _tok("sword", -0.5)]
        assert _segment_certainty(tokens) == pytest.approx(math.exp(-0.5), abs=0.001)

    def test_low_certainty(self):
        """Very negative logprobs → certainty near 0."""
        tokens = [_tok("perhaps", -5.0), _tok("maybe", -5.0)]
        result = _segment_certainty(tokens)
        assert result < 0.01
        assert result >= 0.0

    def test_mixed_logprobs(self):
        """Mean of mixed logprobs."""
        tokens = [_tok("yes", -0.1), _tok("uncertain", -0.9)]
        mean_lp = (-0.1 + -0.9) / 2
        assert _segment_certainty(tokens) == pytest.approx(math.exp(mean_lp), abs=0.001)

    def test_structural_tokens_filtered_out(self):
        """Structural tokens don't contribute to the certainty metric."""
        # Only semantic tokens with -2.0 should count
        tokens = [
            _tok("{", -0.01),        # structural — filtered
            _tok('"utterance"', -0.01), # structural — filtered
            _tok("afraid", -2.0),    # semantic — kept
            _tok("}", -0.01),        # structural — filtered
        ]
        result = _segment_certainty(tokens)
        # Should be exp(-2.0), not diluted by the structural tokens
        assert result == pytest.approx(math.exp(-2.0), abs=0.01)

    def test_all_structural_returns_default(self):
        """Segment with only structural tokens returns 1.0."""
        tokens = [_tok("{", -0.01), _tok("}", -0.01)]
        assert _segment_certainty(tokens) == 1.0

    def test_result_always_in_unit_interval(self):
        """Certainty is always in [0, 1]."""
        for lp in [0.0, -0.001, -0.5, -1.0, -5.0, -50.0]:
            tokens = [_tok("word", lp)]
            c = _segment_certainty(tokens)
            assert 0.0 <= c <= 1.0, f"logprob={lp} gave certainty={c}"


# ---------------------------------------------------------------------------
# _segment_by_agent
# ---------------------------------------------------------------------------

class TestSegmentByAgent:
    def test_single_agent(self):
        tokens = _build_single_agent_response("Lydia", [-0.3, -0.4])
        segments = _segment_by_agent(tokens, ["Lydia"])
        assert "Lydia" in segments
        assert len(segments["Lydia"]) > 0

    def test_multi_agent_segmentation(self):
        tokens = _build_multi_agent_response([
            ("Lydia", [-0.2, -0.3]),
            ("Nazeem", [-0.8, -0.9]),
        ])
        segments = _segment_by_agent(tokens, ["Lydia", "Nazeem"])
        assert "Lydia" in segments
        assert "Nazeem" in segments
        # Both should have tokens
        assert len(segments["Lydia"]) > 0
        assert len(segments["Nazeem"]) > 0
        # Segments should not overlap
        total = len(segments["Lydia"]) + len(segments["Nazeem"])
        assert total <= len(tokens)

    def test_unknown_agent_not_in_segments(self):
        tokens = _build_single_agent_response("Lydia", [-0.3])
        segments = _segment_by_agent(tokens, ["Lydia", "Ghost"])
        assert "Lydia" in segments
        assert "Ghost" not in segments

    def test_empty_tokens(self):
        segments = _segment_by_agent([], ["Lydia"])
        assert segments == {}


# ---------------------------------------------------------------------------
# compute_certainty (public API)
# ---------------------------------------------------------------------------

class TestComputeCertainty:
    def test_no_logprobs_returns_default(self):
        """No logprobs available → all agents get certainty 1.0."""
        result = compute_certainty(None, ["Lydia", "Nazeem"])
        assert result == {"Lydia": 1.0, "Nazeem": 1.0}

    def test_empty_logprobs_returns_default(self):
        result = compute_certainty([], ["Lydia"])
        assert result == {"Lydia": 1.0}

    def test_empty_agent_ids(self):
        result = compute_certainty([_tok("x", -0.5)], [])
        assert result == {}

    def test_solo_agent_skips_segmentation(self):
        """Single agent gets all tokens — no segmentation needed."""
        tokens = _build_single_agent_response("Lydia", [-0.3, -0.4, -0.5])
        result = compute_certainty(tokens, ["Lydia"])
        assert "Lydia" in result
        assert 0.0 < result["Lydia"] < 1.0

    def test_multi_agent_different_certainty(self):
        """Confident agent vs uncertain agent should produce different values.

        With structural token filtering, the semantic tokens dominate the
        certainty metric. The confident agent's word tokens have near-zero
        logprobs; the uncertain agent's have very negative logprobs.
        """
        tokens = _build_multi_agent_response([
            ("Lydia", [-0.05, -0.05, -0.05]),   # very confident
            ("Nazeem", [-2.0, -2.5, -3.0]),      # very uncertain
        ])
        result = compute_certainty(tokens, ["Lydia", "Nazeem"])
        # Relative ordering is the key signal
        assert result["Lydia"] > result["Nazeem"]
        # With filtering, confident semantic tokens dominate Lydia's mean
        assert result["Lydia"] > 0.8
        # With filtering, Nazeem's uncertain tokens aren't diluted
        assert result["Nazeem"] < 0.3

    def test_missing_agent_gets_default(self):
        """Agent not in LLM output gets certainty 1.0."""
        tokens = _build_single_agent_response("Lydia", [-0.3])
        result = compute_certainty(tokens, ["Lydia", "Ghost"])
        assert result["Lydia"] < 1.0
        assert result["Ghost"] == 1.0

    def test_certainty_bounds(self):
        """All certainty values must be in [0, 1]."""
        tokens = _build_multi_agent_response([
            ("A", [0.0, 0.0]),        # perfect
            ("B", [-0.5, -0.5]),      # moderate
            ("C", [-10.0, -10.0]),    # extremely uncertain
        ])
        result = compute_certainty(tokens, ["A", "B", "C"])
        for agent_id, c in result.items():
            assert 0.0 <= c <= 1.0, f"{agent_id} certainty={c} out of bounds"
