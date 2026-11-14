"""Tests for progeny.src.response_expander."""
from __future__ import annotations

import json

from shared.constants import EMOTIONAL_DIM
from shared.schemas import ExtractionLevel
from progeny.src.response_expander import expand_response


# ---------------------------------------------------------------------------
# Valid LLM response JSON
# ---------------------------------------------------------------------------

def _lydia_response_json() -> str:
    return json.dumps({
        "responses": [
            {
                "agent_id": "Lydia",
                "utterance": "Something's not right. Stay behind me.",
                "actor_value_deltas": {"Aggression": 2, "Confidence": 3, "Mood": 1},
                "actions": [{"command": "Follow", "target": "Player"}],
                "updated_harmonics": {
                    "base_vector": [0.1, 0.6, 0.2, 0.0, 0.3, 0.0, 0.1, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.4]
                },
                "new_memories": [
                    {"text": "Sensed danger near the market. Moved to protect."}
                ],
            }
        ]
    })


def _multi_agent_json() -> str:
    return json.dumps({
        "responses": [
            {
                "agent_id": "Lydia",
                "utterance": "Stay behind me.",
                "actor_value_deltas": {"Aggression": 2, "Confidence": 3},
            },
            {
                "agent_id": "Belethor",
                "utterance": "Do come back...",
                "actor_value_deltas": {"Mood": 3},
            },
            {
                "agent_id": "Heimskr",
                "actor_value_deltas": {"Mood": 1},
            },
        ]
    })


# ---------------------------------------------------------------------------
# Strict JSON parsing
# ---------------------------------------------------------------------------

class TestStrictParse:
    def test_single_agent_full_response(self):
        responses = expand_response(_lydia_response_json(), ["Lydia"])
        assert len(responses) == 1
        r = responses[0]
        assert r.agent_id == "Lydia"
        assert r.utterance == "Something's not right. Stay behind me."
        assert r.extraction_level == ExtractionLevel.STRICT

    def test_actor_value_deltas_parsed(self):
        responses = expand_response(_lydia_response_json(), ["Lydia"])
        avd = responses[0].actor_value_deltas
        assert avd is not None
        assert avd.Aggression == 2
        assert avd.Confidence == 3
        assert avd.Mood == 1

    def test_actions_parsed(self):
        responses = expand_response(_lydia_response_json(), ["Lydia"])
        actions = responses[0].actions
        assert len(actions) == 1
        assert actions[0].command == "Follow"
        assert actions[0].target == "Player"

    def test_harmonics_parsed(self):
        responses = expand_response(_lydia_response_json(), ["Lydia"])
        h = responses[0].updated_harmonics
        assert h is not None
        assert len(h.base_vector) == EMOTIONAL_DIM

    def test_new_memories_parsed(self):
        responses = expand_response(_lydia_response_json(), ["Lydia"])
        mems = responses[0].new_memories
        assert len(mems) == 1
        assert "danger" in mems[0].text

    def test_multi_agent(self):
        responses = expand_response(
            _multi_agent_json(),
            ["Lydia", "Belethor", "Heimskr"],
        )
        assert len(responses) == 3
        assert responses[0].agent_id == "Lydia"
        assert responses[1].agent_id == "Belethor"
        assert responses[2].agent_id == "Heimskr"
        assert responses[2].utterance is None  # Heimskr had no utterance

    def test_missing_agent_gets_empty_response(self):
        """LLM skipped an agent — graceful degradation."""
        raw = json.dumps({"responses": [{"agent_id": "Lydia", "utterance": "Hi"}]})
        responses = expand_response(raw, ["Lydia", "Belethor"])
        assert len(responses) == 2
        assert responses[1].agent_id == "Belethor"
        assert responses[1].utterance is None


# ---------------------------------------------------------------------------
# Validation and clamping
# ---------------------------------------------------------------------------

class TestValidation:
    def test_clamp_aggression_above_max(self):
        raw = json.dumps({
            "responses": [{"agent_id": "Lydia", "actor_value_deltas": {"Aggression": 10}}]
        })
        responses = expand_response(raw, ["Lydia"])
        assert responses[0].actor_value_deltas.Aggression == 3  # Max is 3

    def test_clamp_confidence_below_min(self):
        raw = json.dumps({
            "responses": [{"agent_id": "Lydia", "actor_value_deltas": {"Confidence": -5}}]
        })
        responses = expand_response(raw, ["Lydia"])
        assert responses[0].actor_value_deltas.Confidence == 0  # Min is 0

    def test_unknown_command_stripped(self):
        raw = json.dumps({
            "responses": [{
                "agent_id": "Lydia",
                "actions": [
                    {"command": "Follow", "target": "Player"},
                    {"command": "FlyToMoon", "target": "Masser"},  # Invalid
                ],
            }]
        })
        responses = expand_response(raw, ["Lydia"])
        assert len(responses[0].actions) == 1
        assert responses[0].actions[0].command == "Follow"

    def test_invalid_harmonics_ignored(self):
        raw = json.dumps({
            "responses": [{
                "agent_id": "Lydia",
                "updated_harmonics": {"base_vector": [1, 2, 3]},  # Wrong length
            }]
        })
        responses = expand_response(raw, ["Lydia"])
        assert responses[0].updated_harmonics is None

    def test_empty_utterance_becomes_none(self):
        raw = json.dumps({
            "responses": [{"agent_id": "Lydia", "utterance": "   "}]
        })
        responses = expand_response(raw, ["Lydia"])
        assert responses[0].utterance is None

    def test_no_actor_values_returns_none(self):
        raw = json.dumps({
            "responses": [{"agent_id": "Lydia", "actor_value_deltas": {}}]
        })
        responses = expand_response(raw, ["Lydia"])
        assert responses[0].actor_value_deltas is None


# ---------------------------------------------------------------------------
# LLM output repair (fence stripping, trailing commas)
# ---------------------------------------------------------------------------

class TestRepairPass:
    def test_markdown_json_fence_stripped(self):
        """Models often wrap JSON in ```json ... ``` fences."""
        inner = json.dumps({"responses": [{"agent_id": "Lydia", "utterance": "Hello."}]})
        fenced = f"```json\n{inner}\n```"
        responses = expand_response(fenced, ["Lydia"])
        assert responses[0].utterance == "Hello."
        assert responses[0].extraction_level == ExtractionLevel.STRICT

    def test_bare_fence_stripped(self):
        """Some models use bare ``` without json tag."""
        inner = json.dumps({"responses": [{"agent_id": "Lydia", "utterance": "Hi."}]})
        fenced = f"```\n{inner}\n```"
        responses = expand_response(fenced, ["Lydia"])
        assert responses[0].utterance == "Hi."
        assert responses[0].extraction_level == ExtractionLevel.STRICT

    def test_fence_with_surrounding_text(self):
        """Some models add preamble before the fence."""
        inner = json.dumps({"responses": [{"agent_id": "Lydia", "utterance": "Greetings."}]})
        wrapped = f"Here is the response:\n```json\n{inner}\n```\nDone."
        responses = expand_response(wrapped, ["Lydia"])
        assert responses[0].utterance == "Greetings."
        assert responses[0].extraction_level == ExtractionLevel.STRICT

    def test_trailing_comma_fixed(self):
        """Trailing commas before } or ] are a common LLM mistake."""
        raw = '{"responses": [{"agent_id": "Lydia", "utterance": "Hey.",}]}'
        responses = expand_response(raw, ["Lydia"])
        assert responses[0].utterance == "Hey."
        assert responses[0].extraction_level == ExtractionLevel.STRICT

    def test_trailing_comma_in_array(self):
        raw = '{"responses": [{"agent_id": "Lydia", "actions": [{"command": "Follow", "target": "Player"},]},]}'
        responses = expand_response(raw, ["Lydia"])
        assert responses[0].actions[0].command == "Follow"


# ---------------------------------------------------------------------------
# Plaintext fallback
# ---------------------------------------------------------------------------

class TestPlaintextFallback:
    def test_non_json_becomes_utterance(self):
        responses = expand_response(
            "I am sworn to carry your burdens.",
            ["Lydia"],
        )
        assert len(responses) == 1
        assert responses[0].utterance == "I am sworn to carry your burdens."
        assert responses[0].extraction_level == ExtractionLevel.PLAINTEXT

    def test_plaintext_only_first_agent_gets_utterance(self):
        responses = expand_response(
            "Something happened.",
            ["Lydia", "Belethor"],
        )
        assert responses[0].utterance == "Something happened."
        assert responses[1].utterance is None

    def test_empty_text_no_utterance(self):
        responses = expand_response("", ["Lydia"])
        assert responses[0].utterance is None

    def test_json_without_responses_key_falls_back(self):
        raw = json.dumps({"dialogue": "Hello there"})
        responses = expand_response(raw, ["Lydia"])
        # Should fall back to plaintext since no "responses" key
        assert responses[0].extraction_level == ExtractionLevel.PLAINTEXT
