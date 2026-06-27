"""Tests for progeny.src.memory_compressor."""
from __future__ import annotations

from progeny.src.event_accumulator import TieredMemory
from progeny.src.memory_compressor import (
    COMPRESSED_MAX,
    KEYWORDS_MAX,
    VERBATIM_MAX,
    compress_entry,
    distill_keywords,
    slide_window,
)


# ---------------------------------------------------------------------------
# compress_entry
# ---------------------------------------------------------------------------

class TestCompressEntry:
    def test_user_role_becomes_player(self):
        entry = {"role": "user", "content": "Tell me about the war."}
        result = compress_entry(entry)
        assert result.startswith("Player:")

    def test_assistant_role_becomes_npc(self):
        entry = {"role": "assistant", "content": "I am sworn to carry your burdens."}
        result = compress_entry(entry)
        assert result.startswith("NPC:")

    def test_first_sentence_extracted(self):
        entry = {
            "role": "assistant",
            "content": "I'll handle this. Stay behind me, milord. We fight together.",
        }
        result = compress_entry(entry)
        assert "I'll handle this." in result
        # Second sentence should not be in the compressed form
        assert "Stay behind me" not in result

    def test_bracketed_annotations_preserved(self):
        entry = {
            "role": "assistant",
            "content": "Stand down! [Anger/Aggressive -> SheatheWeapon] I mean it.",
        }
        result = compress_entry(entry)
        assert "Anger/Aggressive -> SheatheWeapon" in result

    def test_long_content_truncated(self):
        entry = {"role": "user", "content": "a" * 200}
        result = compress_entry(entry)
        # Should not exceed a reasonable length
        assert len(result) < 100

    def test_empty_content(self):
        entry = {"role": "assistant", "content": ""}
        result = compress_entry(entry)
        assert result.startswith("NPC:")

    def test_retains_later_sentence_entity(self):
        """A salient entity past the first sentence survives the compression."""
        entry = {"role": "assistant", "content": "Hello there. Find me in Whiterun."}
        result = compress_entry(entry)
        assert "Hello there." in result      # first sentence kept
        assert "Whiterun" in result          # later-sentence entity retained

    def test_retains_later_sentence_emotion(self):
        """An emotion word past the first sentence survives the compression."""
        entry = {"role": "assistant", "content": "Stay back. I am afraid of him."}
        result = compress_entry(entry)
        assert "afraid" in result

    def test_no_residual_when_first_sentence_covers(self):
        """No tags appended when the first sentence already holds them all."""
        entry = {"role": "assistant", "content": "Lydia attacked."}
        result = compress_entry(entry)
        assert result == "NPC: Lydia attacked."

    def test_later_sentence_keyword_reaches_keywords_tier(self):
        """The retained entity flows through to the distilled keyword tier."""
        line = compress_entry(
            {"role": "assistant", "content": "Hello there. Find me in Whiterun."}
        )
        assert "Whiterun" in distill_keywords(line)


# ---------------------------------------------------------------------------
# distill_keywords
# ---------------------------------------------------------------------------

class TestDistillKeywords:
    def test_entity_extraction(self):
        compressed = "NPC: Lydia defended against Mikael at BanneredMare"
        result = distill_keywords(compressed)
        assert "Lydia" in result
        assert "Mikael" in result
        assert "BanneredMare" in result

    def test_emotion_pairing(self):
        compressed = "NPC: Lydia was protective of the player"
        result = distill_keywords(compressed)
        assert "Lydia:protective" in result

    def test_action_verb_extraction(self):
        compressed = "NPC: decided to attack the bandit and defend the camp"
        result = distill_keywords(compressed)
        assert "attack" in result
        assert "defend" in result

    def test_bracket_contents_as_tags(self):
        compressed = "NPC: Stood down [Anger -> Calm]"
        result = distill_keywords(compressed)
        assert "Anger -> Calm" in result

    def test_fallback_for_plain_text(self):
        compressed = "NPC: okay then"
        result = distill_keywords(compressed)
        # Should produce something, not empty
        assert len(result) > 0

    def test_pipe_delimited_format(self):
        compressed = "NPC: Lydia attacked Mikael in a brawl"
        result = distill_keywords(compressed)
        assert " | " in result


# ---------------------------------------------------------------------------
# slide_window
# ---------------------------------------------------------------------------

class TestSlideWindow:
    def _make_entry(self, n: int) -> dict:
        return {"role": "assistant", "content": f"Turn {n} response. More details here."}

    def test_no_slide_under_capacity(self):
        mem = TieredMemory(
            verbatim=[self._make_entry(i) for i in range(5)],
        )
        slide_window(mem)
        assert len(mem.verbatim) == 5
        assert len(mem.compressed) == 0
        assert len(mem.keywords) == 0

    def test_slide_verbatim_to_compressed(self):
        """When verbatim exceeds max, oldest entries compress."""
        mem = TieredMemory(
            verbatim=[self._make_entry(i) for i in range(12)],
        )
        slide_window(mem)
        assert len(mem.verbatim) == VERBATIM_MAX
        assert len(mem.compressed) == 4  # 12 - 8 = 4 compressed

    def test_compressed_overflow_to_keywords(self):
        """When compressed exceeds max, oldest entries distill to keywords."""
        mem = TieredMemory(
            verbatim=[self._make_entry(i) for i in range(10)],
            compressed=[f"NPC: Summary {i}." for i in range(9)],
        )
        # 10 verbatim → 8 remain, 2 compress → 9+2=11 compressed → 10 remain, 1 keyword
        slide_window(mem)
        assert len(mem.verbatim) == VERBATIM_MAX
        assert len(mem.compressed) == COMPRESSED_MAX
        assert len(mem.keywords) == 1

    def test_keywords_eviction(self):
        """When keywords exceed max, oldest evict."""
        mem = TieredMemory(
            verbatim=[self._make_entry(i) for i in range(10)],
            compressed=[f"NPC: Summary {i}." for i in range(10)],
            keywords=[f"tag{i}" for i in range(10)],
        )
        # 10 verbatim → 8 remain, 2 compress → 12 compressed → 10 remain, 2 distill
        # → 12 keywords → 10 remain (2 evicted)
        slide_window(mem)
        assert len(mem.verbatim) == VERBATIM_MAX
        assert len(mem.compressed) == COMPRESSED_MAX
        assert len(mem.keywords) == KEYWORDS_MAX

    def test_oldest_verbatim_compressed_first(self):
        """The oldest verbatim entries should be the ones that get compressed."""
        mem = TieredMemory(
            verbatim=[self._make_entry(i) for i in range(10)],
        )
        slide_window(mem)
        # Verbatim should retain the newest 8 (entries 2-9)
        assert "Turn 2" in mem.verbatim[0]["content"]
        assert "Turn 9" in mem.verbatim[-1]["content"]
        # Compressed should have the oldest 2 (entries 0-1)
        assert "Turn 0" in mem.compressed[0]
        assert "Turn 1" in mem.compressed[1]

    def test_full_30_turn_scenario(self):
        """Simulate 30 turns of dialogue — all tiers populated."""
        mem = TieredMemory()
        for i in range(30):
            mem.verbatim.append(self._make_entry(i))
            slide_window(mem)
        assert len(mem.verbatim) == VERBATIM_MAX
        assert len(mem.compressed) == COMPRESSED_MAX
        assert len(mem.keywords) == KEYWORDS_MAX


# ---------------------------------------------------------------------------
# TieredMemory + AgentBuffer integration
# ---------------------------------------------------------------------------

class TestAgentBufferCompat:
    def test_dialogue_history_reads_verbatim(self):
        from progeny.src.event_accumulator import AgentBuffer
        buf = AgentBuffer(agent_id="Lydia")
        buf.memory.verbatim.append({"role": "user", "content": "Hello"})
        assert len(buf.dialogue_history) == 1
        assert buf.dialogue_history[0]["content"] == "Hello"

    def test_dialogue_history_setter_writes_verbatim(self):
        from progeny.src.event_accumulator import AgentBuffer
        buf = AgentBuffer(agent_id="Lydia")
        buf.dialogue_history = [{"role": "user", "content": "Hi"}]
        assert len(buf.memory.verbatim) == 1

    def test_append_via_history_persists_in_memory(self):
        from progeny.src.event_accumulator import AgentBuffer
        buf = AgentBuffer(agent_id="Lydia")
        buf.dialogue_history.append({"role": "assistant", "content": "Greetings."})
        assert len(buf.memory.verbatim) == 1
