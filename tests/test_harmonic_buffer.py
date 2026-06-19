"""Tests for mindcore.harmonic_buffer."""
from __future__ import annotations

import numpy as np
import pytest

from shared.constants import EMOTIONAL_DIM, ZERO_SEMAGRAM
from mindcore.harmonic_buffer import (
    DynamicModulators,
    EmotionalDelta,
    HarmonicBuffer,
    HarmonicConfig,
    HarmonicState,
    build_modulators,
    configure,
)


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset to default config after each test."""
    configure(HarmonicConfig())
    yield
    configure(HarmonicConfig())


def _angry_semagram() -> list[float]:
    """A semagram with high anger (axis 1)."""
    sem = [0.0] * EMOTIONAL_DIM
    sem[1] = 0.8  # anger
    sem[0] = 0.1  # slight fear
    sem[8] = 0.3  # some residual
    return sem


def _calm_semagram() -> list[float]:
    """A semagram with high safety (axis 7) and joy (axis 6)."""
    sem = [0.0] * EMOTIONAL_DIM
    sem[6] = 0.6  # joy
    sem[7] = 0.7  # safety
    sem[8] = 0.2  # low residual
    return sem


# ---------------------------------------------------------------------------
# HarmonicBuffer
# ---------------------------------------------------------------------------

class TestHarmonicBuffer:
    def test_zero_init(self):
        buf = HarmonicBuffer()
        assert buf.fast.tolist() == ZERO_SEMAGRAM
        assert buf.medium.tolist() == ZERO_SEMAGRAM
        assert buf.slow.tolist() == ZERO_SEMAGRAM
        assert not buf._initialized

    def test_first_update_sets_all_traces(self):
        """First delta IS initial values — all traces set directly."""
        buf = HarmonicBuffer()
        angry = _angry_semagram()
        delta = buf.update(angry)

        np.testing.assert_allclose(buf.fast, angry, atol=1e-6)
        np.testing.assert_allclose(buf.medium, angry, atol=1e-6)
        np.testing.assert_allclose(buf.slow, angry, atol=1e-6)
        assert buf._initialized

    def test_first_update_delta_is_full_semagram(self):
        buf = HarmonicBuffer()
        angry = _angry_semagram()
        delta = buf.update(angry)
        np.testing.assert_allclose(delta.delta, angry, atol=1e-6)

    def test_ema_decay_fast_reacts_more(self):
        """Fast trace should move more toward new input than slow."""
        buf = HarmonicBuffer()
        buf.update(_angry_semagram())
        calm = _calm_semagram()
        buf.update(calm)

        # Fast (α=0.7) should be closer to calm than slow (α=0.1)
        fast_dist = np.linalg.norm(buf.fast - np.array(calm))
        slow_dist = np.linalg.norm(buf.slow - np.array(calm))
        assert fast_dist < slow_dist

    def test_curvature_positive_on_change(self):
        buf = HarmonicBuffer()
        buf.update(_angry_semagram())
        delta = buf.update(_calm_semagram())
        assert delta.curvature > 0

    def test_curvature_zero_on_same_input(self):
        """Repeated same semagram should drive curvature toward zero."""
        buf = HarmonicBuffer()
        sem = _angry_semagram()
        buf.update(sem)
        # Several updates with same input — curvature should shrink
        for _ in range(20):
            delta = buf.update(sem)
        assert delta.curvature < 0.01

    def test_snap_detects_phase_change(self):
        """Snap should be large when curvature changes abruptly."""
        buf = HarmonicBuffer()
        sem = _angry_semagram()
        buf.update(sem)
        # Repeated calm — curvature settling
        for _ in range(5):
            buf.update(sem)
        # Sudden shift to calm — curvature spikes, snap should be large
        delta = buf.update(_calm_semagram())
        assert abs(delta.snap) > 0.01

    def test_coherence_perfect_after_init(self):
        """After first update, all traces identical → coherence = 1.0."""
        buf = HarmonicBuffer()
        delta = buf.update(_angry_semagram())
        assert delta.coherence == pytest.approx(1.0, abs=0.001)

    def test_coherence_drops_on_shift(self):
        """After a big emotional shift, buffers diverge → coherence drops."""
        buf = HarmonicBuffer()
        buf.update(_angry_semagram())
        # Shift to calm — fast moves toward calm, slow stays near angry
        delta = buf.update(_calm_semagram())
        assert delta.coherence < 1.0  # Some divergence

    def test_lambda_high_on_first_encounter(self):
        """First update: high curvature + high snap → emotion-first retrieval.

        On the very first event, the agent is reacting to something new.
        λ should be high — "what just happened?" drives emotional recall.
        """
        buf = HarmonicBuffer()
        delta = buf.update(_angry_semagram())
        assert delta.lambda_t > 0.5  # Emotion-first on first encounter

    def test_lambda_low_in_stable_state(self):
        """After many same-input updates: low curvature, high coherence → low λ.

        The agent has settled into a stable state. Retrieval should favor
        domain/residual matching over emotional matching.
        """
        buf = HarmonicBuffer()
        sem = _angry_semagram()
        for _ in range(50):
            delta = buf.update(sem)
        # Curvature near zero, coherence near 1 → λ should be low
        assert delta.lambda_t < 0.5  # Residual-first in calm state

    def test_lambda_rises_on_volatile_shift(self):
        """Sudden shift after stable period: curvature spikes → λ rises."""
        buf = HarmonicBuffer()
        # Build stable angry state
        for _ in range(20):
            buf.update(_angry_semagram())
        stable_delta = buf.update(_angry_semagram())
        # Sudden shift to calm
        volatile_delta = buf.update(_calm_semagram())
        assert volatile_delta.lambda_t > stable_delta.lambda_t

    def test_get_semagram(self):
        buf = HarmonicBuffer()
        buf.update(_angry_semagram())
        sem = buf.get_semagram()
        assert len(sem) == EMOTIONAL_DIM
        assert sem == buf.fast.tolist()


# ---------------------------------------------------------------------------
# HarmonicState (multi-agent container)
# ---------------------------------------------------------------------------

class TestHarmonicState:
    def test_unknown_agent_returns_zero(self):
        state = HarmonicState()
        sem = state.get_semagram("Nobody")
        assert sem == ZERO_SEMAGRAM

    def test_update_creates_buffer(self):
        state = HarmonicState()
        delta = state.update("Lydia", _angry_semagram())
        assert isinstance(delta, EmotionalDelta)
        assert "Lydia" in state.agent_ids

    def test_multiple_agents_independent(self):
        state = HarmonicState()
        state.update("Lydia", _angry_semagram())
        state.update("Mikael", _calm_semagram())

        lydia_sem = state.get_semagram("Lydia")
        mikael_sem = state.get_semagram("Mikael")

        # Lydia should be angry-ish, Mikael calm-ish
        assert lydia_sem[1] > mikael_sem[1]  # anger
        assert mikael_sem[6] > lydia_sem[6]  # joy

    def test_reset_clears_all(self):
        state = HarmonicState()
        state.update("Lydia", _angry_semagram())
        state.update("Mikael", _calm_semagram())
        state.reset()
        assert state.agent_ids == []
        assert state.get_semagram("Lydia") == ZERO_SEMAGRAM

    def test_remove_agent(self):
        state = HarmonicState()
        state.update("Lydia", _angry_semagram())
        state.remove_agent("Lydia")
        assert state.get_semagram("Lydia") == ZERO_SEMAGRAM

    def test_get_delta_returns_none_for_unknown(self):
        state = HarmonicState()
        assert state.get_delta("Nobody") is None

    def test_get_delta_after_update(self):
        state = HarmonicState()
        state.update("Lydia", _angry_semagram())
        delta = state.get_delta("Lydia")
        assert delta is not None
        assert len(delta.semagram) == EMOTIONAL_DIM

    def test_get_delta_returns_exact_cached_result(self):
        """get_delta() returns the exact EmotionalDelta from the last update."""
        state = HarmonicState()
        update_delta = state.update("Lydia", _angry_semagram())
        cached_delta = state.get_delta("Lydia")
        assert cached_delta is update_delta  # Same object, not reconstructed


# ---------------------------------------------------------------------------
# Emotional momentum scenario
# ---------------------------------------------------------------------------

class TestPerAxisModulation:
    def test_default_alpha_is_uniform_9d(self):
        """Default alpha arrays should be uniform (same value per axis)."""
        buf = HarmonicBuffer()
        assert buf._alpha_fast.shape == (9,)
        assert buf._alpha_fast[0] == buf._alpha_fast[8]  # All same

    def test_per_axis_alpha_produces_asymmetric_tracking(self):
        """Different alpha per axis → some axes track faster than others.

        This is the infrastructure that dynamic modulators will use:
        Aggression gain on anger axis, Confidence damping on fear, etc.
        """
        buf = HarmonicBuffer()
        # Make anger axis (dim 1) track almost instantly on fast trace
        buf._alpha_fast[1] = 0.99
        # Make joy axis (dim 6) barely track on fast trace
        buf._alpha_fast[6] = 0.05

        angry = _angry_semagram()  # anger=0.8, joy=0.0
        buf.update(angry)
        # Now shift to calm (anger=0.0, joy=0.6)
        buf.update(_calm_semagram())

        # Anger axis (high alpha) should be close to new value (calm=0.0)
        assert buf.fast[1] < 0.15  # Tracked toward 0 quickly
        # Joy axis (low alpha) should barely have moved toward 0.6
        assert buf.fast[6] < 0.15  # Didn't track much from ~0.0


class TestEmotionalMomentum:
    def test_multi_turn_convergence(self):
        """After many updates with the same semagram, traces converge."""
        buf = HarmonicBuffer()
        sem = _angry_semagram()
        for _ in range(50):
            buf.update(sem)

        np.testing.assert_allclose(buf.fast, sem, atol=0.01)
        np.testing.assert_allclose(buf.medium, sem, atol=0.01)
        np.testing.assert_allclose(buf.slow, sem, atol=0.01)

    def test_slow_trace_resists_change(self):
        """Slow trace should barely move after a single contrary input."""
        buf = HarmonicBuffer()
        angry = _angry_semagram()
        # Build up angry state over many turns
        for _ in range(20):
            buf.update(angry)
        slow_before = buf.slow.copy()
        # One calm input
        buf.update(_calm_semagram())
        slow_after = buf.slow
        # Slow should have moved very little
        drift = np.linalg.norm(slow_after - slow_before)
        assert drift < 0.15  # α=0.1 * one step


# ---------------------------------------------------------------------------
# Dynamic Modulators — engine preset values as harmonic physics
# ---------------------------------------------------------------------------

class TestBuildModulators:
    """Test the build_modulators() constructor from raw engine values."""

    def test_default_values(self):
        mods = build_modulators()
        assert mods.aggression_gain == 0.0
        assert mods.confidence_damp == pytest.approx(0.5)  # default confidence=2 / 4
        assert mods.morality_threshold == 3
        assert mods.mood_axis is None  # Neutral
        assert mods.mood_pull == 0.0
        assert mods.assistance_coupling == 0.0

    def test_frenzied_aggression(self):
        mods = build_modulators(aggression=3)
        assert mods.aggression_gain == pytest.approx(1.0)

    def test_foolhardy_confidence(self):
        mods = build_modulators(confidence=4)
        assert mods.confidence_damp == pytest.approx(1.0)

    def test_cowardly_confidence(self):
        mods = build_modulators(confidence=0)
        assert mods.confidence_damp == 0.0

    def test_happy_mood_maps_to_joy_axis(self):
        mods = build_modulators(mood=3)
        assert mods.mood_axis == 6  # joy axis index
        assert mods.mood_pull > 0.0

    def test_angry_mood_maps_to_anger_axis(self):
        mods = build_modulators(mood=1)
        assert mods.mood_axis == 1  # anger axis index

    def test_neutral_mood_no_pull(self):
        mods = build_modulators(mood=0)
        assert mods.mood_axis is None
        assert mods.mood_pull == 0.0

    def test_full_assistance(self):
        mods = build_modulators(assistance=2)
        assert mods.assistance_coupling == pytest.approx(1.0)

    def test_clamping_out_of_range(self):
        """Out-of-range inputs get clamped to [0, 1]."""
        mods = build_modulators(aggression=10, confidence=10, assistance=10)
        assert mods.aggression_gain <= 1.0
        assert mods.confidence_damp <= 1.0
        assert mods.assistance_coupling <= 1.0


class TestDynamicModulators:
    """Test that modulators change the harmonic buffer dynamics."""

    def test_aggression_increases_fast_alpha_on_anger(self):
        """Frenzied NPC: anger axis tracks faster on fast trace."""
        buf = HarmonicBuffer()
        default_fast_anger = buf._alpha_fast[1]  # anger = dim 1
        buf.apply_modulators(build_modulators(aggression=3))
        assert buf._alpha_fast[1] > default_fast_anger

    def test_aggression_decreases_slow_alpha_on_anger(self):
        """Frenzied NPC: anger persists longer on slow trace."""
        buf = HarmonicBuffer()
        default_slow_anger = buf._alpha_slow[1]
        buf.apply_modulators(build_modulators(aggression=3))
        assert buf._alpha_slow[1] < default_slow_anger

    def test_aggression_affects_excitement_axis_too(self):
        """Aggression gain applies to both anger (1) and excitement (4)."""
        buf = HarmonicBuffer()
        default_fast_exc = buf._alpha_fast[4]
        buf.apply_modulators(build_modulators(aggression=3))
        assert buf._alpha_fast[4] > default_fast_exc

    def test_aggression_leaves_fear_axis_unchanged(self):
        """Aggression only modulates anger/excitement, not fear."""
        buf = HarmonicBuffer()
        default_fast_fear = buf._alpha_fast[0]
        buf.apply_modulators(build_modulators(aggression=3))
        assert buf._alpha_fast[0] == default_fast_fear

    def test_confidence_damping_attenuates_fear_delta(self):
        """Foolhardy NPC: fear delta is dampened in buffer update."""
        # Two buffers: one with damping, one without
        buf_brave = HarmonicBuffer()
        buf_brave.apply_modulators(build_modulators(confidence=4))  # Foolhardy
        buf_coward = HarmonicBuffer()
        buf_coward.apply_modulators(build_modulators(confidence=0))  # Cowardly

        # Initialize both with same neutral state
        neutral = [0.0] * EMOTIONAL_DIM
        buf_brave.update(neutral)
        buf_coward.update(neutral)

        # Hit with a fear-heavy semagram
        fearful = [0.0] * EMOTIONAL_DIM
        fearful[0] = 0.9  # fear axis
        buf_brave.update(fearful)
        buf_coward.update(fearful)

        # The Foolhardy NPC's fast buffer should show less fear
        assert buf_brave.fast[0] < buf_coward.fast[0]

    def test_confidence_damping_does_not_affect_anger(self):
        """Confidence damping only applies to fear axis, not anger."""
        buf_brave = HarmonicBuffer()
        buf_brave.apply_modulators(build_modulators(confidence=4))
        buf_no_mod = HarmonicBuffer()

        neutral = [0.0] * EMOTIONAL_DIM
        buf_brave.update(neutral)
        buf_no_mod.update(neutral)

        # Hit with anger (dim 1)
        angry = [0.0] * EMOTIONAL_DIM
        angry[1] = 0.9
        buf_brave.update(angry)
        buf_no_mod.update(angry)

        # Anger should be the same (confidence doesn't damp anger)
        assert buf_brave.fast[1] == pytest.approx(buf_no_mod.fast[1], abs=1e-6)

    def test_mood_pull_drifts_toward_bias(self):
        """Happy NPC: joy axis drifts toward the mood bias over time."""
        buf = HarmonicBuffer()
        buf.apply_modulators(build_modulators(mood=3))  # Happy → joy axis

        # Feed neutral inputs — mood pull should create drift
        neutral = [0.0] * EMOTIONAL_DIM
        for _ in range(100):
            buf.update(neutral)

        # Joy axis (dim 6) should have drifted positive
        assert buf.fast[6] > 0.01
        assert buf.slow[6] > 0.01

    def test_mood_pull_does_not_affect_unbiased_axes(self):
        """Mood pull only affects the mood axis, not others."""
        buf = HarmonicBuffer()
        buf.apply_modulators(build_modulators(mood=3))  # Happy → joy

        neutral = [0.0] * EMOTIONAL_DIM
        for _ in range(50):
            buf.update(neutral)

        # Fear axis (dim 0) should be essentially zero (no mood pull)
        assert abs(buf.fast[0]) < 0.001

    def test_neutral_mood_no_drift(self):
        """Neutral mood: no pull on any axis."""
        buf = HarmonicBuffer()
        buf.apply_modulators(build_modulators(mood=0))  # Neutral

        neutral = [0.0] * EMOTIONAL_DIM
        for _ in range(50):
            buf.update(neutral)

        # All axes should be near zero (no pull anywhere)
        for i in range(EMOTIONAL_DIM):
            assert abs(buf.fast[i]) < 0.001

    def test_apply_modulators_idempotent(self):
        """Calling apply_modulators twice resets to fresh state."""
        buf = HarmonicBuffer()
        buf.apply_modulators(build_modulators(aggression=3))
        fast_anger_first = buf._alpha_fast[1]
        buf.apply_modulators(build_modulators(aggression=3))
        assert buf._alpha_fast[1] == pytest.approx(fast_anger_first)

    def test_harmonic_state_apply_modulators_forwarding(self):
        """HarmonicState.apply_modulators() forwards to per-agent buffer."""
        state = HarmonicState()
        mods = build_modulators(aggression=3, mood=3)
        state.apply_modulators("Lydia", mods)
        buf = state._buffers["Lydia"]
        assert buf._modulators is not None
        assert buf._modulators.aggression_gain == pytest.approx(1.0)
        assert buf._modulators.mood_axis == 6

    def test_joker_profile_emergent_behavior(self):
        """The Joker Example: Confidence=4, Aggression=3, Mood=Happy.

        Fear signals barely register, anger/excitement amplify fast,
        joy drifts up during calm. Threat situations convert to excitement.
        Nobody scripted psychopath; the gain settings produced one.
        """
        buf = HarmonicBuffer()
        buf.apply_modulators(build_modulators(
            aggression=3, confidence=4, morality=0, mood=3, assistance=0,
        ))

        # Initialize with neutral
        neutral = [0.0] * EMOTIONAL_DIM
        buf.update(neutral)

        # Ambush: fear + anger + excitement
        ambush = [0.0] * EMOTIONAL_DIM
        ambush[0] = 0.8   # fear
        ambush[1] = 0.5   # anger
        ambush[4] = 0.6   # excitement
        buf.update(ambush)

        # The Joker: fear dampened, anger/excitement amplified
        fear_val = buf.fast[0]
        anger_val = buf.fast[1]
        excitement_val = buf.fast[4]

        # Fear should be significantly attenuated (Foolhardy damping)
        assert fear_val < 0.3  # raw would be ~0.56 with α=0.7
        # Anger and excitement should be strong (aggression gain)
        assert anger_val > 0.2
        assert excitement_val > 0.2


# ---------------------------------------------------------------------------
# Certainty modulation — LLM uncertainty → residual axis gain
# ---------------------------------------------------------------------------

class TestCertaintyModulation:
    """Test that certainty factor modulates the residual axis (dim 8).

    The LLM's genuine uncertainty (from token entropy) scales the residual
    axis of incoming semagrams. Full certainty preserves the signal;
    uncertainty attenuates it. Proportional relationships are preserved —
    only the gain changes. When uncertainty drops the residual, emotional
    axes relatively dominate the buffer: the NPC becomes more reactive,
    less grounded. Recovery is natural via EMA smoothing.
    """

    def test_default_certainty_is_one(self):
        """New buffers have full certainty (no modulation)."""
        buf = HarmonicBuffer()
        assert buf._certainty == 1.0

    def test_set_certainty_ema_smoothed(self):
        """Certainty is EMA-smoothed, not instantly overwritten."""
        buf = HarmonicBuffer()
        assert buf._certainty == 1.0  # default
        # Setting to 0.0: EMA blend = 0.3 * 0.0 + 0.7 * 1.0 = 0.7
        buf.set_certainty(0.0)
        assert buf._certainty == pytest.approx(0.7, abs=0.01)
        # Again: 0.3 * 0.0 + 0.7 * 0.7 = 0.49
        buf.set_certainty(0.0)
        assert buf._certainty < 0.7  # Continuing to drop

    def test_set_certainty_clamped_before_ema(self):
        """Out-of-range values are clamped before EMA blending."""
        buf = HarmonicBuffer()
        buf.set_certainty(1.5)  # clamped to 1.0, blended with 1.0 = 1.0
        assert buf._certainty == pytest.approx(1.0)
        buf.set_certainty(-0.5)  # clamped to 0.0, blended
        assert buf._certainty < 1.0  # moved toward 0

    def test_full_certainty_preserves_residual(self):
        """Certainty=1.0 → residual passes through unmodified."""
        buf = HarmonicBuffer()
        buf.set_certainty(1.0)
        sem = [0.0] * EMOTIONAL_DIM
        sem[8] = 0.5  # residual
        buf.update(sem)
        # Fast trace should reflect full residual (first update = direct set)
        assert buf.fast[8] == pytest.approx(0.5, abs=0.001)

    def test_low_certainty_attenuates_residual(self):
        """Low certainty → residual enters buffer at reduced strength."""
        buf_full = HarmonicBuffer()
        buf_full._certainty = 1.0  # bypass EMA for deterministic test
        buf_low = HarmonicBuffer()
        buf_low._certainty = 0.3   # direct set: 30% certainty

        # Initialize both with zero, then hit with residual
        neutral = [0.0] * EMOTIONAL_DIM
        t0 = 100.0
        buf_full.update(neutral, now=t0)
        buf_low.update(neutral, now=t0)

        sem = [0.0] * EMOTIONAL_DIM
        sem[8] = 0.8  # strong residual
        buf_full.update(sem, now=t0)
        buf_low.update(sem, now=t0)

        # Low-certainty buffer should have less residual
        assert buf_low.fast[8] < buf_full.fast[8]
        # At 30% certainty, the residual signal is heavily attenuated
        ratio = buf_low.fast[8] / buf_full.fast[8] if buf_full.fast[8] > 0 else 0
        assert ratio < 0.5  # Meaningfully attenuated

    def test_zero_certainty_suppresses_residual(self):
        """Certainty=0.0 → residual zeroed, emotions dominate."""
        buf = HarmonicBuffer()
        buf._certainty = 0.0  # direct set for deterministic test

        neutral = [0.0] * EMOTIONAL_DIM
        t0 = 100.0
        buf.update(neutral, now=t0)

        sem = [0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.9]
        buf.update(sem, now=t0)

        # Residual should be near zero (suppressed)
        assert abs(buf.fast[8]) < 0.05
        # But emotional axes should still be present
        assert buf.fast[0] > 0.01  # fear
        assert buf.fast[1] > 0.01  # anger

    def test_certainty_does_not_affect_emotional_axes(self):
        """Certainty modulation only touches dim 8, not dims 0-7."""
        buf_full = HarmonicBuffer()
        buf_full._certainty = 1.0
        buf_low = HarmonicBuffer()
        buf_low._certainty = 0.1  # direct set for deterministic test

        neutral = [0.0] * EMOTIONAL_DIM
        t0 = 100.0
        buf_full.update(neutral, now=t0)
        buf_low.update(neutral, now=t0)

        sem = _angry_semagram()  # anger=0.8, fear=0.1, residual=0.3
        buf_full.update(sem, now=t0)
        buf_low.update(sem, now=t0)

        # Emotional axes (0-7) should be identical regardless of certainty
        for dim in range(8):
            assert buf_full.fast[dim] == pytest.approx(buf_low.fast[dim], abs=1e-6), \
                f"dim {dim} differs: {buf_full.fast[dim]} vs {buf_low.fast[dim]}"

    def test_harmonic_state_set_certainty_forwarding(self):
        """HarmonicState.set_certainty() forwards to per-agent buffer."""
        state = HarmonicState()
        state.set_certainty("Lydia", 0.42)
        buf = state._buffers["Lydia"]
        # EMA: 0.3 * 0.42 + 0.7 * 1.0 = 0.826
        assert buf._certainty < 1.0
        assert buf._certainty > 0.42  # Smoothed toward default


# ---------------------------------------------------------------------------
# Temporal decay (cooling) — autonomous regression without stimulus
# ---------------------------------------------------------------------------

class TestTemporalDecay:
    """Test that traces decay proportional to elapsed time.

    The cooling formula: trace *= exp(-α * Δt / τ)
    Fast traces decay quickly. Slow traces barely move. Mood pull
    continues during silence. All tested with explicit timestamps
    to avoid flaky real-time dependencies.
    """

    def test_no_decay_without_time_elapsed(self):
        """cool() with dt=0 does nothing."""
        buf = HarmonicBuffer()
        t0 = 100.0
        buf.update(_angry_semagram(), now=t0)
        fast_before = buf.fast.copy()
        buf.cool(now=t0)  # dt = 0
        np.testing.assert_allclose(buf.fast, fast_before)

    def test_fast_decays_to_half_at_half_life(self):
        """Fast trace decays to ~50% at its half-life (5s)."""
        buf = HarmonicBuffer()
        t0 = 100.0
        buf.update(_angry_semagram(), now=t0)
        fast_before = buf.fast[1]
        # At half-life (5s): 0.5^(5/5) = 0.5
        buf.cool(now=t0 + 5.0)
        ratio = buf.fast[1] / fast_before if fast_before > 0 else 0
        assert ratio == pytest.approx(0.5, abs=0.05)

    def test_slow_barely_decays_in_short_time(self):
        """Slow trace (half-life=16s) barely moves in 3 seconds."""
        buf = HarmonicBuffer()
        t0 = 100.0
        buf.update(_angry_semagram(), now=t0)
        slow_before = buf.slow[1]
        # 3s: 0.5^(3/16) = 0.5^0.1875 ≈ 0.878
        buf.cool(now=t0 + 3.0)
        ratio = buf.slow[1] / slow_before if slow_before > 0 else 0
        assert ratio > 0.85

    def test_medium_decays_between_fast_and_slow(self):
        """Medium trace decays at an intermediate rate."""
        buf = HarmonicBuffer()
        t0 = 100.0
        buf.update(_angry_semagram(), now=t0)
        fast_before = buf.fast[1]
        medium_before = buf.medium[1]
        slow_before = buf.slow[1]
        buf.cool(now=t0 + 3.0)
        fast_ratio = buf.fast[1] / fast_before if fast_before > 0 else 0
        medium_ratio = buf.medium[1] / medium_before if medium_before > 0 else 0
        slow_ratio = buf.slow[1] / slow_before if slow_before > 0 else 0
        assert fast_ratio < medium_ratio < slow_ratio

    def test_update_applies_cooling_automatically(self):
        """update() calls cool() internally based on elapsed time."""
        buf = HarmonicBuffer()
        t0 = 100.0
        buf.update(_angry_semagram(), now=t0)
        anger_after_init = buf.fast[1]
        # Update 5 seconds later with zero input — should see decay
        # from cooling PLUS the EMA pull toward zero.
        neutral = [0.0] * EMOTIONAL_DIM
        buf.update(neutral, now=t0 + 5.0)
        # Should be much less than original (both cooling and EMA)
        assert buf.fast[1] < anger_after_init * 0.3

    def test_mood_pull_during_cooling(self):
        """Mood pull continues during silence — disposition persists."""
        buf = HarmonicBuffer()
        buf.apply_modulators(build_modulators(mood=3))  # Happy → joy (dim 6)
        t0 = 100.0
        # Initialize with neutral
        buf.update([0.0] * EMOTIONAL_DIM, now=t0)
        # Cool for 10 seconds without any events
        buf.cool(now=t0 + 10.0)
        # Joy axis should have drifted positive from mood pull
        assert buf.fast[6] > 0.001

    def test_cool_all_forwards_to_all_agents(self):
        """HarmonicState.cool_all() applies to every buffer."""
        state = HarmonicState()
        t0 = 100.0
        state.update("Lydia", _angry_semagram())
        state.update("Mikael", _calm_semagram())
        # Manually set timestamps for deterministic test
        state._buffers["Lydia"]._last_ts = t0
        state._buffers["Mikael"]._last_ts = t0
        lydia_before = state._buffers["Lydia"].fast[1]
        mikael_before = state._buffers["Mikael"].fast[6]
        state.cool_all(now=t0 + 5.0)
        # Both should have decayed
        assert state._buffers["Lydia"].fast[1] < lydia_before
        assert state._buffers["Mikael"].fast[6] < mikael_before
