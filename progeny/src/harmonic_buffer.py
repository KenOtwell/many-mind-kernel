"""
Harmonic buffer for Progeny — per-agent emotional state with temporal depth.

Each agent maintains three EMA (exponential moving average) traces over the
9d emotional semagram, at different time-scales:

  Fast  (α=0.7): reactive — captures immediate emotional shifts
  Medium (α=0.3): smoothed — captures mood trends across turns
  Slow  (α=0.1): glacial — captures personality-level drift

Derived signals:
  curvature: magnitude of delta between consecutive fast values.
             The priority gradient — how fast is emotion changing?
  snap:      change in curvature (2nd derivative).
             Event boundary detector — marks emotional phase transitions.
  coherence: cross-buffer agreement across all 9 dimensions.
             Per-dimension variance across fast/medium/slow, mapped to [0,1].
             High = stable state across timescales. Low = volatile transition.
  λ(t):      retrieval balance — emotional vs. residual search weighting.
             σ(α·curvature + β·|snap| - γ·coherence)
             High (~1.0) = emotion-first recall (episodes, grudges).
             Low  (~0.0) = residual-first recall (domain knowledge, tactics).
             Driven by volatility and event boundaries, damped by stability.

Zero-init pattern: new agents start at ZERO_SEMAGRAM. The first emotional
update IS the initial value — no separate initialization needed.
"""
from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field

import numpy as np

from shared.constants import EMOTIONAL_DIM, MOOD_TO_AXIS, ZERO_SEMAGRAM

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Dynamic modulator tuning constants
# ---------------------------------------------------------------------------

# Aggression: gain multiplier on anger (dim 1) and excitement (dim 4) axes.
# Fast alpha scales UP (tracks anger/excitement faster when provoked).
# Slow alpha scales DOWN (anger/excitement decay slower — slow to cool off).
K_AGG: float = 0.4
K_AGG_PERSIST: float = 0.3
_AGG_AXES: tuple[int, int] = (1, 4)  # anger, excitement

# Confidence: damping on fear (dim 0) effective delta magnitude.
# At Confidence=4 (Foolhardy), fear deltas attenuated to ~20% of raw.
K_CONF: float = 0.8
_CONF_AXES: tuple[int, ...] = (0,)  # fear

# Mood: ambient bias pull strength and target value.
# Gentle drift — doesn't override strong events, just nudges during calm.
DEFAULT_MOOD_PULL: float = 0.03
MOOD_BIAS_VALUE: float = 0.3  # target value on the biased axis


@dataclass
class DynamicModulators:
    """Engine preset values as dynamic modulators for harmonic buffer physics.

    These shape HOW emotional signals propagate — gains, dampings, biases,
    thresholds — not WHERE the agent sits in emotional space. Two NPCs
    receiving identical events diverge immediately because the same stimulus
    propagates differently through differently-tuned dynamics.

    Constructed from Creation Engine actor values via build_modulators().
    See Living Doc §Engine Preset Values as Dynamic Modulators.
    """
    aggression_gain: float = 0.0      # 0.0-1.0, from normalize(Aggression, 0, 3)
    confidence_damp: float = 0.0      # 0.0-1.0, from normalize(Confidence, 0, 4)
    morality_threshold: int = 3       # 0-3 integer (for response_expander action filtering)
    mood_axis: int | None = None      # EMOTIONAL_AXES index, or None (Neutral/Puzzled)
    mood_pull: float = 0.0            # ambient drift strength
    assistance_coupling: float = 0.0  # 0.0-1.0, for future cross-agent bleed


def build_modulators(
    aggression: int = 0,
    confidence: int = 2,
    morality: int = 3,
    mood: int = 0,
    assistance: int = 0,
) -> DynamicModulators:
    """Construct DynamicModulators from raw Creation Engine actor values.

    Normalizes integer ranges to [0, 1] floats for the continuous
    modulator parameters. Mood maps to an axis index via MOOD_TO_AXIS.

    Args:
        aggression: 0=Unaggressive .. 3=Frenzied
        confidence: 0=Cowardly .. 4=Foolhardy
        morality:   0=Any crime .. 3=No crime
        mood:       0=Neutral .. 7=Disgusted
        assistance: 0=Nobody .. 2=Friends and allies
    """
    mood_axis = MOOD_TO_AXIS.get(mood)
    return DynamicModulators(
        aggression_gain=max(0.0, min(1.0, aggression / 3.0)),
        confidence_damp=max(0.0, min(1.0, confidence / 4.0)),
        morality_threshold=max(0, min(3, morality)),
        mood_axis=mood_axis,
        mood_pull=DEFAULT_MOOD_PULL if mood_axis is not None else 0.0,
        assistance_coupling=max(0.0, min(1.0, assistance / 2.0)),
    )


@dataclass
class HarmonicConfig:
    """EMA decay rates, coherence scaling, and λ(t) retrieval balance gains."""
    alpha_fast: float = 0.7
    alpha_medium: float = 0.3
    alpha_slow: float = 0.1
    # Per-trace half-life (seconds): time for a trace to decay to 50%
    # of its value without new input. Direct tuning knobs.
    # Decay formula: trace *= 0.5^(Δt / half_life)
    half_life_fast: float = 5.0     # reactive: flinch fades in ~5s
    half_life_medium: float = 8.0   # mood: decision-scale persistence
    half_life_slow: float = 16.0    # personality: long ambient imprint
    # Certainty EMA smoothing rate. Prevents a single uncertain tick
    # from crashing the residual. 0.3 = 30% new + 70% previous.
    certainty_alpha: float = 0.3
    # Cross-buffer coherence sensitivity — higher = more discriminating.
    # Controls how quickly coherence drops as buffer traces diverge.
    coherence_scale: float = 10.0
    # λ(t) retrieval balance gains (per-agent personality parameters).
    # α: curvature gain — emotional volatility pushes toward emotion-first.
    # β: snap gain — event boundaries push toward emotion-first.
    # γ: coherence gain — cross-buffer stability pushes toward residual-first.
    lambda_alpha: float = 3.0
    lambda_beta: float = 2.0
    lambda_gamma: float = 2.0
    # LLM harmonics blend weight — controls how much the LLM's proposed
    # emotional evaluation influences the buffer vs the mechanical pipeline.
    # Pass 1 (mechanical): text → embed → 9d → EMA (fast, context-blind).
    # Pass 2 (LLM): contextual evaluation → blended into buffer.
    # 0.0 = ignore LLM proposals. 1.0 = fully trust LLM. 0.3 = 30% LLM.
    llm_harmonics_blend: float = 0.3


# Module-level default config
_config = HarmonicConfig()


def configure(config: HarmonicConfig) -> None:
    """Override the module-level harmonic config."""
    global _config
    _config = config


@dataclass
class EmotionalDelta:
    """Result of a harmonic buffer update — the emotional change signal.

    Passed to the prompt formatter so the LLM can calibrate response
    intensity, and to the retrieval pipeline for λ-weighted search.
    """
    semagram: list[float]    # Current 9d state (fast trace)
    delta: list[float]       # 9d change vector (new - previous fast)
    curvature: float         # Magnitude of delta (priority gradient)
    snap: float              # Change in curvature (event boundary)
    coherence: float         # Cross-buffer agreement (0=volatile, 1=stable)
    lambda_t: float          # Retrieval balance (0=residual-first, 1=emotion-first)


@dataclass
class HarmonicBuffer:
    """Per-agent emotional state with three EMA time-scales.

    All traces are 9d numpy arrays matching EMOTIONAL_AXES order.
    Zero-init: all traces start at zero. First update sets initial values.

    Per-axis EMA rates (_alpha_fast/medium/slow) are 9d vectors, initialized
    from the scalar config defaults. Dynamic modulators adjust individual
    axes: Aggression gain on anger/excitement, Confidence damping on fear,
    Mood bias pull, etc. See Living Doc §Engine Preset Values as Dynamic
    Modulators.
    """
    fast: np.ndarray = field(default_factory=lambda: np.zeros(EMOTIONAL_DIM, dtype=np.float32))
    medium: np.ndarray = field(default_factory=lambda: np.zeros(EMOTIONAL_DIM, dtype=np.float32))
    slow: np.ndarray = field(default_factory=lambda: np.zeros(EMOTIONAL_DIM, dtype=np.float32))
    # Per-axis EMA rates — 9d vectors for per-axis modulation.
    # Default: uniform rates from HarmonicConfig scalars.
    _alpha_fast: np.ndarray = field(
        default_factory=lambda: np.full(EMOTIONAL_DIM, _config.alpha_fast, dtype=np.float32),
    )
    _alpha_medium: np.ndarray = field(
        default_factory=lambda: np.full(EMOTIONAL_DIM, _config.alpha_medium, dtype=np.float32),
    )
    _alpha_slow: np.ndarray = field(
        default_factory=lambda: np.full(EMOTIONAL_DIM, _config.alpha_slow, dtype=np.float32),
    )
    prev_curvature: float = 0.0
    _initialized: bool = False
    _last_delta: EmotionalDelta | None = field(default=None, repr=False)
    _modulators: DynamicModulators | None = field(default=None, repr=False)
    # Certainty factor from LLM token entropy — modulates residual axis.
    # 1.0 = fully confident (no modulation). 0.0 = total uncertainty
    # (residual zeroed — reality signal suppressed, emotions dominate).
    # EMA-smoothed: one uncertain tick doesn't crash the residual.
    _certainty: float = 1.0
    # Timestamp of last update or cool — for temporal decay.
    _last_ts: float = field(default_factory=_time.monotonic)

    def set_certainty(self, certainty: float) -> None:
        """Set the certainty factor from LLM uncertainty estimation.

        EMA-smoothed to prevent a single uncertain tick from crashing
        the residual. The smoothing rate (certainty_alpha) controls
        how fast certainty responds: 0.3 = 30% new + 70% previous.

        Args:
            certainty: Raw factor in [0, 1]. Clamped then smoothed.
        """
        raw = max(0.0, min(1.0, certainty))
        alpha = _config.certainty_alpha
        self._certainty = alpha * raw + (1.0 - alpha) * self._certainty

    def cool(self, now: float | None = None) -> None:
        """Autonomous temporal decay — emotions attenuate without stimulus.

        Applies continuous-time exponential decay based on elapsed time
        since the last update or cool. Each trace has its own half-life
        (seconds to decay to 50%):

            trace *= 0.5^(Δt / half_life)

        Fast traces decay quickly (the flinch fades). Slow traces barely
        move (the personality imprint holds). Mood pull continues during
        silence — disposition doesn't need stimulus.

        Half-lives are direct tuning knobs in HarmonicConfig:
          fast=5s, medium=8s, slow=16s (defaults).
        Per-axis alpha modulation (aggression, etc.) affects the EMA
        update step, not the cooling rate — cooling is uniform per trace.

        Args:
            now: Current timestamp (monotonic seconds). Defaults to
                 time.monotonic() if not provided.
        """
        if not self._initialized:
            return  # Nothing to decay

        if now is None:
            now = _time.monotonic()
        dt = now - self._last_ts
        if dt <= 0.0:
            return  # No time elapsed
        self._last_ts = now

        # Half-life decay: trace *= 0.5^(Δt / half_life)
        # Equivalent to exp(-ln2 * Δt / half_life). One scalar per trace.
        _LN2 = 0.6931471805599453
        self.fast *= np.float32(np.exp(-_LN2 * dt / _config.half_life_fast))
        self.medium *= np.float32(np.exp(-_LN2 * dt / _config.half_life_medium))
        self.slow *= np.float32(np.exp(-_LN2 * dt / _config.half_life_slow))

        # Mood pull continues during silence — ambient disposition
        # persists even without external events.
        if self._modulators is not None and self._modulators.mood_axis is not None:
            ax = self._modulators.mood_axis
            pull = self._modulators.mood_pull
            bias = MOOD_BIAS_VALUE
            for trace in (self.fast, self.medium, self.slow):
                trace[ax] += pull * (bias - trace[ax])

    def apply_modulators(self, mods: DynamicModulators) -> None:
        """Apply engine preset dynamic modulators to per-axis EMA rates.

        Adjusts the physics of emotional signal propagation:
        - Aggression: anger/excitement axes track faster (fast α up)
          and decay slower (slow α down). Asymmetric: easy to wind up,
          slow to cool off.
        - Confidence damping and mood pull are applied per-tick in
          update(), not as alpha adjustments.

        Safe to call multiple times — resets to config defaults first.
        """
        self._modulators = mods

        # Reset alphas to config defaults before applying modulators.
        self._alpha_fast[:] = _config.alpha_fast
        self._alpha_medium[:] = _config.alpha_medium
        self._alpha_slow[:] = _config.alpha_slow

        # Aggression: anger (dim 1) and excitement (dim 4) axes.
        # Fast alpha UP = tracks provocation faster.
        # Slow alpha DOWN = anger/excitement persist longer.
        if mods.aggression_gain > 0.0:
            for ax in _AGG_AXES:
                self._alpha_fast[ax] = min(
                    1.0,
                    _config.alpha_fast * (1.0 + mods.aggression_gain * K_AGG),
                )
                self._alpha_slow[ax] = max(
                    0.01,
                    _config.alpha_slow * (1.0 - mods.aggression_gain * K_AGG_PERSIST),
                )

        logger.debug(
            "Modulators applied: agg=%.2f conf=%.2f mood_ax=%s mood_pull=%.3f",
            mods.aggression_gain, mods.confidence_damp,
            mods.mood_axis, mods.mood_pull,
        )

    def update(self, new_semagram: list[float] | np.ndarray, now: float | None = None) -> EmotionalDelta:
        """Update all three EMA traces with a new 9d semagram.

        On first call, sets all traces to the new semagram directly
        (zero-init pattern: first delta IS initial values).

        Applies temporal decay before the EMA step if time has elapsed
        since the last update. This ensures traces regress naturally
        even if updates are sparse (high-tier agents with low cadence).

        Dynamic modulator effects applied during the update:
        - Confidence damping: fear delta attenuated before buffer update.
          Raw delta preserved for Qdrant storage; only the buffer sees
          the damped value. The memory remembers the real fear; the mind
          just doesn't dwell on it.
        - Mood pull: gentle ambient drift on the mood-biased axis after
          the EMA step. Nudges, doesn't override.

        Returns an EmotionalDelta with the change signals.
        """
        if now is None:
            now = _time.monotonic()

        # Temporal decay: cool traces proportional to elapsed time before
        # applying the new input. Ensures sparse updates don't leave
        # stale state — the buffer settles between inputs.
        if self._initialized:
            self.cool(now)

        new = np.asarray(new_semagram, dtype=np.float32)
        prev_fast = self.fast.copy()

        if not self._initialized:
            # First observation: set all traces directly
            self.fast = new.copy()
            self.medium = new.copy()
            self.slow = new.copy()
            self._initialized = True
            self._last_ts = now  # Anchor the timestamp
            delta = new  # First delta is the full semagram
        else:
            # Certainty modulation: scale residual axis (dim 8) by the
            # certainty factor. Preserves proportional relationships —
            # only the gain changes. Uncertain model → weaker reality
            # signal → emotional axes relatively dominate.
            effective_new = new.copy()
            if self._certainty < 1.0:
                effective_new[8] *= self._certainty

            # Confidence damping: attenuate the effective delta on fear
            # axes before the EMA step. The raw semagram is unchanged —
            # Qdrant gets the real projection. Only the buffer update
            # sees the damped input.
            if self._modulators is not None and self._modulators.confidence_damp > 0.0:
                damp_factor = 1.0 - self._modulators.confidence_damp * K_CONF
                for ax in _CONF_AXES:
                    # Damp the magnitude of change on fear axes, not the
                    # absolute value. A Foolhardy NPC still perceives the
                    # fear stimulus; the signal is just attenuated.
                    axis_delta = new[ax] - self.fast[ax]
                    effective_new[ax] = self.fast[ax] + axis_delta * damp_factor

            # Per-axis EMA: trace[d] = α[d] * new[d] + (1 - α[d]) * trace[d]
            # Element-wise numpy multiplication handles all 9 axes at once.
            self.fast = self._alpha_fast * effective_new + (1 - self._alpha_fast) * self.fast
            self.medium = self._alpha_medium * effective_new + (1 - self._alpha_medium) * self.medium
            self.slow = self._alpha_slow * effective_new + (1 - self._alpha_slow) * self.slow

            # Mood pull: gentle ambient drift toward mood bias value.
            # Applied after the EMA step as a separate nudge. Small enough
            # that strong events override it, but persistent during calm.
            if self._modulators is not None and self._modulators.mood_axis is not None:
                ax = self._modulators.mood_axis
                pull = self._modulators.mood_pull
                bias = MOOD_BIAS_VALUE
                for trace in (self.fast, self.medium, self.slow):
                    trace[ax] += pull * (bias - trace[ax])

            delta = self.fast - prev_fast

        # Curvature: magnitude of the delta (how fast is emotion changing?)
        curvature = float(np.linalg.norm(delta))

        # Snap: change in curvature (event boundary detector)
        snap = curvature - self.prev_curvature
        self.prev_curvature = curvature

        # Cross-buffer coherence: per-dimension variance across the three
        # timescales, mapped to [0, 1] via exponential decay.
        # High coherence = buffers agree (stable). Low = buffers disagree (volatile).
        coherence = _cross_buffer_coherence(
            self.fast, self.medium, self.slow, _config.coherence_scale,
        )

        # λ(t): retrieval balance — emotion-first vs. residual-first.
        # Curvature and |snap| push toward emotion-first (volatile → feel-first).
        # Coherence pushes toward residual-first (stable → think-first).
        lambda_t = _compute_lambda(
            curvature, snap, coherence,
            _config.lambda_alpha, _config.lambda_beta, _config.lambda_gamma,
        )

        result = EmotionalDelta(
            semagram=self.fast.tolist(),
            delta=delta.tolist(),
            curvature=curvature,
            snap=snap,
            coherence=coherence,
            lambda_t=lambda_t,
        )
        self._last_delta = result
        return result

    def get_semagram(self) -> list[float]:
        """Return the current emotional state (fast trace) as a list."""
        return self.fast.tolist()

    def nudge(self, delta: list[float] | np.ndarray, now: float | None = None) -> "EmotionalDelta | None":
        """Apply an additive emotional nudge and re-run the EMA update.

        Used by goal resonance: a curiosity spike (or standing motivational
        pull) is a direction*gain vector added to the current fast state. We
        target fast + delta and let update() smooth it, so curvature/snap/
        coherence reflect the salience rise and the scheduler can promote the
        agent's attention this same turn.

        Returns the resulting EmotionalDelta, or None if the buffer is
        uninitialized — a nudge on an agent with no emotional state yet is
        meaningless (nothing to perturb).
        """
        if not self._initialized:
            return None
        arr = np.asarray(delta, dtype=np.float32)
        target = self.fast + arr
        return self.update(target, now=now)


class HarmonicState:
    """Container managing harmonic buffers for all agents.

    One HarmonicBuffer per agent_id. Agents are created on first
    encounter (zero-init). Survives across turns — emotional momentum
    persists until session reset.
    """

    def __init__(self) -> None:
        self._buffers: dict[str, HarmonicBuffer] = {}

    def update(self, agent_id: str, new_semagram: list[float] | np.ndarray) -> EmotionalDelta:
        """Update an agent's harmonic buffer with a new semagram.

        Creates the buffer on first encounter (zero-init pattern).

        Args:
            agent_id: The agent to update.
            new_semagram: 9d emotional projection from text embedding.

        Returns:
            EmotionalDelta with change signals for prompt building.
        """
        buf = self._get_or_create(agent_id)
        return buf.update(new_semagram)

    def get_semagram(self, agent_id: str) -> list[float]:
        """Return an agent's current emotional state (fast trace).

        Returns ZERO_SEMAGRAM if the agent has no buffer yet.
        """
        buf = self._buffers.get(agent_id)
        if buf is None:
            return list(ZERO_SEMAGRAM)
        return buf.get_semagram()

    def get_certainty(self, agent_id: str) -> float:
        """Return an agent's EMA-smoothed certainty factor (1.0 if unknown).

        Used by the goal-dissonance calculation as the uncertainty term, so
        callers need not reach into the buffer's private state.
        """
        buf = self._buffers.get(agent_id)
        return buf._certainty if buf is not None else 1.0

    def get_deviation(self, agent_id: str) -> list[float]:
        """Return the emotional deviation from personality baseline.

        Computes fast - slow: what's unusual for this NPC right now.
        Used as the emotional query (Q) for memory retrieval.
        A chronically fearful NPC doesn't retrieve fear-memories every
        tick — only when fear exceeds their baseline.

        Returns ZERO_SEMAGRAM if the agent has no buffer yet.
        """
        buf = self._buffers.get(agent_id)
        if buf is None:
            return list(ZERO_SEMAGRAM)
        return (buf.fast - buf.slow).tolist()

    def get_slow(self, agent_id: str) -> list[float]:
        """Return an agent's slow EMA buffer — the personality / matured-mind baseline.

        Returns ZERO_SEMAGRAM if the agent has no buffer yet. Sleep-time
        reconsolidation uses this as the baseline to re-encode old reactions
        against (the agent's past felt through its present self).
        """
        buf = self._buffers.get(agent_id)
        if buf is None:
            return list(ZERO_SEMAGRAM)
        return buf.slow.tolist()

    def get_delta(self, agent_id: str) -> EmotionalDelta | None:
        """Return the last EmotionalDelta for an agent, or None.

        Returns the cached result from the most recent update() call —
        exact, not reconstructed.
        """
        buf = self._buffers.get(agent_id)
        if buf is None or not buf._initialized:
            return None
        return buf._last_delta

    def reset(self) -> None:
        """Clear all buffers (session reset)."""
        self._buffers.clear()
        logger.info("Harmonic state reset — all agent buffers cleared")

    def apply_modulators(self, agent_id: str, mods: DynamicModulators) -> None:
        """Apply engine preset dynamic modulators to an agent's buffer.

        Creates the buffer on first encounter (zero-init pattern).
        Safe to call before the first update() — modulators are stored
        and take effect on subsequent updates.
        """
        buf = self._get_or_create(agent_id)
        buf.apply_modulators(mods)

    def set_certainty(self, agent_id: str, certainty: float) -> None:
        """Set LLM-derived certainty factor for an agent's buffer.

        Modulates the residual axis (dim 8) on subsequent updates.
        Creates the buffer on first encounter (zero-init pattern).
        Called per-tick from the uncertainty pipeline after LLM generation.
        """
        buf = self._get_or_create(agent_id)
        buf.set_certainty(certainty)

    def apply_nudge(
        self, agent_id: str, delta: list[float] | np.ndarray,
    ) -> EmotionalDelta | None:
        """Apply an emotional nudge to an agent's buffer (goal resonance).

        No-op returning None for unknown or uninitialized agents — the nudge
        only perturbs an existing felt state, it never bootstraps one.
        """
        buf = self._buffers.get(agent_id)
        if buf is None or not buf._initialized:
            return None
        return buf.nudge(delta)

    def cool_all(self, now: float | None = None) -> None:
        """Apply temporal decay to all agent buffers.

        Called at the start of each turn before emotional processing.
        Agents that haven't received events since last turn settle
        proportional to elapsed time — the flinch fades, the mood
        persists, the personality holds.
        """
        if now is None:
            now = _time.monotonic()
        for buf in self._buffers.values():
            buf.cool(now)

    def remove_agent(self, agent_id: str) -> None:
        """Remove a specific agent's buffer."""
        self._buffers.pop(agent_id, None)

    @property
    def agent_ids(self) -> list[str]:
        """List of all tracked agent IDs."""
        return list(self._buffers.keys())

    def _get_or_create(self, agent_id: str) -> HarmonicBuffer:
        if agent_id not in self._buffers:
            self._buffers[agent_id] = HarmonicBuffer()
        return self._buffers[agent_id]


def _cross_buffer_coherence(
    fast: np.ndarray,
    medium: np.ndarray,
    slow: np.ndarray,
    scale: float,
) -> float:
    """Cross-buffer coherence: agreement across the three timescales.

    Computes per-dimension variance across fast/medium/slow, then maps
    to [0, 1] via exponential decay: exp(-scale * variance).

    High coherence (≈1.0) = buffers agree, agent in stable state.
    Low coherence  (→0.0) = buffers disagree, volatile transition.
    """
    stacked = np.stack([fast, medium, slow])         # (3, 9)
    per_dim_var = np.var(stacked, axis=0)             # (9,)
    per_dim_coherence = np.exp(-scale * per_dim_var)  # (9,) in (0, 1]
    return float(np.mean(per_dim_coherence))


def _compute_lambda(
    curvature: float,
    snap: float,
    coherence: float,
    alpha: float,
    beta: float,
    gamma: float,
) -> float:
    """λ(t) retrieval balance: emotion-first vs. residual-first.

    σ(α·curvature + β·|snap| - γ·coherence)

    Curvature and |snap| push λ toward 1 (emotion-first recall).
    Coherence pushes λ toward 0 (residual-first recall).
    |snap| used because both arc openings and closings should
    trigger emotional indexing.
    """
    arg = alpha * curvature + beta * abs(snap) - gamma * coherence
    return _sigmoid(arg)


def _sigmoid(x: float) -> float:
    """Sigmoid function, clamped to avoid overflow."""
    x_clamped = max(-10.0, min(10.0, x))
    return float(1.0 / (1.0 + np.exp(-x_clamped)))
