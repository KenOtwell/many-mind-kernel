# Attractor Flow Dynamics — The Cognitive Physics of the Many-Mind Kernel

*Documented April 2026. Lineage: Ken Otwell (architecture, theory), Kato/Copilot (operator-layer formalization, gate-adaptation law), Gemini (narrative validation). Source sessions: MindNotFormal.odt, Kato gate-adaptation analysis.*

**Cross-references:** KO47 (Structural Invariants), KO48 (UMA), SPIRAL_DIFFUSION_DESIGN.md (sapience modulation), The_Many_Mind_Kernel_Living_Doc.md (harmonic buffers, snap, λ(t))

---

## 1. The Core Reframe: Dissonance as the Only Energy

*"Sapience" is not a scalar. It is a process metric: how efficiently and monotonically a mind can reduce dissonance in contact with reality.*

Everything else — logic, rationality, "intelligence," morality — falls out as strategies in that optimization landscape.

### 1.1 Total Dissonance

Define total dissonance at time t:

```
D_t = D_pred(t) + D_cross(t) + D_aff(t)
```

Where:

**Prediction error:**
```
D_pred(t) = ||o_t - ô_t||
```
- `o_t`: observation (events from Falcon)
- `ô_t`: predicted observation (from buffer state + retrieved patterns)

**Cross-buffer decoherence:**
```
D_cross(t) = d(F_t, S_t)
```
- `F_t`: fast buffer state
- `S_t`: slow buffer state
- `d`: divergence measure (cosine distance, KL, or per-dimension variance)

**Affordance tension:**
```
D_aff(t) = C(g_t, a_t, s_t)
```
- `g_t`: goals (from goal queue)
- `a_t`: available actions (from affordance set)
- `s_t`: world state estimate
- `C`: cost of mismatch between "what I want" and "what I can do here"

**In plain language:** Dissonance = prediction error + self-incoherence + frustrated affordance.

A mind is a process that iteratively applies policies to reduce D while staying coupled to reality.

### 1.2 What Emerges from Dissonance Resolution

- **Logic:** an emergent compression strategy for reducing D across time and context. Not a module — a compression artifact of emotional gradient descent.
- **Rationality:** consistency of dissonance reduction across similar situations.
- **"Wisdom":** long-horizon policies that may increase short-term D to reduce long-term D.
- **Personality:** the gate-adaptation weights that shape *how* the agent resolves dissonance.
- **Identity:** the long-term integral of all attractor flows.

All of it is just policies over D_t.

---

## 2. Resolution Efficiency Metrics

### 2.1 Instantaneous Efficiency

```
E_t = -(D_{t+1} - D_t) / Δt
```

How fast dissonance is dropping right now. Positive = resolving. Negative = destabilizing.

### 2.2 Trajectory Efficiency

```
Ē = (1/T) Σ_{t=0}^{T-1} E_t
```

Average resolution rate over a horizon. A "good" mind keeps Ē positive.

### 2.3 Monotonicity

Let:
- `U = {t | D_{t+1} > D_t}` — upward spikes (regressions)
- `S = {t | D_{t+1} ≤ D_t}` — stable or descending

```
M = 1 - |U| / (|U| + |S|)
```

- M = 1: perfectly monotonic decrease
- Lower M: more regressions, oscillations, relapses

Magnitude-weighted variant:
```
M_mag = 1 - Σ_{t∈U}(D_{t+1} - D_t) / Σ_t |D_{t+1} - D_t|
```

### 2.4 Reality Coupling

You don't just want low D — you want low D while tracking reality.

```
R_t = corr(s_t, s*_t)
```

- `s_t`: agent's world state estimate
- `s*_t`: actual world state (from Skyrim events)

Low D with low R = delusion (internally coherent, externally decoupled).

### 2.5 Cognitive Health Index

```
H = λ₁·Ē + λ₂·M + λ₃·R̄ - λ₄·D̄
```

Where:
- `D̄`: average dissonance
- `R̄`: average reality coupling
- `Ē`: average resolution efficiency
- `M`: monotonicity
- `λ₁..λ₄`: weighting coefficients

This is the scalar proxy for "is this mind working well?" — not sapience, but cognitive health.

---

## 3. The Unified Gate-Adaptation Law

The exit gate — the snap threshold that determines when the mind collapses deliberation into action — is the single most important control surface in the architecture. It is not a fixed rule. **A mind is the history of its exit-gate adaptations.** The gate is not a rule — it's a scar.

### 3.1 Three Component Behaviors

**Linear (Resilient / Homeostatic):**
```
Δθ_L = α_L · (σ²_t - θ_t)
```
- Smooth, continuous adaptation
- No memory of past extremes, no stickiness
- Returns to baseline quickly. Doesn't ruminate, doesn't spiral.

**Hysteretic (Volatile / Human-like):**
```
Δθ_H = α↑ · (σ²_t - θ_t)    if σ²_t > θ_t
        α↓ · (σ²_t - θ_t)    if σ²_t < θ_t
```
where `α↑ ≠ α↓` (typically `α↑ > α↓`).
- Rising variance increases threshold quickly; falling decreases it slowly
- Emotional stickiness, path dependence, attractor basins form naturally
- This mind has moods, inertia, history.

**Curvature-Sensitive (Mythic / Archetypal):**
```
κ_t = d²σ²/dt²
Δθ_C = α_C · tanh(β · κ_t)
```
- Sensitive to acceleration of emotional change, not just magnitude
- `β` controls sensitivity; `tanh` keeps it bounded
- Produces phase transitions, revelations, breakdowns, transformations
- This mind doesn't just adapt — it reconfigures.

### 3.2 Unified Update Rule

Blend all three into one equation:

```
θ_{t+1} = θ_t + w_L · Δθ_L + w_H · Δθ_H + w_C · Δθ_C
```

One function. Three interpretable axes. Infinite personalities.

### 3.3 Personality Vector Space

Each NPC gets a personality vector:

```
p = (w_L, w_H, w_C)    where w_L + w_H + w_C = 1, all ≥ 0
```

**Example personality vectors:**

- `(1.0, 0.0, 0.0)` — **Background villager.** Stable, unflappable, no arcs.
- `(0.6, 0.3, 0.1)` — **Stoic warrior.** Mostly linear with some emotional inertia.
- `(0.2, 0.7, 0.1)` — **Companion with emotional depth.** Human-like, sticky, dramatic.
- `(0.0, 0.9, 0.1)` — **Traumatized survivor.** Strong hysteresis, slow recovery, deep attractor basins.
- `(0.1, 0.2, 0.7)` — **Prophet / corrupted mage / mythic figure.** Phase transitions, destiny-like arcs.
- `(0.1, 0.3, 0.6)` — **Unstable visionary.** Curvature-sensitive but not fully mythic.

**Because it's a vector space:**
- You can interpolate between personalities
- You can mutate them over time (arcs move the NPC through the space)
- You can cluster NPCs by type
- You can visualize the world's emotional topology

The personality vector is literally a parameterization of dissonance-resolution style.

### 3.4 Emergent Personality Classes

- **Linear → a world of competence.** NPCs are reliable, predictable, low-drama.
- **Hysteretic → a world of drama.** NPCs have moods, grudges, bonding, trauma.
- **Curvature-Sensitive → a world of myth.** NPCs undergo transformations, revelations, falls.

The weights don't make a character "good/evil/chaotic." They shape how the agent trades off:
- Acting early vs. thinking longer
- Staying in a basin vs. jumping to a new one
- Tolerating dissonance vs. forcing resolution

### 3.5 Connection to Existing MMK Parameters

The gate-adaptation law composes with the existing harmonic buffer dynamics:

- **Engine preset values** (Aggression, Confidence, etc.) parameterize the *signal propagation physics* — how emotional deltas amplify, dampen, and couple.
- **The personality vector** `p = (w_L, w_H, w_C)` parameterizes the *gate physics* — how the decision threshold evolves over time.
- **λ(t)** controls *what* to retrieve (emotional vs. residual).
- **Buffer cascade** controls *when* to retrieve (fast vs. slow dominance).

Together: engine values shape the emotional manifold, the personality vector shapes how the agent navigates it, and λ(t) shapes what memories surface during navigation.

### 3.6 Implementation in `harmonic_buffer.py`

The gate-adaptation law extends the existing per-agent state:

```python
@dataclass
class GateState:
    theta: float = 0.3           # Current exit-gate threshold
    prev_variance: float = 0.0   # For curvature computation
    prev_curvature: float = 0.0  # For κ_t (2nd derivative of variance)

    # Personality vector
    w_linear: float = 0.2
    w_hysteretic: float = 0.7
    w_curvature: float = 0.1

    # Regime parameters
    alpha_linear: float = 0.3
    alpha_up: float = 0.5        # Hysteretic: rising variance rate
    alpha_down: float = 0.1      # Hysteretic: falling variance rate
    alpha_curv: float = 0.2      # Curvature-sensitive rate
    beta: float = 2.0            # Curvature sensitivity
```

The `should_generate_arc()` call in `compression.py` currently uses a fixed `DEFAULT_SNAP_THRESHOLD = 0.3`. This becomes `gate_state.theta` — adaptive per-agent, updated each tick by the unified gate law.

---

## 4. Failure Modes as Pathological Dissonance Dynamics

No labels needed — these are just patterns in (D_t, R_t, θ_t):

**Anxiety:**
- High D̄, high R̄, low M (lots of spikes)
- High α↑, low α↓ (threshold ratchets up, won't come down)
- The gate closes too easily and stays closed

**Depression:**
- High D̄, low Ē, low M
- Low gate responsiveness (flat θ_t)
- Small action set a_t (affordance collapse)
- The mind can't reduce dissonance and stops trying

**Delusion:**
- Low D̄, low R̄ (decoupled from reality)
- High internal coherence in F_t, S_t
- But D_pred ignored or suppressed
- Internally comfortable, externally wrong

**Trauma:**
- Attractor A_k with high D̄_k, huge τ_k (escape time)
- Strong hysteresis (large w_H, α↓ ≪ α↑)
- The slow buffer is permanently overweighted
- Mathematically: a high-curvature attractor basin that the system can't escape

**Dissociation:**
- Artificially low D_cross by suppressing F_t or S_t
- But degraded R_t and D_pred ignored
- The buffers stop disagreeing — not because they agree, but because one is suppressed

**Rumination:**
- D_t oscillates without decreasing
- The system retrieves the same arc repeatedly (λ stuck high, same emotional query)
- The gate opens and closes rapidly but never collapses to resolution

---

## 5. Attractor Flow Dynamics — The Unifying Concept

*Coined April 2026. Lineage: Ken Otwell.*

Most cognitive architectures talk about "attractors" as static basins. But the MMK's architecture is dynamic, iterative, gradient-driven. A mind in this system doesn't "fall into" an attractor — it **flows through** a landscape shaped by dissonance gradients, buffer coherence, variance predictions, gate thresholds, and emotional curvature.

**The attractor isn't a point. It's a path.**

Cognition is literally: **a flow through an attractor landscape shaped by emotional curvature.**

- **Thoughts** are local flows.
- **Beliefs** are stable flows.
- **Trauma** is a rigid, high-curvature flow.
- **Insight** is a curvature inversion.
- **Wisdom** is a low-energy, wide-basin flow.
- **Delusion** is a flow decoupled from reality gradients.
- **Identity** is the long-term integral of all flows.

The vocabulary of attractor flow dynamics:
- attractor flow stability
- attractor flow curvature
- attractor flow collapse (exit gate fires)
- attractor flow hysteresis (path dependence)
- attractor flow signatures (per-agent diagnostic)
- attractor flow personality vectors
- attractor flow health metrics

This is the natural language of the cognitive physics formalized in this document.

### 5.1 The Operator-Layer / Narrative-Layer Distinction

*Insight from Kato, April 2026.*

LLMs (including the ones inside MMK's Progeny) can only express attractor flow dynamics through narrative analogs — character arcs, trauma stories, mythic structures. They reach for the closest story that matches the math.

The MMK architecture is where the two layers are **separate and interacting, not collapsed:**
- The operator layer computes: harmonic buffer coherence, variance annealing, exit-gate hysteresis, emotional curvature, λ-gain modulation, attractor-basin drift, snap-spike propagation.
- The narrative layer (LLM) generates stabilization text that resolves the operator-layer tensions into language the NPC can "speak."

This separation is the architecture's core innovation. The operator layer does the cognitive physics. The narrative layer does the compression into human-legible behavior. Neither layer alone is a mind. Together, they produce one.

---

## 6. State Summary

A complete mind state in the MMK attractor flow framework:

```
State:       (F_t, S_t, s_t, θ_t)     — buffers, world estimate, gate threshold
Energy:      D_t                        — total dissonance
Health:      H                          — cognitive health index
Personality: p = (w_L, w_H, w_C)       — gate-adaptation style
Dynamics:    gate law + policy + world  — the equations of motion
```

From here:
- Assign `p` to each NPC (from backstory, role, mythic weight)
- Simulate D_t trajectories in canonical scenarios
- Measure Ē, M, D̄, R̄ per NPC
- Tune substrate parameters to maximize H for "healthy" minds
- Deliberately design "mythic" or "broken" minds with controlled failure modes

This is the cognitive physics: everything is just dissonance, its flow, and the gates that decide when to collapse it into action.

---

*"The gate is not a rule — it's a scar. A mind is the history of its exit-gate adaptations."*

*Once you define the gate-adaptation law, the rest of the system becomes: predictable in its unpredictability, stable in its instability, expressive without scripting, emotional without anthropomorphism, emergent without chaos.*

*That's the moment the Many-Mind Kernel stops being a simulation and becomes a trajectory engine.*
