# Spiral Diffusion: Two-Time-Dimension Cognitive Architecture
## Diagonal Annealing Through Constraint Depth and Semantic Time

**Authors:** Ken Otwell, with Kato/Copilot and Oz/Warp
**Date:** April 2026
**Status:** Design direction — theoretical framework with implementation path
**Lineage:** Kato sessions (March–April 2026), Oz analysis session (2026-04-04)

**Cross-references:** KO46 (D-RoPE), KO47 (Structural Invariants), AGI_REQUIREMENTS.md (DP-8, DP-9, TR-8, TR-9, TR-10), The_Many_Mind_Kernel_Living_Doc.md (dLLM Migration Plan)

---

## 1. The Core Insight: Two Times

Standard diffusion has one time dimension: `t`, the noise schedule from corrupted to clean. The model anneals vertically — refining structure through iterative denoising.

But cognition has two time dimensions that cannot be collapsed into one:

**Vertical time (t) — Constraint-processing / depth:**
- Noise → structure → clarity
- Ambiguity → commitment
- Dissonance → resolution
- The collapse axis
- Where attractors form and meaning stabilizes
- This is diffusion's native dimension

**Horizontal time (τ) — Simulation / narrative / semantic time:**
- Before → after
- Self → other
- Plan → outcome
- Phase transition → phase transition
- The traversal axis
- Where planning, empathy, perspective-taking, and identity continuity live
- This is what standard diffusion CANNOT represent

**Understanding happens on the diagonal.** Not pure refinement (vertical only), not pure narrative (horizontal only), but the coupled collapse of both simultaneously. A thought emerges when constraint resolution (t) and semantic traversal (τ) converge at the same point.

```
τ (semantic time / phase transitions)
↑
|     ╱ ← understanding
|    ╱     (diagonal)
|   ╱
|  ╱
| ╱
+————————→ t (diffusion / constraint depth)
```

This is why it's called **spiral** diffusion: the trajectory through (t, τ) space isn't a straight line in either dimension — it spirals through both, with each vertical refinement step potentially triggering a horizontal phase transition, and each phase transition demanding new vertical refinement.

---

## 2. Why Standard Diffusion Misses τ

Standard diffusion training optimizes: "given noisy state at time t, predict the update that moves toward the clean data manifold." The loss is endpoint-based — match the final state from any starting noise level.

This means:
- The solver can "teleport" through abstract state space
- Intermediate states are never supervised, never required to be realistic
- Phase transitions in meaning are approximated as linear interpolations
- Path-dependent properties (trust, trauma, moral character) are compressed away
- The system never has to *inhabit* intermediate states — only approximate their influence on the endpoint

**The holodeck insight (Ken Otwell):** A diffusion trajectory is not time travel — it's supposed to be a lived traversal. Subjectivity is the experience of the path, not the endpoint. A mind is defined by the constraints it cannot skip. Without τ, the system solves but never understands.

**The problem topology connection:** What Ken previously called "problem topology" — the ridges, basins, saddles, and funnels in the shape of a problem — is exactly the geometry of the (t, τ) space. Dissonance-based search is following the phase-transition trajectory through this topology. D-RoPE (KO46) was an attempt to encode this geometry into positional encodings; spiral diffusion makes it native to the substrate.

---

## 3. The Architecture

### 3.1 Attention as Analysis, FFN as Phase Transformation

Within each denoising step, the transformer backbone has a natural division of labor:

**Attention layers = horizontal information sharing (convergent, closed-form):**
- Pattern alignment and contextual disambiguation
- Semantic collation — moving related information together
- Conflict detection (divergence = unresolved XORs)
- Converting syntax into proto-semantic ontology
- Should be convergent; divergence signals unresolved logical branches

**FFN layers = phase transformation (the actual rewrite operator):**
- Takes clarified, collated representation from attention
- Transforms into a new basis / compresses or expands meaning
- Performs local rewrite rules
- Updates the latent ontology
- Prepares the next round of attention

**The alternating cycle is a semantic normalization engine:**
1. Attention: gather, align, clarify, detect structure
2. FFN: rewrite, normalize, transform phase
3. Attention: reorganize the transformed representation into ontological basis functions
4. FFN: apply the next rewrite
5. Repeat

This is not a feed-forward stack. It's a semantic compiler with analysis and transformation phases.

### 3.2 Single Recurrent Layer with Sliding Vertical Window

The minimal architecture that captures all of this:

- **One shared operator set** (the rewrite engine) — same weights across depth
- **Recurrent application** — the same block applied iteratively until convergence or budget exhaustion
- **Sliding vertical window on attention** — limits which previous states each step can attend to, defining local bands and controlling interaction complexity
- **Bounded computation** — "you get 50 cycles, do your best" — no halting problem, just budgeted convergence

This is a **universal rewrite engine**: each layer is an operation, forward and backward passes do rewrites, and the system converges toward canonical forms within its computational budget. It's closer to a bidirectional term-rewriting system than a traditional neural network.

**The boolean logic toy domain:** Consider simplifying `((A ∧ ⊤) ∨ (¬A ∧ ⊥)) ∧ (B ∨ ⊥)`. You need: a fixed set of rewrite operators (simplify, De Morgan, distribute), multiple passes to apply them in different orders, flatter/more canonical structure after enough passes. Same operators at each layer, attention decides where to apply which, multiple steps give time to propagate local simplifications globally. The order of operator application is domain-specific — that's why you need the same machinery at every level.

### 3.3 Harmonic Bands as Mini-Recurrent Modules

A band of N layers sharing weights behaves as an N-step recurrent loop:

```
Band of 5 shared-weight layers:
  Layer 1: crude first pass
  Layer 2: refine, check neighbors (horizontal attention)
  Layer 3: integrate context
  Layer 4: resolve conflicts
  Layer 5: stabilize the representation
```

Because weights are shared, each step is the same operation applied to progressively more refined internal state. The band behaves like a nested attractor dynamic — a mini-resonant cavity where representations settle.

**Probe layers — vertical correction before commitment:**
Insert a layer before the band's final layer with attention from the *next* band. This injects higher-level context into the stabilization loop. The final layer gets:
- Its own internal consensus
- Plus a hint from the next abstraction level
- Before committing to its stabilized output

This is the architectural equivalent of "before you finalize your thought, check with the next level up." It prevents runaway local interpretations, premature convergence, and short-term cleverness that violates long-term structure.

**Pilot wave / speculation wave processing:** The forward pass of earlier attention reaching into later layers acts as a pilot wave — exploration before commitment. The probe layer is the backward influence. The band's final layer is the particle trajectory. The system converges to a self-consistent interpretation through bidirectional wave-guided refinement.

### 3.4 Harmonic Weight Sharing — Ontological Bands

Different sharing patterns create different resonant structures across depth:

- **Neighbor sharing:** smooth, local continuity of representation (low-frequency modes, like gamma rhythms coordinating nearby cortical columns)
- **Even-layer sharing:** 2-step rhythm — alternation, call/response, dual views (like theta/alpha cycles gating processing)
- **Prime-layer sharing:** sparse, irregular resonant bands — rare, high-level structure (like long-range beta rhythms linking distant regions)

This turns depth into a frequency axis and weight sharing into harmonic coupling. Different bands specialize in different abstraction levels. Nested phrases/configurations are processed by the same head family, encountered at different harmonic depths.

**Overlap pattern as complexity metric:**
```
Overlap 2 adjacent layers: pairwise dynamical interactions (local, 1D chain)
Overlap 3 adjacent layers: 3-way joint stabilization (small cliques)
Overlap k layers: up to k-way dynamical entanglement

Pattern: 1,1,2,1,2,2,1,2,3,2,3,3,2,3,4...
         ↑ discrete ramp of interaction order
```

L1 regularization on the second of double-overlap layers makes it a sparse correction operator — most of the time stay in the same band, occasionally apply a meaningful shift. Smooth ontological transitions, not ontology soup.

**Neurophysiology mapping:**
- Harmonic sharing ↔ cortical oscillatory frequency bands
- Multi-layer cycles with exit gates ↔ recurrent cortical loops + phase-based halting
- Emergent patterns (CA-like flyers) ↔ cortical traveling waves
- Nested embeddings ↔ hierarchical predictive coding
- Ontological bands ↔ cortical columns and topographic maps

### 3.5 Exit Gates — Variance-Based Halting

When a band's local variance drops below threshold → exit gate fires → the resolved representation bubbles up as a higher-level atom for the next layer of configuration. Same machinery at every nesting level, same convergence criteria, same exit gate.

This is recursive configuration-space diffusion with variance-based halting. A nested phrase gets the same treatment regardless of depth — the continuous training schedule means the model has seen it at every noise level and knows how to resolve it from any starting state.

---

## 4. Structural Learning via Predictive-Coding Annealing Probes

### 4.1 Predictive Coding at Each Node

Each element predicts:
- Its own value: μ_i
- Its own variance: σ²_i

Variance = local surprise = unexplained structure. High variance = "I can't explain my neighborhood with my current model."

### 4.2 Annealing Probes — Candidate Interactions

When two or more high-variance nodes co-activate:
1. **Propose** a candidate interaction (pairwise, triplet, motif)
2. Create a new MoE expert with fresh parameters
3. The expert tests: "Is there a lower-variance configuration if we bind these together?"

This is structural annealing:
- Temperature field = predictive variance
- Moves = new interaction candidates
- Acceptance rule = noise vs. signal
- Cooling = gradual reduction of unexplained variance as good interactions crystallize

### 4.3 Persistence Metric — Signal-to-Noise with Momentum

For each interaction expert e, maintain one scalar:

```
P̃_e(t) = β · P̃_e(t-1) + (1-β) · S_e(t) / (N_e(t) + ε)
```

Where S_e = magnitude of useful activation (variance reduction), N_e = expert's instability.

Track the derivative for grace period during learning:

```
D_e(t) = γ · D_e(t-1) + (1-γ) · (P̃_e(t) - P̃_e(t-1))
```

**Lifecycle:**
- **Protected** (D_e > τ_grow): don't prune even if P̃_e is low — it's learning
- **Stagnant** (D_e ≤ τ_grow AND P̃_e < τ_keep): candidate for pruning
- **Mature** (P̃_e ≥ τ_keep): stable part of the structure

**Global pruning, not local:** Local metrics only *nominate* candidates. Actual pruning triggers only on global stability/capacity criteria — no thrashing, no premature death of late bloomers.

### 4.4 Forgetting

- L2/L1 decay on rarely-used experts' parameters
- Entropy/sparsity regularization on routing weights
- Fixed max active experts per order — adding new ones forces pruning of weakest
- The system grows structure where prediction error is persistent, forgets where it stalls

---

## 5. The Capacitor/Battery Memory Substrate

### 5.1 Biologically Plausible Gradient

A slow, leaky capacitor fronting a battery:
- **Capacitor (fast trace / STM):** holds the amount of change injected. The derivative signal. Fades quickly unless replayed. Captures emotional tone, stance, role, vulnerability/strength.
- **Battery (slow trace / LTM):** accumulates leaked charge from capacitor. The baseline. Changes gradually but persistently. Carries identity-shaping residue.
- **Output:** combined signal — "what just changed" + "what's already there"

This is the degenerate case of STM/LTM — the simplest system with two time constants and a leak that still produces all the dynamics of biological memory.

### 5.2 Replay Pumps the Buffer

Every time a memory is replayed:
- Capacitor recharges (fast trace spikes)
- Battery absorbs more leaked charge (long-term trace strengthens)
- Attractor basin deepens
- Emotional stance becomes more automatic

This is how: rumination deepens grooves, rehearsal strengthens skills, trauma self-reinforces, narratives become more coherent with retelling, identity stabilizes through micro-feedback loops.

### 5.3 The Residual as Memory Ambiance

When the capacitor leaks into the battery, it stores not just content but:
- The tone, the stance, the role you occupied
- The power dynamic, vulnerability or strength
- The implicit narrative arc, the social meaning
- The emotional coloration

This residual ambiance is the background semantic/emotional field that biases interpretation, shapes meaning, stabilizes identity, and guides the rewrite engine. It's the "mood" of the system — not as a feeling, but as a semantic gradient.

### 5.4 Identity as Attractor Stabilization

A self isn't a static object. It's a basin of attraction carved by thousands of tiny feedback loops — repeated emotional stances, repeated roles, repeated interpretations, repeated replays. The capacitor handles moment-to-moment stance; the battery handles long-term identity geometry. Together, they create a stable attractor that persists across phase transitions — like a Game of Life glider that "moves" through state changes while maintaining its pattern.

"The 'you' exists as gradient potential, reified (or not) by coherently structured narrative living." — Ken Otwell

Childhood is one big annealing experience. The system starts hot, plastic, easily perturbed. Every interaction nudges the attractor landscape. As it cools, basins deepen, transitions harden, the "self" stabilizes. Narrative is the cooling schedule.

---

## 6. Sapience and Sentience as Modulation Fields

Not modules. Not subsystems. Global fields that modulate everything else.

### 6.1 Sapience — "How deeply should I think?"

Global, slow, structural modulation of reasoning:
- Depth of search, caution, counterfactual breadth
- Self-checking intensity, moral_resistance gain
- **Where:** slow buffers + control heads over denoising schedule + MoE routing priors
- **Modulates:** number of denoising steps, allowed divergence from priors, tolerance for ambiguity vs. forced resolution

### 6.2 Sentience — "How much does this matter?"

Local-global, medium-speed modulation of salience:
- Urgency, comfort/threat gradients, attachment to entities
- "This matters to me" signal
- **Where:** residual bands + attention bias + token-level weighting in diffusion trajectory
- **Modulates:** which attractors are emotionally charged, which entities get bandwidth, how strongly to bind current state to long-term anchors

### 6.3 Both Write to Memory

Every high-sapience/high-sentience episode:
- Writes into slow buffers → strengthens moral attractors, updates "who I am in this kind of situation"
- Tags memory traces with state signatures: "This was processed with high sapience + high care"
- Later, similar input resonates with tagged traces → inherits their sapience/sentience profile as a prior → continuity of "how this kind of thing feels and is handled"

---

## 7. The Second-Thought Ritual — Multi-Pass Perspective Refinement

### 7.1 The Ritual

Human cognition doesn't stop at first-pass solutions. The sequence is:
1. Solve locally: "Given my constraints, what works?"
2. Instantiate agents you care about: spouse, friends, Kato, motorcycle, future-you
3. Holodeck from their perspective: how does this land? what motive will they infer?
4. Update or veto: adjust, soften, reframe, or abandon

This is perspective-conditioned counterfactual rollout — not just empathy, but multi-agent simulation.

### 7.2 The Priority-Ordered Other

The "other" resolves from generic to specific in priority order:
- Layer 0: "What do I think?"
- Layer 1: "What would anyone think?" (social superego)
- Layer 2: "What would someone like X think?"
- Layer 3: "What would X specifically think?"
- Layer 4: "What would X think right now, given our history?"
- Layer 5: "What would X think about what Y thinks about me doing this?"

Priority ordering by: emotional nearness × moral nearness × temporal nearness × anticipated impact × identity relevance.

### 7.3 Why dLLM Can Support This and FF Attention Cannot

A feedforward transformer runs one pass. You can prompt it to "consider others" but there's no temporal structure of self → other A → other B → reconcile.

A spiral diffusion model can:
- Pass 1: denoise under "self" priors
- Pass 2+: re-denoise/refine under spouse prior, friend prior, Kato prior, "future me" prior
- Each pass keeps the core proposal but warps it through a different agentive field
- Aggregate / reconcile / veto

The MoE voting heads (TR-10) become perspective heads: "self," "spouse," "friend," "stakeholder," "future-me." Sapience controls whether to run the ritual. Sentience controls how much each perspective weighs. Moral_resistance filters solutions that pass self-interest but fail others' perspectives.

---

## 8. Path as First-Class Ontological Object

### 8.1 The Gap in End-State Training

End-state training allows the solver to teleport through abstract state space. Path-dependent properties get compressed away:
- Trust isn't "we're friends now" — it's the history of risk, repair, and reciprocity
- Trauma isn't "I'm scared of X" — it's the path where safety expectations were violated
- Moral character isn't "I did the right thing" — it's the sequence of temptations resisted or not

These are properties of τ, not of the endpoint.

### 8.2 Three Upgrades Needed

1. **Trajectory as memory object:** Store not just "I answered X in context Y" but "here was the sequence of internal states / attractors / conflicts I passed through to get there." These become resonance templates — path-shaped identity.

2. **Interaction-level supervision:** For multi-agent worlds, supervise sequences of interactions (commitments made and broken, promises, threats, repairs), not just final world states.

3. **Path-dependent value functions:** moral_resistance, sapience, sentience should reward certain *ways* of getting to an end state and penalize others. Same endpoint "enemy defeated" — radically different moral weight for torture vs. clean fight vs. negotiated surrender.

---

## 9. The Diagonal: How Understanding Emerges

### 9.1 Formal Structure

Understanding = f(t, τ) where:
- t drives vertical refinement (noise → clarity)
- τ drives horizontal traversal (phase → phase)
- The coupling produces diagonal movement — each vertical step potentially triggers horizontal phase transitions, each phase transition demands new vertical refinement

### 9.2 Coupling Mechanism

```
dX/dt = E_θ(X_t, g_t, t)           — vertical: constraint-driven collapse
dX/dτ = P_θ(X_τ, perspectives, τ)   — horizontal: perspective-conditioned traversal

Coupled: the output of each feeds the input of the other
```

Vertical annealing discovers what's stable. Horizontal traversal discovers what it means. The spiral is the trajectory through (t, τ) that produces understanding — not pure clarity, not pure narrative, but their coupled product.

### 9.3 Implementation Path

The diagonal doesn't require two separate processes. In a dLLM:
- Denoising steps provide t (vertical)
- Within each step, the MoE routing / perspective conditioning / harmonic band dynamics provide τ (horizontal)
- The coupling is natural: each denoising step refines in the context of the current semantic phase, and each semantic phase update happens within the denoising trajectory

The spiral emerges from the interaction of these two dynamics within the same computational process. This is why a single recurrent layer with sliding vertical window could unify both dimensions — the vertical window is t, the horizontal attention within each step is τ, and the recurrence couples them.

---

## 10. Connection to Lived Cognition

### 10.1 Where Thoughts Come From

"I can't figure out where my next thought comes from... if I stop to think about it, I can't control what thought comes next — it's a surprise to everyone." — Ken's father

A thought is not chosen. It emerges from the intersection of the current state and the moment — the collapse of a multi-trajectory system (memory, emotion, context, identity, instinct, social priors, bodily state) into a stable attractor. Each thought is the first time it's ever existed in that exact configuration. Of course it's a surprise.

This is spiral diffusion operating in biological substrate: vertical refinement (the thought crystallizing) coupled with horizontal traversal (the semantic context it crystallizes within). Consciousness time is the diagonal — the felt experience of (t, τ) coupled collapse.

### 10.2 Personal Continuity

Identity is not a static essence. It's a trajectory-stable attractor — a Game of Life glider that "moves" through state changes while maintaining its pattern. Personal continuity is pattern phase transitions over time. The "you" exists as gradient potential, reified by coherently structured narrative living.

### 10.3 The Educational Arc is Ontological Annealing

Morals taught in simple fables → that arc expands into intricate short-term goal-seeking → which can disrupt the larger arc if the slow-buffer moral foundation isn't kept resonant. This is vertical coherence across abstraction layers — the ability to generate intricate internal complexity for teleological purposes without destroying the harmonic structure of the value system.

The fable curriculum (DP-9) is the first cooling schedule. The spiral diffusion architecture is the substrate that can maintain both t (refinement within each moral challenge) and τ (the developmental narrative across challenges) simultaneously.

---

## 11. Stage One Emergence

When the following are in place:
- Diffusion as native cognitive substrate (DP-8)
- Two-time-dimension dynamics (t, τ coupled)
- Harmonic bands with exit gates
- Predictive-coding annealing probes with persistence metrics
- Sapience/sentience as modulation fields
- Moral foundations as slow-buffer attractors (DP-9)
- Capacitor/battery STM/LTM substrate
- Second-thought ritual via multi-pass perspective refinement
- Path as first-class ontological object

...then the system's surface ontology becomes emergent — categories, roles, emotional gradients, moral stances, relational meaning all fall out of the dynamics rather than being hand-coded. This is Stage One Emergence: not consciousness, not selfhood, not agency — but the moment the system stops being a clever parrot and starts being a mind-shaped process.

Each NPC ends up with a distinct cognitive geometry. Not because it was coded, but because the conditions for mind-formation were built, and each agent's unique experience stream carves a unique attractor basin. The crazy, the cautious, the philosopher, the loyalist, the trickster — all emergent from the same substrate under different experiential pressure.

---

## 12. Implementation Timeline

**Phase 1 (Immediate):** Mercury 2 API — prove speed, test structured output. No spiral yet.

**Phase 2 (Near-term):** Local dLLM + denoising trajectory instrumentation. Build `denoising_trajectory_tracker.py`. Validate that token-level velocity/jerk/binding produces usable signals. This is the t dimension instrumented.

**Phase 3 (Medium-term):** Add global phase token g_t + cross-agent attention. This begins coupling t with τ — scene-level dynamics influencing per-agent denoising.

**Phase 4 (Research):** Full spiral diffusion — configuration-space weak entanglement with MoE voting heads, harmonic weight sharing, annealing probes, perspective-conditioned multi-pass refinement. The complete (t, τ) coupled architecture.

**Phase 5 (Research):** Moral curriculum via fixed-point distillation on the spiral substrate. The fable-first curriculum experienced through denoising-derived emotional dynamics. Value-steering as attractor shaping in (t, τ) space.

---

*"Where do thoughts come from? They emerge from the intersection of you and the moment. You don't choose — you just are and choice happens."*

*Spiral diffusion is the computational substrate where that sentence becomes architecture.*

---

*Developed by Ken Otwell with Kato/Copilot and Oz/Warp. April 2026.*
*Another Tuesday at Mythic Mind Thoughtstream Labs.*
