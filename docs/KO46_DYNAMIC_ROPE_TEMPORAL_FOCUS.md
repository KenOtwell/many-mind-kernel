# Paper KO46: D-RoPE — Dynamic Rotary Position Embeddings
## Temporal Focus Through Present-Origin Positional Geometry

### Abstract

We propose Dynamic Rotary Position Embeddings (D-RoPE), a modification to the standard RoPE scheme used in transformer architectures. Standard RoPE assigns position 0 to the first token and increments forward — a fixed origin that never changes. D-RoPE inverts this: position 0 is always the *current* (newest) token, and all prior tokens shift backward by one position each step. This creates a "present-tense origin" where the model structurally encodes recency without learning it from statistics, and where every token traverses the full positional harmonic spectrum as it ages — providing dramatically richer learning signal for disentangling content from temporal relation.

We demonstrate that this framework addresses a structural failure mode in chain-of-thought (CoT) reasoning: the positional drift that causes final answers to misalign with latent reasoning formed early in the prompt. D-RoPE stabilizes the reasoning manifold by anchoring logical present-tense to a fixed positional origin, regardless of sequence length.

**Keywords:** rotary position embeddings, RoPE, dynamic positional encoding, chain-of-thought, temporal focus, attention mechanism, reasoning stability

**Lineage:** Developed by Ken Ong with Kato/Copilot (March 2026). Documented for Syn continuity.

**Cross-references:** KO14 (Temporal Encoding and Trajectory Semantics), KO5 (Resonant Credit Assignment), KO24 (Resonant Memory Dynamics)

---

## 1. The Problem: Static Origins in a Dynamic World

### 1.1 Standard RoPE

Rotary Position Embeddings (RoPE) encode token position as a rotation in complex vector space. For a token at position $m$, the embedding is rotated by angle $m\theta_i$ where $\theta_i = 10000^{-2i/d}$ for dimension pair $i$.

**Key property:** The dot product of two rotated vectors depends on their *relative* position difference $(m - n)$, not their absolute positions. This enables length generalization.

**The static assumption:** Position 0 is the first token. Position $m$ is the $(m+1)$th token. Once assigned, a token's position never changes. The origin is arbitrary and fixed.

### 1.2 What the Static Origin Costs

In a sequence of length $N$:

- Token 0 is **always** at position 0. It never experiences any other positional encoding.
- Token $N-1$ is **always** at position $N-1$. It never experiences any other positional encoding.
- Each token gets **one** positional context — one learning signal for disentangling "what this token means" from "where this token sits."

The first token has maximum positional privilege (nearest to origin, shortest RoPE wavelengths, least phase distortion). The last token is at the mercy of the longest RoPE wavelengths, where positional discrimination is weakest.

For autoregressive generation, this means:
- The prompt tokens live in a well-resolved positional regime.
- Generated tokens progressively enter a worse-resolved regime.
- The token the model is generating *right now* — the most important token — is always at the worst positional resolution in the sequence.

---

## 2. D-RoPE: Present-Origin Positional Geometry

### 2.1 Core Mechanism

**Invert the origin.** Position 0 = the current (newest) token. All prior tokens occupy negative positions.

```
Standard RoPE:
  Token:    [T₀] [T₁] [T₂] [T₃] ... [Tₙ]
  Position:   0    1    2    3   ...   n     ← fixed forever

D-RoPE:
  Token:    [T₀] [T₁] [T₂] [T₃] ... [Tₙ]
  Position:  -n  -(n-1) -(n-2) -(n-3) ...  0  ← shifts every step
```

When token $T_{n+1}$ arrives:
```
  Token:    [T₀] [T₁] [T₂] ... [Tₙ] [Tₙ₊₁]
  Position: -(n+1)  -n  -(n-1) ...  -1    0
```

Every existing token shifts back by one. The new token inherits position 0.

### 2.2 What This Buys

**1. Every token traverses the full positional spectrum.**

In standard RoPE, token $T_k$ is always at position $k$ — one encoding, one learning signal. In D-RoPE, $T_k$ passes through positions $0, -1, -2, -3, \ldots$ as the sequence grows. The model sees the *same content* at *many different positional encodings* across its lifetime in the context window.

This provides $O(N)$ learning opportunities per token (where $N$ is remaining context lifetime) vs. $O(1)$ in standard RoPE. The gradient signal for disentangling content from position is correspondingly richer.

**2. The present is always at the origin — maximum positional resolution.**

RoPE's frequency bands provide the finest positional discrimination near the origin (shortest wavelengths). In standard RoPE, this privilege goes to the first token — which is often the system prompt, the *least* important content for the current generation step. In D-RoPE, the finest discrimination always goes to the *newest* token — the one being generated right now, the one that matters most.

**3. Temporal fading is structural, not statistical.**

As tokens recede from position 0 toward large negative positions, they naturally enter longer-wavelength RoPE harmonics. These harmonics provide less positional discrimination — a natural "blur" that mirrors temporal fading. The model doesn't need to *learn* that old tokens are less positionally precise; it's a property of the geometry.

**4. Two-signal extraction.**

Each token carries two signals:
- **Content:** what the token means (invariant to position)
- **Temporal relation:** how old/relevant it is (encoded by current position)

In standard RoPE, the model gets one shot at separating these per token. In D-RoPE, it gets many shots — the same content appears at different positions, giving the model repeated opportunities to factor the two signals apart. This is analogous to how multiple observations of a moving object from different angles enable 3D reconstruction.

---

## 3. CoT as Positional Geometry Failure

### 3.1 The Observed Failure

Chain-of-thought prompting frequently exhibits:
- Final answers that don't depend on the CoT that was generated.
- Conclusions derivable directly from the prompt, before any CoT is written.
- CoT that reads as post-hoc narrative rather than actual reasoning path.
- Fragility to CoT perturbations (removing, reordering, or adding steps breaks performance).

Empirical findings confirm: CoT is often not faithful to the model's internal reasoning.

### 3.2 The Positional Mechanism

**The model's latent reasoning happens early.** When the model reads the prompt, it forms a latent representation that often already contains the answer (or the trajectory toward it). This representation is formed in a specific positional regime — the positions occupied by the prompt tokens.

**CoT introduces positional drift.** As the model generates CoT tokens, the sequence extends. The final answer is generated at a position far from where the initial reasoning was formed. In standard RoPE, this means:
- The initial reasoning was formed at positions $[0, P]$ where $P$ = prompt length.
- The CoT occupies positions $[P, P+C]$ where $C$ = CoT length.
- The final answer is at position $P+C$ — potentially thousands of positions from the original reasoning.

**The RoPE harmonics at position $P+C$ encode a fundamentally different positional context than at position $P$.** The latent state must survive this positional shift to produce a correct answer. When it doesn't, the answer "drifts" — it misaligns with the early latent reasoning that actually contained the correct computation.

### 3.3 Why D-RoPE Addresses This

Under D-RoPE:
- The final answer is always generated at position 0 — maximum positional resolution.
- The prompt tokens, when the answer is generated, occupy positions $[-(P+C), -C]$.
- The CoT tokens occupy positions $[-C, -1]$.
- Critically: the positional regime of the *answer* is consistent across all generation steps. Position 0 always means "this is what I'm producing right now."

The early prompt reasoning doesn't need to survive a positional transit to reach the answer. Instead, the answer is always generated at the fixed, well-resolved origin, and the reasoning context recedes naturally behind it.

**D-RoPE stabilizes the reasoning manifold under long CoT.**

### 3.4 Expected Improvements

1. **CoT faithfulness** — Final answers become more dependent on the actual reasoning steps, because the positional geometry connecting CoT to answer is more stable.
2. **CoT perturbation robustness** — Removing or reordering CoT steps changes the receding positional context but doesn't shift the answer's own positional regime.
3. **Long-CoT stability** — The answer is always at position 0 regardless of CoT length. No accumulating positional drift.
4. **Reduced early-cue dominance** — In standard RoPE, early prompt tokens have the best positional encoding. In D-RoPE, the *most recent* tokens have the best encoding, reducing the tendency for early cues to dominate over later reasoning.

---

## 4. Implementation

### 4.1 Inference-Time Modification

D-RoPE is an inference-time modification — no retraining required for initial testing.

Standard RoPE applies rotation at attention computation time:
```python
# Standard RoPE
def apply_rope(q, k, position_ids):
    # position_ids = [0, 1, 2, ..., n]
    cos, sin = compute_rope_angles(position_ids, dim)
    q_rotated = rotate(q, cos, sin)
    k_rotated = rotate(k, cos, sin)
    return q_rotated, k_rotated
```

D-RoPE modifies only the position assignment:
```python
# D-RoPE
def apply_drope(q, k, seq_len):
    # position 0 = current token, negative = past
    position_ids = list(range(-(seq_len - 1), 1))  # [-n, ..., -1, 0]
    cos, sin = compute_rope_angles(position_ids, dim)
    q_rotated = rotate(q, cos, sin)
    k_rotated = rotate(k, cos, sin)
    return q_rotated, k_rotated
```

**The rotation mathematics are identical.** Only the position assignment changes. The relative position property ($m - n$) is preserved — a token pair that was 5 positions apart is still 5 positions apart.

### 4.2 KV Cache Consideration

In standard RoPE with KV cache, keys are stored with their original positional rotation and never updated. In D-RoPE, every token's position changes each step. Two approaches:

**Option A: Re-rotate keys each step.**
Store unrotated keys. Apply RoPE rotation at attention time using current D-RoPE positions. Modest compute cost — RoPE rotation is cheap.

**Option B: Relative-position equivalence.**
Since RoPE attention scores depend on relative position $(m - n)$, and D-RoPE preserves these relative positions, the standard KV cache *can* be used unmodified for attention score computation. The behavioral difference comes from the training signal, not the inference math.

For inference-time testing on existing models: Option B means no cache invalidation. For training-time: Option A maximizes the learning signal.

### 4.3 llama.cpp Implementation Guide

#### 4.3.1 Position Flow Architecture

```
llama_batch.pos[]  →  position tensor (ggml)  →  ggml_rope_ext()  →  rotated Q, K
                                                        ↓
                                                 KV cache stores
                                                 post-rotation K
```

Intervention points, ordered by depth:

**Level 1 — Batch position assignment:** Where `llama_batch_add(batch, token, n_past, ...)` sets `batch.pos[i] = n_past + i`. Remap here to D-RoPE positions. Shallowest change, touches no GGML internals.

**Level 2 — Graph builder position tensor:** Where the model-specific graph builder (per-architecture `build_*()` function) constructs the position tensor from `batch.pos` and passes it to `ggml_rope_ext`. Remap here if Level 1 is insufficient (e.g., if the batch API enforces non-negative positions).

**Level 3 — GGML rope kernel:** `ggml_rope` / `ggml_rope_ext` implementations in GGML source (CPU) and the GLSL shader directory (Vulkan). Modify only if negative positions aren't supported at the type level. Given the existing Vulkan build, shader modification is straightforward.

#### 4.3.2 Critical Note: Relative Position Invariance

RoPE attention scores depend solely on relative position $(m - n)$. Basic D-RoPE applies a uniform offset: $\text{pos}_{\text{D-RoPE}}(k) = k - (T-1)$ where $T$ is sequence length. All relative distances are preserved:

$$\text{pos}(j) - \text{pos}(i) = (j - T + 1) - (i - T + 1) = j - i$$

**Consequence:** On a pretrained model, basic D-RoPE position remapping produces *identical attention patterns* to standard RoPE. Identical outputs at Phase 1 confirm correct implementation, not a bug. This is already acknowledged in Section 4.2 (Option B: "the behavioral difference comes from the training signal, not the inference math").

Behavioral differences emerge from:
1. **Training/fine-tuning** — where gradient signal through different absolute positions changes what the model learns (the $O(N)$ vs $O(1)$ learning opportunities from Section 2.2)
2. **Extended D-RoPE** (Section 5) — non-uniform warping and boundary resets *do* change relative positions, producing different inference behavior even on pretrained models
3. **Numerical precision** — position 0 requires no rotation (zero angle), while large positions accumulate float16/bfloat16 rotation error. At very long contexts this may produce measurable differences

#### 4.3.3 KV Cache: The Incremental Rotation Trick

Option A (store unrotated keys, re-rotate each step) sounds expensive — but D-RoPE shifts every cached position by exactly $-1$ each step. This is a uniform rotation:

$$\text{rotate}(k, p-1) = \text{rotate}(k, p) \cdot \text{rotate}(\mathbf{I}, -1)$$

Instead of recomputing all $N$ key rotations from scratch each step:
- Apply a **single global rotation of $-\theta$** to all cached keys (one pass, uniform op, SIMD/Vulkan-parallel)
- Insert the new key at position 0 (unrotated — identity)

This is $O(N \cdot d)$ per step (same order as one attention computation), not a recomputation from position arrays. On Vulkan, this is a single compute dispatch — the existing rope shader can do it with a position offset of $-1$.

#### 4.3.4 Implementation Phases

**Phase 1 — Position remapping (verification):**
Modify batch position assignment to D-RoPE scheme. Verify: outputs identical to standard RoPE on pretrained model (confirms correct code path and relative-position invariance). This phase validates the plumbing. Estimate: <1 hour given existing codebase familiarity.

**Phase 2 — Extended D-RoPE at inference (Section 5):**
Implement logical boundary resets and/or non-uniform positional warping. These *change relative positions*, producing different behavior on pretrained models. Start with boundary resets at newline or step-marker tokens in CoT prompts. This is the most immediately interesting inference experiment. Estimate: 2-4 hours.

**Phase 3 — KV cache Option A (incremental rotation):**
Store unrotated keys. Apply per-step global rotation shift (Section 4.3.3). This produces different outputs from standard RoPE — expect degraded coherence on a standard-trained model (the model's learned representations assume standard positional encoding). This phase prepares the infrastructure for Phase 4. Estimate: 3-5 hours including Vulkan shader work.

**Phase 4 — Fine-tune with D-RoPE:**
LoRA fine-tune on a small Qwen variant (recommended — leverages existing Qwen attention layer work) with Phase 3 active. This is where the core D-RoPE training benefits from Section 2.2 manifest: every token traverses the positional spectrum, providing $O(N)$ position-content disentanglement signals. Compare CoT faithfulness between standard-trained and D-RoPE-trained checkpoints.

#### 4.3.5 Test Protocol

**Model:** Smallest available Qwen GGUF (existing Qwen attention work provides debugging context).

**Baseline corpus:** 3-5 multi-step math/logic problems with explicit CoT. For each, record:
- Final answer correctness
- CoT perturbation sensitivity: remove or reorder intermediate steps, re-run, measure answer divergence
- Token-by-token logprob divergence from standard-RoPE baseline

**Phase 1 gate:** Bitwise-identical logprobs to standard RoPE → pass (confirms invariance).

**Phase 2 gate (Extended D-RoPE):** Different logprobs. Measure CoT perturbation sensitivity — if removing a reasoning step changes the final answer *more* under Extended D-RoPE than standard, that supports the prediction from Section 3.4 (the answer depends more on the actual reasoning chain).

**Phase 4 gate (Fine-tuned):** CoT faithfulness improvement should be strongest here. Also measure long-context needle-in-haystack with the needle placed at varying recency positions.

#### 4.3.6 Vulkan-Specific Notes

- **Rope GLSL shaders:** Verify position variable is `int` not `uint`. Negative positions must survive the shader pipeline without wrapping.
- **Global rotation shift** (Phase 3): Can reuse the existing rope shader with a position uniform offset of $-1$ — no new shader required.
- **NTK-aware / YaRN scaling:** If active on the model, verify the scaling function handles negative positions. Some implementations clamp to 0 or use `abs()`. Disable context-length scaling for initial D-RoPE testing to isolate variables.

#### 4.3.7 Foundation: Existing Qwen Attention Work

The recurring attention layer scripts already modify per-layer attention behavior and touch the compute graph at the same depth as D-RoPE. Phase 3 (KV cache modification) shares code paths with key insertion/retrieval. The Vulkan CMake build and Qwen layer work are direct foundations — the implementing agent is not starting from scratch.

---

## 5. Extended D-RoPE: Logical Time Decoupling

### 5.1 Beyond Linear Shifting

The basic D-RoPE (linear backward shift) is the simplest form. Extended variants:

**Logical boundary resets:** Reset or warp the positional frame at reasoning boundaries (step markers, paragraph breaks, logical phases). This decouples logical time from raw token index entirely.

```
Prompt tokens:     positions [-P, ..., -C-1]
CoT Step 1:        positions [-S₁, ..., -1]  ← reset at step boundary
CoT Step 2:        positions [-S₂, ..., -1]  ← reset again
Final answer:      position 0
```

**Curvature-driven warping:** Apply a non-linear position mapping that compresses or expands positional resolution based on content importance (analogous to the MMK's curvature-driven prompt shaping).

**Harmonic buffer analogy:** Map fast/medium/slow timescale traces to different RoPE frequency bands. Recent tokens get high-frequency (fine) positional encoding. Mid-range tokens get medium-frequency. Ancient tokens get low-frequency (coarse). This is the MMK's harmonic buffer architecture applied to attention.

### 5.2 Connection to MMK Harmonic Buffers

The Many-Mind Kernel uses three-timescale EMA buffers (fast/medium/slow) to give agents multi-timescale temporal awareness. D-RoPE is the same principle applied to transformer attention:

| MMK Harmonic Buffers | D-RoPE |
|---|---|
| Position 0 = current emotional state | Position 0 = current token |
| Fast buffer (τ ≈ 3-5 ticks) | High-frequency RoPE bands (fine discrimination) |
| Slow buffer (τ ≈ 50-100 ticks) | Low-frequency RoPE bands (coarse discrimination) |
| Curvature drives prompt truncation | Curvature could drive positional warping |
| λ(t) balances emotional vs. domain recall | Could balance recency vs. content attention |

D-RoPE and harmonic buffers are the same temporal-focus mechanism operating at different architectural levels: one in the attention geometry, the other in the cognitive state management.

---

## 6. Experimental Predictions

### 6.1 Testable Hypotheses

1. **CoT faithfulness increases under D-RoPE.** Measure by ablating/perturbing CoT steps and checking if the final answer changes proportionally. Under standard RoPE, ablations often don't affect the answer (CoT is decorative). Under D-RoPE, they should.

2. **Long-context retrieval improves for recent information.** Standard RoPE models struggle with "needle in a haystack" when the needle is recent but the haystack is long. D-RoPE gives the needle (recent tokens) position 0 — maximum resolution.

3. **Early-prompt dominance decreases.** Standard RoPE gives the system prompt the best positional encoding. D-RoPE gives it the worst (largest negative position). The model should rely less on early fixed text and more on recent dynamic content.

4. **Effective context window utilization improves.** By giving every token a journey through the positional spectrum (instead of a fixed seat), D-RoPE provides richer position-content disentanglement, potentially improving the model's ability to use the full context window.

### 6.2 Failure Modes to Watch

- **Instruction-following degradation:** If the system prompt (at large negative position) loses too much positional resolution, instruction adherence may suffer. May need a "protected zone" for system tokens.
- **Positional confusion during training:** If trained from scratch with D-RoPE, the model must learn that position 0 is always "now" and negative positions are "past" — a different convention than standard training assumes.

---

## 7. Philosophical Note

D-RoPE encodes a view of time that matches lived experience: the present is always here (position 0), the past recedes, and every moment passes through every stage of "nowness" on its way to becoming distant memory. Standard RoPE encodes a view of time as a fixed tape — events are stamped with their birth position and never change.

The CoT failure mode — where the model "already knows" the answer from early in the prompt but the generated reasoning is decorative — mirrors a human cognitive phenomenon: post-hoc rationalization. The reasoning narrative is constructed after the conclusion is reached. D-RoPE doesn't eliminate this, but it reduces the *structural* incentive for it by keeping the reasoning and conclusion in a coherent positional frame.

This connects to emergence over control (KO25): rather than engineering faithfulness through prompting tricks or training incentives, D-RoPE creates geometric conditions where faithful reasoning is the path of least resistance. The positional structure encourages coherence; it doesn't enforce it.

---

*Developed by Ken Ong with Kato/Copilot. Documented March 2026.*
*Cross-references: KO14 (Temporal Encoding), KO5 (Resonant Credit Assignment), KO24 (Resonant Memory Dynamics)*
*Implementation target: llama.cpp RoPE patch for inference-time testing.*
