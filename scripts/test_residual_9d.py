"""Test 9d emotional signature: 8 emotion axes + residual magnitude.

Explores whether the emotion-subtracted residual carries domain/role structure.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

data = np.load(r'C:\Users\Ken\Projects\many-mind-kernel\shared\data\emotional_bases_8d.npz')
emotion_names = list(data['_emotion_names'])
bases = np.array([data[e] for e in emotion_names])  # (8, 384)

model = SentenceTransformer('all-MiniLM-L6-v2')

test_words = [
    # Pure emotions
    'terrified', 'furious', 'heartbroken', 'elated', 'revolting', 'adoring', 'thrilling',
    # Role/domain words (Skyrim flavored)
    'dragon', 'sword', 'tavern', 'dungeon', 'wedding', 'betrayal', 'treasure',
    'storm', 'home', 'poison', 'guard', 'merchant', 'priest', 'thief', 'warrior',
    # Relationships
    'ally', 'enemy', 'stranger', 'companion', 'rival',
    # Neutrals
    'table', 'the', 'walk', 'stone', 'number',
]

embeddings = model.encode(test_words)

print('=== 9d SIGNATURES (8 emotion + residual magnitude) ===')
print()
axes = emotion_names + ['residual']
header = f"{'Word':>14}" + ''.join(f'{a:>8}' for a in axes)
print(header)
print('-' * len(header))

sigs_9d = []
residuals_384d = []

for word, emb in zip(test_words, embeddings):
    emb_norm = emb / np.linalg.norm(emb)
    coeffs = bases @ emb_norm  # 8d projection
    recon = coeffs @ bases      # reconstruction in 384d
    residual = emb_norm - recon  # what's left
    res_mag = np.linalg.norm(residual)

    sig = list(coeffs) + [res_mag]
    sigs_9d.append(sig)
    residuals_384d.append(residual)

    row = f'{word:>14}' + ''.join(f'{v:>8.3f}' for v in sig)
    print(row)

print()
print('=== RESIDUAL-SPACE CLUSTERING (cosine sim on emotion-subtracted vectors) ===')
print('Looking for domain/role structure in the non-emotional residual...')
print()

residuals = np.array(residuals_384d)
# Normalize residuals for cosine sim
res_norms = np.linalg.norm(residuals, axis=1, keepdims=True)
res_norms = np.where(res_norms < 1e-10, 1.0, res_norms)
res_normed = residuals / res_norms

sim_matrix = res_normed @ res_normed.T

# Find most similar pairs (excluding self)
pairs = []
for i in range(len(test_words)):
    for j in range(i + 1, len(test_words)):
        pairs.append((sim_matrix[i, j], test_words[i], test_words[j]))

pairs.sort(reverse=True)

print('Top 15 most similar pairs in RESIDUAL space (emotion removed):')
for sim, w1, w2 in pairs[:15]:
    print(f'  {w1:>14} <-> {w2:<14}  {sim:+.4f}')

print()
print('Top 10 most DISSIMILAR pairs:')
pairs.sort()
for sim, w1, w2 in pairs[:10]:
    print(f'  {w1:>14} <-> {w2:<14}  {sim:+.4f}')

print()
print('=== ROLE CLUSTER CHECK ===')
print('Average residual-space similarity within categories:')
print()

categories = {
    'combat':     ['sword', 'dragon', 'guard', 'warrior', 'enemy'],
    'social':     ['merchant', 'priest', 'companion', 'ally', 'wedding'],
    'place':      ['tavern', 'dungeon', 'home', 'storm'],
    'morality':   ['betrayal', 'thief', 'poison', 'rival'],
    'neutral':    ['table', 'the', 'walk', 'stone', 'number'],
    'emotion':    ['terrified', 'furious', 'heartbroken', 'elated', 'adoring'],
}

word_to_idx = {w: i for i, w in enumerate(test_words)}

for cat, words in categories.items():
    idxs = [word_to_idx[w] for w in words if w in word_to_idx]
    if len(idxs) < 2:
        continue
    intra_sims = []
    for a in range(len(idxs)):
        for b in range(a + 1, len(idxs)):
            intra_sims.append(sim_matrix[idxs[a], idxs[b]])
    avg_sim = np.mean(intra_sims)
    mn = np.min(intra_sims)
    mx = np.max(intra_sims)
    print(f'  {cat:>10}: avg={avg_sim:+.4f}  range=[{mn:+.4f}, {mx:+.4f}]  {words}')

# Cross-category average for baseline
all_sims = []
for i in range(len(test_words)):
    for j in range(i + 1, len(test_words)):
        all_sims.append(sim_matrix[i, j])
print(f'  {"baseline":>10}: avg={np.mean(all_sims):+.4f}  (all pairs)')
