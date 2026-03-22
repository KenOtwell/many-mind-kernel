"""Add residual magnitude as 9th dimension to emotional_bases_8d.npz.

The residual magnitude measures how much of a text's embedding is NOT
captured by the 8 emotional projection axes. It serves as:
- An emotionality meter (low residual = high emotional content)
- A domain-content tiebreaker in emotional similarity search
- Orthogonal to all 8 emotion axes by construction

Output: emotional_bases_9d.npz with the same structure plus '_residual_info' metadata.
"""
import numpy as np

SRC = r'C:\Users\Ken\Projects\many-mind-kernel\shared\data\emotional_bases_8d.npz'
DST = r'C:\Users\Ken\Projects\many-mind-kernel\shared\data\emotional_bases_9d.npz'

data = np.load(SRC)
emotion_names = list(data['_emotion_names'])
bases_dict = {e: data[e] for e in emotion_names}
raw_dict = {e: data[f'{e}_raw'] for e in emotion_names}

# New axis order: 8 emotions + residual
all_names = emotion_names + ['residual']

save_dict = {}
for e in emotion_names:
    save_dict[e] = bases_dict[e]            # orthogonalized
    save_dict[f'{e}_raw'] = raw_dict[e]     # raw for reference

# Residual is not a basis vector in 384d space — it's computed at projection time.
# Store a sentinel to document this. The actual computation is:
#   coeffs = bases @ emb_norm          (8d projection)
#   recon = coeffs @ bases             (384d reconstruction)
#   residual_mag = ||emb_norm - recon|| (9th dimension)
save_dict['residual'] = np.zeros(384)  # placeholder — not used for projection
save_dict['residual_raw'] = np.zeros(384)
save_dict['_emotion_names'] = np.array(all_names)
save_dict['_residual_note'] = np.array([
    'residual is computed, not projected.',
    'residual_mag = norm(emb_norm - (coeffs @ bases))',
    'coeffs = bases_8x384 @ emb_norm_384',
])

np.savez(DST, **save_dict)

print(f'Saved {DST}')
print(f'Axes: {all_names}')
print(f'Load: data = np.load("emotional_bases_9d.npz")')
print(f'  emotion_bases = {{k: data[k] for k in data["_emotion_names"][:8]}}')
print(f'  # 9th dim (residual) computed at projection time, not from a basis vector')
