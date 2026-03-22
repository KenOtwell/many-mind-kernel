"""Compute emotional projection basis vectors from embedding space.

Phase 1: Validate synonym/antonym sets via Datamuse thesaurus API
  - Bidirectional synonym check: keep only mutual references
  - Seeds always kept (hand-curated, bypass bidirectional check)
  - Multi-word phrases dropped (single tokens only)
  - Score-weighted embedding means (Datamuse relevance score)
  - Cross-validate: positive antonyms should appear in negative synonyms

Phase 2: Compute projection bases (score-weighted)
  v = weighted_mean(embed(synonyms)) - weighted_mean(embed(antonyms))
  v = v / ||v||

Phase 3: Orthogonality check and emotional signature tests
"""

import time
import numpy as np
import requests
from sentence_transformers import SentenceTransformer


# --- Datamuse API (free, no key) ---

DATAMUSE_URL = "https://api.datamuse.com/words"
_syn_cache = {}
_ant_cache = {}


def get_synonyms_scored(word):
    """Get synonyms with relevance scores from Datamuse. Cached."""
    if word not in _syn_cache:
        try:
            resp = requests.get(DATAMUSE_URL, params={"rel_syn": word, "max": 30}, timeout=5)
            _syn_cache[word] = [(w["word"], w.get("score", 0)) for w in resp.json()]
        except Exception:
            _syn_cache[word] = []
        time.sleep(0.05)
    return _syn_cache[word]


def get_antonyms_scored(word):
    """Get antonyms with relevance scores from Datamuse. Cached."""
    if word not in _ant_cache:
        try:
            resp = requests.get(DATAMUSE_URL, params={"rel_ant": word, "max": 30}, timeout=5)
            _ant_cache[word] = [(w["word"], w.get("score", 0)) for w in resp.json()]
        except Exception:
            _ant_cache[word] = []
        time.sleep(0.05)
    return _ant_cache[word]


def is_single_token(word):
    """Keep only single tokens. Drop multi-word phrases (spaces).
    Hyphenated compounds are fine — tokenizers treat them as one unit."""
    return " " not in word


# Max words per axis after thesaurus expansion. Prevents large sets from
# drifting toward generic sentiment and drowning smaller, sharper axes.
TOP_N = 25


# --- Seed words: starting point before thesaurus validation ---

SEED_EMOTIONS = {
    "fear": {
        "seeds": ["fear", "afraid", "scared", "terrified", "anxious", "uneasy", "startled"],
        "antonym_seeds": ["brave", "calm", "fearless", "confident"],
    },
    "anger": {
        "seeds": ["anger", "angry", "furious", "irritated", "enraged", "annoyed"],
        "antonym_seeds": ["peaceful", "calm", "patient", "forgiving"],
    },
    "joy": {
        "seeds": ["joyful", "happy", "grateful", "content", "satisfied", "blessed",
                  "fulfilled", "celebrating", "thankful", "pleased"],
        "antonym_seeds": ["disappointed", "dissatisfied", "unfulfilled", "regretful", "letdown"],
    },
    "sadness": {
        "seeds": ["sad", "sorrowful", "grieving", "mournful", "heartbroken",
                  "lonely", "melancholy", "bereaved", "forlorn", "despairing"],
        "antonym_seeds": ["comforted", "consoled", "healed", "hopeful", "uplifted"],
    },
    "love": {
        "seeds": ["love", "affection", "fondness", "warmth", "caring"],
        "antonym_seeds": ["hate", "coldness", "indifference"],
    },
    "disgust": {
        "seeds": ["disgusted", "revolted", "repulsed", "appalled", "loathing",
                  "contempt", "abhorrent", "offensive", "vile", "repugnant"],
        "antonym_seeds": ["appealing", "attractive", "admirable", "delightful", "pleasant"],
    },
    "safety": {
        "seeds": ["safety", "safe", "secure", "protected", "comfortable"],
        "antonym_seeds": ["threatened", "unsafe", "exposed"],
    },
    "excitement": {
        "seeds": ["excited", "eager", "anticipating", "expectant", "hopeful",
                  "impatient", "yearning", "restless"],
        "antonym_seeds": ["dreading", "apathetic", "indifferent", "resigned", "bored"],
    },
}

# Natural polarity pairings for cross-validation
POLARITY_PAIRS = [
    ("joy", "sadness"),
    ("love", "disgust"),
    ("safety", "fear"),
    ("excitement", "anger"),  # arousal vs frustrated arousal
]


def validate_emotion_axes():
    """Use Datamuse to validate synonym/antonym sets via bidirectional reference.

    Algorithm:
      1. For each seed word, get thesaurus synonyms (forward lookup)
      2. Filter to single tokens only (drop multi-word phrases)
      3. For each candidate, get ITS synonyms (reverse lookup)
      4. Keep only candidates with bidirectional link to a seed
      5. Seeds always kept unconditionally (hand-curated)
      6. Score = best Datamuse relevance score across all lookups that returned it
      7. Cross-check polarity pairs
    """
    print("\n" + "=" * 60)
    print("PHASE 1: THESAURUS VALIDATION (Datamuse API)")
    print("=" * 60)

    validated = {}

    for emotion, data in SEED_EMOTIONS.items():
        seeds = set(data["seeds"])
        ant_seeds = set(data["antonym_seeds"])

        print(f"\n--- {emotion.upper()} ---")
        print(f"  Seeds: {sorted(seeds)}")

        # --- Synonym validation ---
        # Forward: get scored synonym candidates, single-token only
        candidates = {}  # word -> best_score
        for seed in seeds:
            for word, score in get_synonyms_scored(seed):
                if is_single_token(word):
                    candidates[word] = max(candidates.get(word, 0), score)

        print(f"  Forward pass: {len(candidates)} single-token candidates")

        # Reverse: bidirectional check — candidate's synonyms must include a seed
        bidirectional = {}
        for candidate, score in candidates.items():
            reverse_syns = {w for w, s in get_synonyms_scored(candidate)}
            if reverse_syns & seeds:
                bidirectional[candidate] = score

        dropped = set(candidates) - set(bidirectional)
        print(f"  Bidirectional: {len(bidirectional)} kept, {len(dropped)} dropped")
        if dropped:
            print(f"  Dropped: {sorted(dropped)}")

        # Seeds always get max score — they're hand-curated and must anchor the axis
        max_score = max(bidirectional.values()) if bidirectional else 10000
        for seed in seeds:
            if is_single_token(seed):
                bidirectional[seed] = max_score  # overwrite even if already present at lower score

        # Sort by score descending, cap at TOP_N
        syn_ranked = sorted(bidirectional.items(), key=lambda x: -x[1])[:TOP_N]
        print(f"  Final synonyms ({len(syn_ranked)}, capped at {TOP_N}):")
        for w, s in syn_ranked:
            tag = " [seed]" if w in seeds else ""
            print(f"    {w:<20} score={s:>6}{tag}")

        # --- Antonym validation ---
        syn_words = set(bidirectional.keys())

        # Forward: get antonyms of each validated synonym + seed antonyms
        ant_candidates = {}
        for seed in ant_seeds:
            if is_single_token(seed):
                ant_candidates[seed] = ant_candidates.get(seed, 0)
        for syn in syn_words:
            for word, score in get_antonyms_scored(syn):
                if is_single_token(word):
                    ant_candidates[word] = max(ant_candidates.get(word, 0), score)

        print(f"  Antonym candidates: {len(ant_candidates)}")

        # Reverse: antonym's antonyms should include a validated synonym
        ant_bidirectional = {}
        for candidate, score in ant_candidates.items():
            reverse_ants = {w for w, s in get_antonyms_scored(candidate)}
            if reverse_ants & syn_words:
                ant_bidirectional[candidate] = score

        # Antonym seeds always get max score
        ant_max = max(ant_bidirectional.values()) if ant_bidirectional else 10000
        for seed in ant_seeds:
            if is_single_token(seed):
                ant_bidirectional[seed] = ant_max  # overwrite even if already present

        ant_dropped = set(ant_candidates) - set(ant_bidirectional)
        ant_ranked = sorted(ant_bidirectional.items(), key=lambda x: -x[1])[:TOP_N]
        print(f"  Antonyms validated: {len(ant_bidirectional)} kept, {len(ant_dropped)} dropped (capped at {TOP_N})")
        if ant_dropped:
            print(f"  Dropped: {sorted(ant_dropped)}")
        print(f"  Final antonyms ({len(ant_ranked)}):")
        for w, s in ant_ranked:
            tag = " [seed]" if w in ant_seeds else ""
            print(f"    {w:<20} score={s:>6}{tag}")

        validated[emotion] = {
            "synonyms": syn_ranked,   # [(word, score), ...]
            "antonyms": ant_ranked,
        }

    # --- Cross-validate polarity pairs ---
    print(f"\n--- POLARITY CROSS-VALIDATION ---")
    for pos, neg in POLARITY_PAIRS:
        pos_ant_words = {w for w, s in validated[pos]["antonyms"]}
        neg_syn_words = {w for w, s in validated[neg]["synonyms"]}
        neg_ant_words = {w for w, s in validated[neg]["antonyms"]}
        pos_syn_words = {w for w, s in validated[pos]["synonyms"]}

        overlap_a = pos_ant_words & neg_syn_words
        overlap_b = neg_ant_words & pos_syn_words

        print(f"\n  {pos} <-> {neg}:")
        print(f"    {pos} antonyms in {neg} synonyms: {sorted(overlap_a) if overlap_a else 'NONE'}")
        print(f"    {neg} antonyms in {pos} synonyms: {sorted(overlap_b) if overlap_b else 'NONE'}")
        if not overlap_a and not overlap_b:
            print(f"    !! No cross-reference -- these axes may not be properly opposed")

    print(f"\n  API calls: {len(_syn_cache)} syn + {len(_ant_cache)} ant = {len(_syn_cache) + len(_ant_cache)} total")
    return validated


# Priority order for Gram-Schmidt: cleanest/sharpest axes first.
# Earlier vectors are preserved more faithfully; later ones get adjusted.
# Order rationale: fear/anger are tightest clusters, love/disgust are
# well-separated, excitement is newly tuned, then the overlapping pairs
# (sadness, joy, safety) go last so their shared components get cleaned.
GS_PRIORITY = ["fear", "anger", "love", "disgust", "excitement", "sadness", "joy", "safety"]


def gram_schmidt_orthogonalize(bases, priority_order):
    """Modified Gram-Schmidt orthogonalization of emotion basis vectors.

    Processes vectors in priority_order. Earlier vectors are preserved
    more faithfully; later vectors have prior components projected out.
    This forces exact orthogonality while preserving semantic direction.

    Returns:
        orthogonal_bases: dict of {emotion: unit_vector}
        drift: dict of {emotion: cosine_similarity_to_original}
    """
    ortho = {}
    drift = {}

    for emotion in priority_order:
        v = bases[emotion].copy()

        # Project out components of all previously orthogonalized vectors
        for prev_emotion in ortho:
            u = ortho[prev_emotion]
            v = v - np.dot(v, u) * u

        # Normalize
        norm = np.linalg.norm(v)
        if norm < 1e-10:
            print(f"  WARNING: {emotion} collapsed to near-zero after orthogonalization!")
            v = bases[emotion].copy()
            norm = np.linalg.norm(v)

        v = v / norm

        # Track how much the direction drifted from the raw semantic vector
        drift[emotion] = float(np.dot(v, bases[emotion]))
        ortho[emotion] = v

    return ortho, drift


def compute_projection_bases(model, emotion_axes):
    """Compute score-weighted projection vectors via synonym-antonym differential.

    For each axis:
      syn_mean = weighted_mean(embed(synonyms), weights=datamuse_scores)
      ant_mean = weighted_mean(embed(antonyms), weights=datamuse_scores)
      v = normalize(syn_mean - ant_mean)

    Higher-scoring words (stronger thesaurus match) contribute more to the direction.
    """
    bases = {}
    for emotion, data in emotion_axes.items():
        syn_words = [w for w, s in data["synonyms"]]
        syn_scores = np.array([s for w, s in data["synonyms"]], dtype=np.float64)
        ant_words = [w for w, s in data["antonyms"]]
        ant_scores = np.array([s for w, s in data["antonyms"]], dtype=np.float64)

        syn_embeddings = model.encode(syn_words)
        ant_embeddings = model.encode(ant_words)

        # Score-weighted means: high-confidence words pull harder
        syn_weights = syn_scores / syn_scores.sum()
        ant_weights = ant_scores / ant_scores.sum()

        syn_mean = np.average(syn_embeddings, axis=0, weights=syn_weights)
        ant_mean = np.average(ant_embeddings, axis=0, weights=ant_weights)

        v = syn_mean - ant_mean
        v = v / np.linalg.norm(v)
        bases[emotion] = v
    return bases


def emotional_signature(text, model, bases):
    """Project a text embedding onto the emotion basis -> 8d emotional coordinate."""
    embedding = model.encode([text])[0]
    embedding_norm = embedding / np.linalg.norm(embedding)

    return {emotion: float(np.dot(embedding_norm, basis))
            for emotion, basis in bases.items()}


def check_orthogonality(bases):
    """Pairwise cosine similarity matrix between all basis vectors."""
    emotions = list(bases.keys())
    n = len(emotions)
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i][j] = np.dot(bases[emotions[i]], bases[emotions[j]])
    return emotions, matrix


def print_orthogonality_report(emotions, matrix):
    """Print the similarity matrix and flag problematic pairs."""
    print("\n=== ORTHOGONALITY MATRIX (cosine similarity) ===\n")

    # Header
    print(f"{'':>12}" + "".join(f"{e:>10}" for e in emotions))
    for i, emotion in enumerate(emotions):
        row = f"{emotion:>12}" + "".join(f"{matrix[i][j]:>10.3f}" for j in range(len(emotions)))
        print(row)

    # Pair analysis
    print("\n=== PAIR ANALYSIS ===\n")
    warnings = 0
    for i in range(len(emotions)):
        for j in range(i + 1, len(emotions)):
            sim = matrix[i][j]
            if sim > 0.4:
                print(f"  {emotions[i]:>10} x {emotions[j]:<10} = {sim:+.3f}  !! TOO CLOSE - adjust word sets")
                warnings += 1
            elif sim < -0.4:
                print(f"  {emotions[i]:>10} x {emotions[j]:<10} = {sim:+.3f}  ~~ opposed (ok, natural polarity)")
            elif abs(sim) < 0.1:
                print(f"  {emotions[i]:>10} x {emotions[j]:<10} = {sim:+.3f}  .. near-orthogonal")
            else:
                print(f"  {emotions[i]:>10} x {emotions[j]:<10} = {sim:+.3f}")

    if warnings:
        print(f"\n  {warnings} pair(s) above 0.4 threshold - consider adjusting synonym/antonym sets")
    else:
        print(f"\n  All pairs within tolerance.")


def print_signatures(test_words, model, bases):
    """Print emotional signatures for a list of test words."""
    emotions = list(bases.keys())

    print(f"\n=== EMOTIONAL SIGNATURES ===\n")
    print(f"{'word':>14}" + "".join(f"{e:>8}" for e in emotions) + "   strongest")
    print("-" * (14 + 8 * len(emotions) + 20))

    for word in test_words:
        sig = emotional_signature(word, model, bases)
        strongest = max(sig, key=lambda k: abs(sig[k]))
        strongest_val = sig[strongest]
        row = (f"{word:>14}"
               + "".join(f"{sig[e]:>8.3f}" for e in emotions)
               + f"   {strongest} ({strongest_val:+.3f})")
        print(row)


def main():
    # --- Phase 1: Thesaurus validation ---
    validated_axes = validate_emotion_axes()

    # --- Phase 2: Compute bases ---
    print("\n" + "=" * 60)
    print("PHASE 2: PROJECTION BASIS COMPUTATION")
    print("=" * 60)

    print("\nLoading all-MiniLM-L6-v2...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Computing 8 raw emotion projection bases (384d each)...\n")
    raw_bases = compute_projection_bases(model, validated_axes)

    for emotion, v in raw_bases.items():
        syns = validated_axes[emotion]["synonyms"]
        ants = validated_axes[emotion]["antonyms"]
        top_syn = syns[0][0] if syns else "?"
        top_ant = ants[0][0] if ants else "?"
        print(f"  {emotion:>10}: {len(syns)} syns (top: {top_syn}), {len(ants)} ants (top: {top_ant})")

    # --- Raw orthogonality (before Gram-Schmidt) ---
    print("\n--- RAW BASIS (before orthogonalization) ---")
    emotions_raw, matrix_raw = check_orthogonality(raw_bases)
    print_orthogonality_report(emotions_raw, matrix_raw)

    # --- Phase 2.5: Gram-Schmidt ---
    print("\n" + "=" * 60)
    print("PHASE 2.5: GRAM-SCHMIDT ORTHOGONALIZATION")
    print("=" * 60)

    print(f"\nPriority order: {' > '.join(GS_PRIORITY)}")
    print("(earlier = preserved more faithfully, later = adjusted more)\n")

    bases, drift = gram_schmidt_orthogonalize(raw_bases, GS_PRIORITY)

    print("  Drift report (cosine similarity to raw direction):")
    for emotion in GS_PRIORITY:
        d = drift[emotion]
        bar = "*" * int(d * 20)
        status = "(untouched)" if d > 0.999 else "(minor adjustment)" if d > 0.95 else "(moderate shift)" if d > 0.85 else "(significant reshape)"
        print(f"    {emotion:>10}: {d:.4f}  {bar}  {status}")

    # --- Phase 3: Orthogonality + signatures (on orthogonalized bases) ---
    print("\n" + "=" * 60)
    print("PHASE 3: VALIDATION (orthogonalized)")
    print("=" * 60)

    emotions, matrix = check_orthogonality(bases)
    print_orthogonality_report(emotions, matrix)

    test_words = [
        # Pure emotion words (should light up their axis)
        "terrified", "furious", "elated", "heartbroken",
        "adoring", "revolting", "sanctuary", "thrilling",
        # Skyrim-flavored
        "dragon", "sword", "tavern", "dungeon", "wedding",
        "betrayal", "treasure", "storm", "home", "poison",
        # Neutral / near-zero expected
        "table", "stone", "walk", "the", "number",
    ]
    print_signatures(test_words, model, bases)

    # --- Save both raw and orthogonalized ---
    outpath = "emotional_bases_8d.npz"
    save_dict = {}
    for emotion in bases:
        save_dict[emotion] = bases[emotion]              # orthogonalized (primary)
        save_dict[f"{emotion}_raw"] = raw_bases[emotion]  # raw for reference
    save_dict["_emotion_names"] = np.array(list(bases.keys()))
    np.savez(outpath, **save_dict)
    print(f"\nBases saved to {outpath} (orthogonalized + raw)")
    print("Load: data = np.load('emotional_bases_8d.npz')")
    print("  bases = {k: data[k] for k in data['_emotion_names']}")
    print("  raw   = {k: data[f'{k}_raw'] for k in data['_emotion_names']}")


if __name__ == "__main__":
    main()
