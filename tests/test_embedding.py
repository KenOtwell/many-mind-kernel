"""Tests for mindcore.embedding.

These tests load the real all-MiniLM-L6-v2 model. First run downloads
~80MB to ~/.cache/huggingface/. Subsequent runs use the cached model.
"""
from __future__ import annotations

import numpy as np
import pytest

from shared.config import settings


@pytest.fixture(autouse=True)
def _reset_embedding_state():
    """Reset the module-level model between tests."""
    import mindcore.embedding as mod
    saved = mod._model
    yield
    mod._model = saved


class TestLoadModel:
    def test_loads_successfully(self):
        from mindcore.embedding import load_model, is_loaded
        load_model()
        assert is_loaded()

    def test_idempotent(self):
        from mindcore.embedding import load_model, is_loaded
        load_model()
        load_model()  # Should not raise
        assert is_loaded()


class TestEmbed:
    def test_single_text_shape(self):
        from mindcore.embedding import embed_one, load_model
        load_model()
        vec = embed_one("I am sworn to carry your burdens.")
        assert vec.shape == (settings.embedding.semantic_dim,)
        assert vec.dtype == np.float32

    def test_batch_shape(self):
        from mindcore.embedding import embed, load_model
        load_model()
        texts = ["Hello", "World", "Skyrim"]
        vecs = embed(texts)
        assert vecs.shape == (3, settings.embedding.semantic_dim)

    def test_empty_batch(self):
        from mindcore.embedding import embed, load_model
        load_model()
        vecs = embed([])
        assert vecs.shape == (0, settings.embedding.semantic_dim)

    def test_semantic_similarity(self):
        """Similar sentences should have higher cosine similarity than dissimilar ones."""
        from mindcore.embedding import embed, load_model
        load_model()
        vecs = embed([
            "I will protect you with my life.",     # protective
            "I am sworn to carry your burdens.",    # protective (similar)
            "The cheese wheel costs three septims.", # unrelated
        ])
        # Cosine similarity
        def cos_sim(a, b):
            return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

        sim_related = cos_sim(vecs[0], vecs[1])
        sim_unrelated = cos_sim(vecs[0], vecs[2])
        assert sim_related > sim_unrelated

    def test_nonzero_output(self):
        """Embedding should not be all zeros."""
        from mindcore.embedding import embed_one, load_model
        load_model()
        vec = embed_one("Test")
        assert np.linalg.norm(vec) > 0
