"""Tests for mindcore.emotional."""
from __future__ import annotations

import numpy as np
import pytest

from shared.constants import EMOTIONAL_DIM, EMOTIONAL_AXES

# Reset module state between tests
@pytest.fixture(autouse=True)
def _reset_projection_state():
    """Reset the module-level bases so each test starts clean."""
    import mindcore.emotional as mod
    saved = mod._bases
    mod._bases = None
    yield
    mod._bases = saved


class TestLoadBases:
    def test_loads_from_default_path(self):
        from mindcore.emotional import load_bases, is_loaded
        load_bases()
        assert is_loaded()

    def test_bases_shape(self):
        from mindcore.emotional import load_bases
        import mindcore.emotional as mod
        load_bases()
        assert mod._bases is not None
        assert mod._bases.shape == (8, 384)

    def test_bases_orthonormal(self):
        """GS bases should be orthonormal — dot products ~0 off-diagonal."""
        from mindcore.emotional import load_bases
        import mindcore.emotional as mod
        load_bases()
        dot = mod._bases @ mod._bases.T
        # Diagonal should be ~1.0
        np.testing.assert_allclose(np.diag(dot), 1.0, atol=1e-5)
        # Off-diagonal should be ~0.0
        off_diag = dot - np.eye(8)
        assert np.abs(off_diag).max() < 1e-5


class TestProject:
    def test_output_length(self):
        from mindcore.emotional import project, load_bases
        load_bases()
        emb = np.random.randn(384).astype(np.float32)
        sem = project(emb)
        assert len(sem) == EMOTIONAL_DIM

    def test_residual_nonnegative(self):
        """Residual magnitude (axis 8) should always be >= 0."""
        from mindcore.emotional import project, load_bases
        load_bases()
        for _ in range(10):
            emb = np.random.randn(384).astype(np.float32)
            sem = project(emb)
            assert sem[8] >= 0.0

    def test_zero_vector_returns_zeros(self):
        from mindcore.emotional import project, load_bases
        load_bases()
        sem = project(np.zeros(384, dtype=np.float32))
        assert all(v == 0.0 for v in sem)

    def test_known_direction(self):
        """A vector aligned with the fear basis should project high on fear."""
        from mindcore.emotional import load_bases, project
        import mindcore.emotional as mod
        load_bases()
        # Use the fear basis vector itself as input
        fear_basis = mod._bases[0].copy()
        sem = project(fear_basis)
        # Fear (axis 0) should be ~1.0 (it's a unit vector projected onto itself)
        assert sem[0] > 0.9
        # Other axes should be near zero (orthogonal)
        for i in range(1, 8):
            assert abs(sem[i]) < 0.1
        # Residual should be near zero (perfectly captured)
        assert sem[8] < 0.1


class TestProjectBatch:
    def test_batch_matches_single(self):
        """Batch project should give same results as individual projects."""
        from mindcore.emotional import project, project_batch, load_bases
        load_bases()
        embs = np.random.randn(5, 384).astype(np.float32)
        batch_result = project_batch(embs)
        for i in range(5):
            single_result = project(embs[i])
            np.testing.assert_allclose(
                batch_result[i], single_result, atol=1e-5,
            )

    def test_batch_shape(self):
        from mindcore.emotional import project_batch, load_bases
        load_bases()
        embs = np.random.randn(10, 384).astype(np.float32)
        result = project_batch(embs)
        assert result.shape == (10, EMOTIONAL_DIM)

    def test_empty_batch(self):
        from mindcore.emotional import project_batch, load_bases
        load_bases()
        result = project_batch(np.empty((0, 384), dtype=np.float32))
        assert result.shape == (0, EMOTIONAL_DIM)
