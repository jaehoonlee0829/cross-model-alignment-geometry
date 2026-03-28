"""Unit tests for CKA computation.

Run: python -m pytest tests/test_cka.py -v
"""

import numpy as np
import pytest

from src.cka_analysis import compute_cka, linear_kernel, rbf_kernel, hsic


class TestLinearCKA:
    """Test CKA with linear kernel."""

    def test_identical_representations(self):
        """CKA of identical matrices should be ~1.0."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 50))
        cka = compute_cka(X, X, kernel="linear")
        assert cka == pytest.approx(1.0, abs=0.01)

    def test_orthogonal_representations(self):
        """CKA of orthogonal representations should be ~0.0."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 50))
        # Create Y orthogonal to X (in high dim, random is approximately orthogonal)
        Y = rng.normal(size=(100, 50))
        cka = compute_cka(X, Y, kernel="linear")
        # Should be low but not exactly 0 due to finite sample
        assert cka < 0.15

    def test_rotated_representations(self):
        """CKA should be invariant to orthogonal transformations."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(200, 30))

        # Random orthogonal rotation
        Q, _ = np.linalg.qr(rng.normal(size=(30, 30)))
        Y = X @ Q

        cka = compute_cka(X, Y, kernel="linear")
        assert cka == pytest.approx(1.0, abs=0.01)

    def test_scaled_representations(self):
        """CKA should be invariant to isotropic scaling."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 50))
        Y = X * 7.3  # Scale by arbitrary constant

        cka = compute_cka(X, Y, kernel="linear")
        assert cka == pytest.approx(1.0, abs=0.01)

    def test_different_dimensions(self):
        """CKA should work with different feature dimensions."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 30))
        Y = rng.normal(size=(100, 50))

        cka = compute_cka(X, Y, kernel="linear")
        assert 0.0 <= cka <= 1.0

    def test_symmetry(self):
        """CKA(X, Y) should equal CKA(Y, X)."""
        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 30))
        Y = rng.normal(size=(100, 50))

        cka_xy = compute_cka(X, Y, kernel="linear")
        cka_yx = compute_cka(Y, X, kernel="linear")
        assert cka_xy == pytest.approx(cka_yx, abs=1e-10)


class TestProcrustes:
    """Test Procrustes alignment."""

    def test_perfect_rotation(self):
        """Procrustes should perfectly recover an orthogonal rotation."""
        from src.procrustes_alignment import orthogonal_procrustes_alignment

        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 32))

        # Known rotation
        Q, _ = np.linalg.qr(rng.normal(size=(32, 32)))
        Y = X @ Q

        # Split
        result = orthogonal_procrustes_alignment(
            X[:400], Y[:400], X[400:], Y[400:]
        )

        assert result.test_loss < 0.05
        assert result.explained_variance > 0.95
        assert result.orthogonality_error < 1e-10

    def test_dimension_mismatch_raises(self):
        """Procrustes should raise on dimension mismatch."""
        from src.procrustes_alignment import orthogonal_procrustes_alignment

        rng = np.random.default_rng(42)
        X = rng.normal(size=(100, 30))
        Y = rng.normal(size=(100, 50))

        with pytest.raises(ValueError, match="matching dimensions"):
            orthogonal_procrustes_alignment(X[:80], Y[:80], X[80:], Y[80:])


class TestLinearProjection:
    """Test linear projection alignment."""

    def test_mismatched_dimensions(self):
        """Linear projection should work with different d_model."""
        from src.procrustes_alignment import linear_projection_alignment

        rng = np.random.default_rng(42)
        X = rng.normal(size=(500, 64))
        # Y is a linear function of X + noise
        W_true = rng.normal(size=(64, 32)) * 0.1
        Y = X @ W_true + rng.normal(size=(500, 32)) * 0.01

        result = linear_projection_alignment(
            X[:400], Y[:400], X[400:], Y[400:]
        )

        assert result.explained_variance > 0.9
        assert result.d_source == 64
        assert result.d_target == 32


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
