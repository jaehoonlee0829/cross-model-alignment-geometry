"""Learn alignment mappings between activation spaces of different models.

Two approaches:
1. Orthogonal Procrustes (SVD-based): Requires d_model_a == d_model_b.
   Finds the orthogonal matrix W that minimizes ||AW - B||_F.
   Preserves geometric structure (distances, angles).

2. Learned Linear Projection: Works for ANY d_model pair.
   Learns W: R^{d_a} -> R^{d_b} via least-squares or gradient descent.
   More flexible but doesn't preserve geometry.

The Platonic Representation Hypothesis suggests representations converge
up to an orthogonal transformation — so if it holds, Procrustes should
work well for matching d_model. The linear projection is our fallback
for mismatched dimensions, but also interesting in its own right.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from rich.console import Console

console = Console()


@dataclass
class AlignmentResult:
    """Result of learning an alignment mapping."""
    method: str  # "procrustes" or "linear"
    W: np.ndarray  # The mapping matrix
    train_loss: float  # ||AW - B||_F on training data (normalized)
    test_loss: float  # Same on held-out data
    d_source: int
    d_target: int
    explained_variance: float  # How much of target variance is captured
    # For Procrustes only
    orthogonality_error: Optional[float] = None  # ||W^T W - I||_F


def orthogonal_procrustes_alignment(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
) -> AlignmentResult:
    """Learn orthogonal Procrustes mapping: W = argmin ||XW - Y||_F s.t. W^T W = I.

    Requires X and Y to have the same d_model.

    The solution is: W = U @ V^T where U S V^T = SVD(X^T Y).
    This is the classic Procrustes solution from Schönemann (1966).

    Args:
        X_train: Source activations (n_train, d_model)
        Y_train: Target activations (n_train, d_model)
        X_test: Source activations for evaluation (n_test, d_model)
        Y_test: Target activations for evaluation (n_test, d_model)

    Returns:
        AlignmentResult with the orthogonal mapping.
    """
    d_source = X_train.shape[1]
    d_target = Y_train.shape[1]

    if d_source != d_target:
        raise ValueError(
            f"Orthogonal Procrustes requires matching dimensions. "
            f"Got d_source={d_source}, d_target={d_target}. "
            f"Use linear_projection_alignment() instead."
        )

    console.print("[bold]Learning orthogonal Procrustes alignment...[/bold]")

    # Center the data (important for Procrustes)
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    X_centered = X_train - X_mean
    Y_centered = Y_train - Y_mean

    # Solve: W = argmin ||X_centered @ W - Y_centered||_F  s.t. W^T W = I
    # scipy's orthogonal_procrustes returns (W, scale)
    W, scale = orthogonal_procrustes(X_centered, Y_centered)

    # Evaluate on train
    X_mapped_train = (X_train - X_mean) @ W + Y_mean
    train_residual = np.linalg.norm(X_mapped_train - Y_train, "fro")
    train_baseline = np.linalg.norm(Y_train - Y_train.mean(axis=0), "fro")
    train_loss = train_residual / train_baseline

    # Evaluate on test
    X_mapped_test = (X_test - X_mean) @ W + Y_mean
    test_residual = np.linalg.norm(X_mapped_test - Y_test, "fro")
    test_baseline = np.linalg.norm(Y_test - Y_test.mean(axis=0), "fro")
    test_loss = test_residual / test_baseline

    # Explained variance: 1 - (residual_var / total_var)
    total_var = np.var(Y_test, axis=0).sum()
    residual_var = np.var(X_mapped_test - Y_test, axis=0).sum()
    explained_var = 1.0 - (residual_var / total_var) if total_var > 0 else 0.0

    # Orthogonality check
    orth_error = float(np.linalg.norm(W.T @ W - np.eye(d_source), "fro"))

    console.print(f"  Train loss (normalized): {train_loss:.4f}")
    console.print(f"  Test loss (normalized):  {test_loss:.4f}")
    console.print(f"  Explained variance:      {explained_var:.4f}")
    console.print(f"  Orthogonality error:     {orth_error:.2e}")

    # Store centering info in the matrix for later use
    # (We'll need to apply centering when using W for oracle transfer)
    result = AlignmentResult(
        method="procrustes",
        W=W,
        train_loss=float(train_loss),
        test_loss=float(test_loss),
        d_source=d_source,
        d_target=d_target,
        explained_variance=float(explained_var),
        orthogonality_error=orth_error,
    )

    # Store centering params as extra attributes
    result._X_mean = X_mean  # type: ignore
    result._Y_mean = Y_mean  # type: ignore

    return result


def linear_projection_alignment(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    regularization: float = 1e-4,
) -> AlignmentResult:
    """Learn a linear projection W: R^{d_a} -> R^{d_b} via ridge regression.

    Works for any d_model pair. Solves:
        W = argmin ||XW - Y||_F^2 + λ||W||_F^2

    Closed-form solution: W = (X^T X + λI)^{-1} X^T Y

    Args:
        X_train: Source activations (n_train, d_source)
        Y_train: Target activations (n_train, d_target)
        X_test: Source activations (n_test, d_source)
        Y_test: Target activations (n_test, d_target)
        regularization: Ridge regularization strength

    Returns:
        AlignmentResult with the linear projection.
    """
    d_source = X_train.shape[1]
    d_target = Y_train.shape[1]

    console.print(
        f"[bold]Learning linear projection: "
        f"R^{d_source} -> R^{d_target} (λ={regularization})[/bold]"
    )

    # Center data
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    X_centered = X_train - X_mean
    Y_centered = Y_train - Y_mean

    # Ridge regression closed-form
    # W = (X^T X + λI)^{-1} X^T Y
    XtX = X_centered.T @ X_centered
    XtY = X_centered.T @ Y_centered
    W = np.linalg.solve(
        XtX + regularization * np.eye(d_source),
        XtY,
    )

    # Evaluate on train
    X_mapped_train = (X_train - X_mean) @ W + Y_mean
    train_residual = np.linalg.norm(X_mapped_train - Y_train, "fro")
    train_baseline = np.linalg.norm(Y_train - Y_train.mean(axis=0), "fro")
    train_loss = train_residual / train_baseline

    # Evaluate on test
    X_mapped_test = (X_test - X_mean) @ W + Y_mean
    test_residual = np.linalg.norm(X_mapped_test - Y_test, "fro")
    test_baseline = np.linalg.norm(Y_test - Y_test.mean(axis=0), "fro")
    test_loss = test_residual / test_baseline

    # Explained variance
    total_var = np.var(Y_test, axis=0).sum()
    residual_var = np.var(X_mapped_test - Y_test, axis=0).sum()
    explained_var = 1.0 - (residual_var / total_var) if total_var > 0 else 0.0

    console.print(f"  Train loss (normalized): {train_loss:.4f}")
    console.print(f"  Test loss (normalized):  {test_loss:.4f}")
    console.print(f"  Explained variance:      {explained_var:.4f}")
    console.print(f"  W shape: {W.shape}, rank: {np.linalg.matrix_rank(W)}")

    result = AlignmentResult(
        method="linear",
        W=W,
        train_loss=float(train_loss),
        test_loss=float(test_loss),
        d_source=d_source,
        d_target=d_target,
        explained_variance=float(explained_var),
    )

    result._X_mean = X_mean  # type: ignore
    result._Y_mean = Y_mean  # type: ignore

    return result


def learn_alignment(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    method: str = "both",
    train_fraction: float = 0.8,
    regularization: float = 1e-4,
    seed: int = 42,
) -> dict[str, AlignmentResult]:
    """Learn alignment mapping(s) between two activation sets.

    Args:
        acts_a: Source activations (n_samples, d_source)
        acts_b: Target activations (n_samples, d_target)
        method: "procrustes", "linear", or "both"
        train_fraction: Fraction of data for training
        regularization: Ridge regularization for linear method
        seed: Random seed for train/test split

    Returns:
        Dict mapping method name -> AlignmentResult
    """
    n = acts_a.shape[0]
    d_source = acts_a.shape[1]
    d_target = acts_b.shape[1]

    # Train/test split
    rng = np.random.default_rng(seed)
    n_train = int(n * train_fraction)
    perm = rng.permutation(n)
    train_idx, test_idx = perm[:n_train], perm[n_train:]

    X_train, X_test = acts_a[train_idx], acts_a[test_idx]
    Y_train, Y_test = acts_b[train_idx], acts_b[test_idx]

    console.print(f"\n[bold]Alignment: d_source={d_source}, d_target={d_target}[/bold]")
    console.print(f"  Train: {n_train}, Test: {n - n_train}")

    results = {}

    if method in ("procrustes", "both") and d_source == d_target:
        results["procrustes"] = orthogonal_procrustes_alignment(
            X_train, Y_train, X_test, Y_test
        )
    elif method == "procrustes" and d_source != d_target:
        console.print(
            f"[yellow]Skipping Procrustes: d_model mismatch "
            f"({d_source} vs {d_target})[/yellow]"
        )

    if method in ("linear", "both"):
        results["linear"] = linear_projection_alignment(
            X_train, Y_train, X_test, Y_test,
            regularization=regularization,
        )

    return results


def apply_mapping(
    activations: np.ndarray,
    alignment: AlignmentResult,
) -> np.ndarray:
    """Apply a learned alignment mapping to transform activations.

    Args:
        activations: Source activations (n, d_source)
        alignment: A learned AlignmentResult

    Returns:
        Mapped activations (n, d_target)
    """
    X_mean = alignment._X_mean  # type: ignore
    Y_mean = alignment._Y_mean  # type: ignore
    return (activations - X_mean) @ alignment.W + Y_mean


def save_alignment(alignment: AlignmentResult, path: Path):
    """Save alignment result to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        method=alignment.method,
        W=alignment.W,
        X_mean=alignment._X_mean,  # type: ignore
        Y_mean=alignment._Y_mean,  # type: ignore
        train_loss=alignment.train_loss,
        test_loss=alignment.test_loss,
        d_source=alignment.d_source,
        d_target=alignment.d_target,
        explained_variance=alignment.explained_variance,
    )
    console.print(f"  Saved alignment to {path}")


def load_alignment(path: Path) -> AlignmentResult:
    """Load alignment result from disk."""
    data = np.load(path)
    result = AlignmentResult(
        method=str(data["method"]),
        W=data["W"],
        train_loss=float(data["train_loss"]),
        test_loss=float(data["test_loss"]),
        d_source=int(data["d_source"]),
        d_target=int(data["d_target"]),
        explained_variance=float(data["explained_variance"]),
    )
    result._X_mean = data["X_mean"]  # type: ignore
    result._Y_mean = data["Y_mean"]  # type: ignore
    return result
