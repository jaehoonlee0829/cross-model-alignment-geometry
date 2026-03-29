"""Learn alignment mappings between activation spaces of different models.

Three approaches:
1. Orthogonal Procrustes (SVD-based): Requires d_model_a == d_model_b.
   Finds the orthogonal matrix W that minimizes ||AW - B||_F.
   Preserves geometric structure (distances, angles).

2. Learned Linear Projection (Ridge): Works for ANY d_model pair.
   Learns W: R^{d_a} -> R^{d_b} via ridge regression.
   More flexible but doesn't preserve geometry.

3. Low-Rank Alignment (LoRA-style): Works for ANY d_model pair.
   Learns W = A @ B where A: R^{d_a} -> R^{rank} and B: R^{rank} -> R^{d_b}.
   Parameter efficient and interpretable.

The Platonic Representation Hypothesis suggests representations converge
up to an orthogonal transformation — so if it holds, Procrustes should
work well for matching d_model. Low-rank and ridge methods are fallbacks
for mismatched dimensions.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import minimize
from rich.console import Console

console = Console()


@dataclass
class AlignmentResult:
    """Result of learning an alignment mapping."""
    method: str  # "procrustes", "linear", "low_rank", or "lasso"
    W: np.ndarray  # The mapping matrix (or A for low_rank)
    train_loss: float  # ||AW - B||_F on training data (normalized)
    test_loss: float  # Same on held-out data
    d_source: int
    d_target: int
    explained_variance: float  # How much of target variance is captured
    # For Procrustes only
    orthogonality_error: Optional[float] = None  # ||W^T W - I||_F
    # For low_rank only
    W_B: Optional[np.ndarray] = None  # B matrix for low_rank (W = A @ B)
    rank: Optional[int] = None  # Rank parameter for low_rank
    regularization: Optional[float] = None  # Ridge or LASSO regularization


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
        f"[bold]Learning linear projection (ridge): "
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
        regularization=regularization,
    )

    result._X_mean = X_mean  # type: ignore
    result._Y_mean = Y_mean  # type: ignore

    return result


def low_rank_alignment(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    rank: int = 64,
    regularization: float = 1e-4,
) -> AlignmentResult:
    """Learn a low-rank alignment W = A @ B via ridge regression.

    LoRA-style decomposition: Instead of learning full W (d_source × d_target),
    learn A (d_source × rank) and B (rank × d_target), then W = A @ B.
    Parameter efficient and interpretable.

    Solves:
        minimize ||X @ A @ B - Y||_F^2 + λ(||A||_F^2 + ||B||_F^2)

    Uses alternating least squares or direct ridge regression.

    Args:
        X_train: Source activations (n_train, d_source)
        Y_train: Target activations (n_train, d_target)
        X_test: Source activations (n_test, d_source)
        Y_test: Target activations (n_test, d_target)
        rank: Low-rank dimension
        regularization: Ridge regularization strength

    Returns:
        AlignmentResult with A and B matrices.
    """
    d_source = X_train.shape[1]
    d_target = Y_train.shape[1]

    console.print(
        f"[bold]Learning low-rank alignment (LoRA): "
        f"R^{d_source} -> R^{d_target} via rank-{rank} (λ={regularization})[/bold]"
    )

    # Center data
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    X_centered = X_train - X_mean
    Y_centered = Y_train - Y_mean

    # Initialize A and B randomly
    rng = np.random.default_rng(42)
    A = rng.normal(0, 0.01, (d_source, rank))
    B = rng.normal(0, 0.01, (rank, d_target))

    # Alternating least squares: optimize A, then B, repeat
    for iteration in range(5):  # Few iterations for reasonable time
        # Solve for B: minimize ||X @ A @ B - Y||_F^2 + λ||B||_F^2
        # Reshape: (X @ A) is (n_train, rank), solve (X @ A) @ B = Y
        XA = X_centered @ A
        B_rhs = XA.T @ Y_centered
        B_lhs = XA.T @ XA + regularization * np.eye(rank)
        B = np.linalg.solve(B_lhs, B_rhs)

        # Solve for A: minimize ||X @ A @ B - Y||_F^2 + λ||A||_F^2
        # Rewrite as: minimize ||X @ A - Y @ B^+ ||_F^2 where B^+ = B^T(BB^T)^{-1}
        # This decouples to a standard ridge problem for A.
        # B is (rank, d_target), B^T is (d_target, rank), BB^T is (rank, rank)
        B_pinv = B.T @ np.linalg.solve(B @ B.T + regularization * np.eye(rank), np.eye(rank))
        # B_pinv: (d_target, rank) — pseudo-inverse of B
        target_A = Y_centered @ B_pinv  # (n_train, rank)
        A = np.linalg.solve(
            X_centered.T @ X_centered + regularization * np.eye(d_source),
            X_centered.T @ target_A,
        )

    # Compute full W = A @ B
    W = A @ B

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
    console.print(f"  A shape: {A.shape}, B shape: {B.shape}, W rank: {np.linalg.matrix_rank(W)}")
    console.print(f"  Parameter reduction: {(d_source*rank + rank*d_target) / (d_source*d_target) * 100:.1f}%")

    result = AlignmentResult(
        method="low_rank",
        W=A,
        W_B=B,
        train_loss=float(train_loss),
        test_loss=float(test_loss),
        d_source=d_source,
        d_target=d_target,
        explained_variance=float(explained_var),
        rank=rank,
        regularization=regularization,
    )

    result._X_mean = X_mean  # type: ignore
    result._Y_mean = Y_mean  # type: ignore

    return result


def lasso_alignment(
    X_train: np.ndarray,
    Y_train: np.ndarray,
    X_test: np.ndarray,
    Y_test: np.ndarray,
    regularization: float = 1e-3,
) -> AlignmentResult:
    """Learn sparse alignment via L1 regularization (LASSO).

    Solves:
        W = argmin ||XW - Y||_F^2 + λ||W||_1

    Uses iterative soft-thresholding or coordinate descent.
    Useful for diagnostic purposes: sparse W reveals which source
    dimensions are most important for alignment.

    Args:
        X_train: Source activations (n_train, d_source)
        Y_train: Target activations (n_train, d_target)
        X_test: Source activations (n_test, d_source)
        Y_test: Target activations (n_test, d_target)
        regularization: LASSO (L1) regularization strength

    Returns:
        AlignmentResult with sparse W.
    """
    d_source = X_train.shape[1]
    d_target = Y_train.shape[1]

    console.print(
        f"[bold]Learning sparse alignment (LASSO): "
        f"R^{d_source} -> R^{d_target} (λ={regularization})[/bold]"
    )

    # Center data
    X_mean = X_train.mean(axis=0)
    Y_mean = Y_train.mean(axis=0)
    X_centered = X_train - X_mean
    Y_centered = Y_train - Y_mean

    # Start with ridge solution as warm start
    ridge_lam = regularization / 100.0
    XtX = X_centered.T @ X_centered
    XtY = X_centered.T @ Y_centered
    W = np.linalg.solve(
        XtX + ridge_lam * np.eye(d_source),
        XtY,
    )

    # Iterative soft-thresholding for L1 regularization
    step_size = 1.0 / (2.0 * np.linalg.norm(X_centered, 2) ** 2 + 1e-6)
    for iteration in range(20):
        # Gradient: X^T (X @ W - Y)
        grad = X_centered.T @ (X_centered @ W - Y_centered)
        W_new = W - step_size * grad
        # Soft thresholding: sign(x) * max(|x| - λ, 0)
        threshold = regularization * step_size
        W_new = np.sign(W_new) * np.maximum(np.abs(W_new) - threshold, 0)
        W = W_new

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

    sparsity = np.mean(W == 0)
    console.print(f"  Train loss (normalized): {train_loss:.4f}")
    console.print(f"  Test loss (normalized):  {test_loss:.4f}")
    console.print(f"  Explained variance:      {explained_var:.4f}")
    console.print(f"  W shape: {W.shape}, rank: {np.linalg.matrix_rank(W)}, sparsity: {sparsity:.2%}")

    result = AlignmentResult(
        method="lasso",
        W=W,
        train_loss=float(train_loss),
        test_loss=float(test_loss),
        d_source=d_source,
        d_target=d_target,
        explained_variance=float(explained_var),
        regularization=regularization,
    )

    result._X_mean = X_mean  # type: ignore
    result._Y_mean = Y_mean  # type: ignore

    return result


def learn_alignment(
    acts_a: np.ndarray,
    acts_b: np.ndarray,
    method: str = "orthogonal_procrustes",
    train_fraction: float = 0.8,
    regularization: float = 1e-4,
    rank: int = 64,
    seed: int = 42,
) -> dict[str, AlignmentResult]:
    """Learn alignment mapping(s) between two activation sets.

    Args:
        acts_a: Source activations (n_samples, d_source)
        acts_b: Target activations (n_samples, d_target)
        method: "orthogonal_procrustes", "linear", "low_rank", "lasso", or "all"
        train_fraction: Fraction of data for training
        regularization: Ridge/LASSO regularization strength
        rank: Rank for low_rank method
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

    # Handle various method specifications
    methods_to_run = []
    if method == "all":
        methods_to_run = ["orthogonal_procrustes", "linear", "low_rank", "lasso"]
    elif method == "both":  # Legacy support
        methods_to_run = ["orthogonal_procrustes", "linear"]
    else:
        methods_to_run = [method]

    if "orthogonal_procrustes" in methods_to_run:
        if d_source == d_target:
            results["orthogonal_procrustes"] = orthogonal_procrustes_alignment(
                X_train, Y_train, X_test, Y_test
            )
        else:
            console.print(
                f"[yellow]Skipping orthogonal_procrustes: d_model mismatch "
                f"({d_source} vs {d_target})[/yellow]"
            )

    if "linear" in methods_to_run:
        results["linear"] = linear_projection_alignment(
            X_train, Y_train, X_test, Y_test,
            regularization=regularization,
        )

    if "low_rank" in methods_to_run:
        results["low_rank"] = low_rank_alignment(
            X_train, Y_train, X_test, Y_test,
            rank=rank,
            regularization=regularization,
        )

    if "lasso" in methods_to_run:
        results["lasso"] = lasso_alignment(
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

    if alignment.method == "low_rank":
        # For low_rank: W = A @ B, stored as W=A and W_B=B
        W = alignment.W @ alignment.W_B
    else:
        W = alignment.W

    return (activations - X_mean) @ W + Y_mean


def save_alignment(alignment: AlignmentResult, path: Path):
    """Save alignment result to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_dict = {
        "method": alignment.method,
        "W": alignment.W,
        "X_mean": alignment._X_mean,  # type: ignore
        "Y_mean": alignment._Y_mean,  # type: ignore
        "train_loss": alignment.train_loss,
        "test_loss": alignment.test_loss,
        "d_source": alignment.d_source,
        "d_target": alignment.d_target,
        "explained_variance": alignment.explained_variance,
    }
    if alignment.W_B is not None:
        save_dict["W_B"] = alignment.W_B
    if alignment.rank is not None:
        save_dict["rank"] = alignment.rank
    if alignment.regularization is not None:
        save_dict["regularization"] = alignment.regularization

    np.savez(path, **save_dict)
    console.print(f"  Saved alignment to {path}")


def load_alignment(path: Path) -> AlignmentResult:
    """Load alignment result from disk."""
    data = np.load(path)
    W_B = data.get("W_B", None) if "W_B" in data.files else None
    rank = int(data["rank"]) if "rank" in data.files else None
    regularization = float(data["regularization"]) if "regularization" in data.files else None

    result = AlignmentResult(
        method=str(data["method"]),
        W=data["W"],
        W_B=W_B,
        train_loss=float(data["train_loss"]),
        test_loss=float(data["test_loss"]),
        d_source=int(data["d_source"]),
        d_target=int(data["d_target"]),
        explained_variance=float(data["explained_variance"]),
        rank=rank,
        regularization=regularization,
    )
    result._X_mean = data["X_mean"]  # type: ignore
    result._Y_mean = data["Y_mean"]  # type: ignore
    return result
