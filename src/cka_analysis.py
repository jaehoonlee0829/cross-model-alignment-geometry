"""Centered Kernel Alignment (CKA) for comparing representations across models.

Implements:
- Linear CKA (Kornblith et al., 2019)
- RBF CKA
- Debiased CKA (for more reliable comparison with finite samples)
- Full layer-pair CKA heatmap computation

References:
- Kornblith et al., "Similarity of Neural Network Representations Revisited" (ICML 2019)
- Nguyen et al., "Do Wide Neural Networks Really Need to be Wide?" (2021)
"""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from rich.console import Console
from rich.table import Table

console = Console()


def centering_matrix(n: int) -> np.ndarray:
    """Compute the centering matrix H = I - (1/n) * 11^T."""
    return np.eye(n) - np.ones((n, n)) / n


def linear_kernel(X: np.ndarray) -> np.ndarray:
    """Compute linear kernel K = X @ X^T."""
    return X @ X.T


def rbf_kernel(X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
    """Compute RBF (Gaussian) kernel.

    If sigma is None, use the median heuristic.
    """
    sq_dists = np.sum(X ** 2, axis=1, keepdims=True) - 2 * X @ X.T + np.sum(X ** 2, axis=1)
    sq_dists = np.maximum(sq_dists, 0)

    if sigma is None:
        # Median heuristic
        median_dist = np.median(sq_dists[np.triu_indices_from(sq_dists, k=1)])
        sigma = np.sqrt(median_dist / 2)
        if sigma == 0:
            sigma = 1.0

    return np.exp(-sq_dists / (2 * sigma ** 2))


def hsic(K: np.ndarray, L: np.ndarray, debiased: bool = True) -> float:
    """Compute the Hilbert-Schmidt Independence Criterion.

    Args:
        K: Kernel matrix for representation X, shape (n, n)
        L: Kernel matrix for representation Y, shape (n, n)
        debiased: Use debiased estimator (Song et al., 2012)

    Returns:
        HSIC value
    """
    n = K.shape[0]
    assert K.shape == L.shape == (n, n)

    if debiased:
        # Debiased HSIC estimator
        # Zero out diagonals
        K_tilde = K.copy()
        L_tilde = L.copy()
        np.fill_diagonal(K_tilde, 0)
        np.fill_diagonal(L_tilde, 0)

        # Compute terms (using sum(A*B) instead of trace(A@B) for O(n^2) not O(n^3))
        term1 = np.sum(K_tilde * L_tilde)
        term2 = (K_tilde.sum() * L_tilde.sum()) / ((n - 1) * (n - 2))
        term3 = 2 * (K_tilde.sum(axis=0) @ L_tilde.sum(axis=0)) / (n - 2)

        return (term1 + term2 - term3) / (n * (n - 3))
    else:
        # Biased HSIC
        H = centering_matrix(n)
        return np.trace(K @ H @ L @ H) / ((n - 1) ** 2)


def compute_cka(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",
    debiased: bool = True,
    rbf_sigma: Optional[float] = None,
) -> float:
    """Compute CKA between two representation matrices.

    Args:
        X: Activations from model A, shape (n_samples, d_model_a)
        Y: Activations from model B, shape (n_samples, d_model_b)
        kernel: "linear" or "rbf"
        debiased: Use debiased HSIC estimator
        rbf_sigma: Sigma for RBF kernel (None = median heuristic)

    Returns:
        CKA similarity score in [0, 1] (higher = more similar)
    """
    assert X.shape[0] == Y.shape[0], f"Sample count mismatch: {X.shape[0]} vs {Y.shape[0]}"

    if kernel == "linear":
        K = linear_kernel(X)
        L = linear_kernel(Y)
    elif kernel == "rbf":
        K = rbf_kernel(X, sigma=rbf_sigma)
        L = rbf_kernel(Y, sigma=rbf_sigma)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    hsic_kl = hsic(K, L, debiased=debiased)
    hsic_kk = hsic(K, K, debiased=debiased)
    hsic_ll = hsic(L, L, debiased=debiased)

    denom = np.sqrt(hsic_kk * hsic_ll)
    if denom < 1e-10:
        return 0.0

    return float(hsic_kl / denom)


def permutation_test_cka(
    X: np.ndarray,
    Y: np.ndarray,
    kernel: str = "linear",
    debiased: bool = True,
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Aristotelian-style permutation calibration for CKA.

    Shuffles sample indices of X to break correspondence with Y,
    computes CKA on each permutation to build null distribution,
    then reports z-score against the null.

    This controls for the dimensionality inflation confound identified
    by Chun et al. (2026) "Revisiting the Platonic Representation
    Hypothesis: An Aristotelian View" (arxiv.org/abs/2602.14486).

    Args:
        X: Activations from model A, shape (n_samples, d_model_a)
        Y: Activations from model B, shape (n_samples, d_model_b)
        kernel: "linear" or "rbf"
        debiased: Use debiased HSIC estimator
        n_permutations: Number of permutations for null distribution
        seed: Random seed

    Returns:
        Dict with keys:
        - observed_cka: float
        - null_mean: float
        - null_std: float
        - calibrated_cka: float  (= (observed - null_mean) / null_std, i.e. effect size)
        - p_value: float  (fraction of null >= observed)
        - null_95th: float
        - n_permutations: int
    """
    rng = np.random.default_rng(seed)

    observed = compute_cka(X, Y, kernel=kernel, debiased=debiased)

    null_ckas = []
    for i in range(n_permutations):
        perm = rng.permutation(X.shape[0])
        null_cka = compute_cka(X[perm], Y, kernel=kernel, debiased=debiased)
        null_ckas.append(null_cka)

    null_ckas = np.array(null_ckas)
    null_mean = float(null_ckas.mean())
    null_std = float(null_ckas.std())

    calibrated = (observed - null_mean) / null_std if null_std > 0 else float('inf')
    p_value = float(np.mean(null_ckas >= observed))

    return {
        "observed_cka": float(observed),
        "null_mean": null_mean,
        "null_std": null_std,
        "calibrated_cka": calibrated,
        "p_value": p_value,
        "null_95th": float(np.percentile(null_ckas, 95)),
        "n_permutations": n_permutations,
    }


def compute_cka_matrix(
    acts_a: dict[int, torch.Tensor],
    acts_b: dict[int, torch.Tensor],
    kernel: str = "linear",
    debiased: bool = True,
    subsample_n: Optional[int] = 5000,
    rbf_sigma: Optional[float] = None,
) -> tuple[np.ndarray, list[int], list[int]]:
    """Compute CKA between all layer pairs of two models.

    Args:
        acts_a: Dict mapping layer_index -> (n_samples, d_model_a) tensor
        acts_b: Dict mapping layer_index -> (n_samples, d_model_b) tensor
        kernel: "linear" or "rbf"
        debiased: Use debiased HSIC
        subsample_n: Subsample this many examples for speed (None = use all)

    Returns:
        (cka_matrix, layers_a, layers_b) where cka_matrix[i, j] = CKA(layer_a_i, layer_b_j)
    """
    layers_a = sorted(acts_a.keys())
    layers_b = sorted(acts_b.keys())

    # Determine sample count
    n_samples = next(iter(acts_a.values())).shape[0]
    if subsample_n is not None and subsample_n < n_samples:
        rng = np.random.default_rng(42)
        indices = rng.choice(n_samples, size=subsample_n, replace=False)
    else:
        indices = np.arange(n_samples)

    cka_matrix = np.zeros((len(layers_a), len(layers_b)))

    console.print(
        f"[bold]Computing CKA matrix: {len(layers_a)} x {len(layers_b)} layer pairs "
        f"({len(indices)} samples)[/bold]"
    )

    for i, la in enumerate(layers_a):
        X = acts_a[la][indices].numpy()
        for j, lb in enumerate(layers_b):
            Y = acts_b[lb][indices].numpy()
            cka_val = compute_cka(X, Y, kernel=kernel, debiased=debiased, rbf_sigma=rbf_sigma)
            cka_matrix[i, j] = cka_val

        console.print(
            f"  Layer {la} (model A): max CKA = {cka_matrix[i].max():.4f} "
            f"(best match: layer {layers_b[cka_matrix[i].argmax()]} of model B)"
        )

    return cka_matrix, layers_a, layers_b


def find_best_layer_pairs(
    cka_matrix: np.ndarray,
    layers_a: list[int],
    layers_b: list[int],
    threshold: float = 0.3,
) -> list[tuple[int, int, float]]:
    """Find best-matching layer pairs above a CKA threshold.

    Returns list of (layer_a, layer_b, cka_score) sorted by CKA descending.
    """
    pairs = []
    for i, la in enumerate(layers_a):
        j_best = int(cka_matrix[i].argmax())
        score = float(cka_matrix[i, j_best])
        if score >= threshold:
            pairs.append((la, layers_b[j_best], score))

    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs


def plot_cka_heatmap(
    cka_matrix: np.ndarray,
    layers_a: list[int],
    layers_b: list[int],
    alias_a: str = "Model A",
    alias_b: str = "Model B",
    save_path: Optional[Path] = None,
    title: Optional[str] = None,
):
    """Plot CKA similarity heatmap between two models' layers."""
    fig, ax = plt.subplots(figsize=(10, 8))

    sns.heatmap(
        cka_matrix,
        xticklabels=[str(l) for l in layers_b],
        yticklabels=[str(l) for l in layers_a],
        annot=True,
        fmt=".3f",
        cmap="viridis",
        vmin=0,
        vmax=1,
        square=True,
        ax=ax,
        cbar_kws={"label": "CKA Similarity"},
    )

    if title is None:
        title = f"CKA Similarity: {alias_a} vs {alias_b}"
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xlabel(f"{alias_b} Layer Index", fontsize=12)
    ax.set_ylabel(f"{alias_a} Layer Index", fontsize=12)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        console.print(f"  Saved heatmap to {save_path}")

    return fig


def print_cka_summary(
    cka_matrix: np.ndarray,
    layers_a: list[int],
    layers_b: list[int],
    alias_a: str = "Model A",
    alias_b: str = "Model B",
):
    """Print a rich summary table of CKA results."""
    table = Table(title=f"CKA Summary: {alias_a} vs {alias_b}")
    table.add_column(f"{alias_a} Layer", style="cyan")
    table.add_column(f"Best Match ({alias_b})", style="green")
    table.add_column("CKA Score", style="yellow")
    table.add_column("Verdict", style="bold")

    for i, la in enumerate(layers_a):
        j_best = int(cka_matrix[i].argmax())
        score = cka_matrix[i, j_best]
        if score >= 0.5:
            verdict = "[green]Strong alignment[/green]"
        elif score >= 0.3:
            verdict = "[yellow]Moderate alignment[/yellow]"
        else:
            verdict = "[red]Weak alignment[/red]"

        table.add_row(str(la), str(layers_b[j_best]), f"{score:.4f}", verdict)

    console.print(table)
    console.print(f"\n  Overall mean CKA: {cka_matrix.mean():.4f}")
    console.print(f"  Max CKA: {cka_matrix.max():.4f}")
    console.print(f"  Diagonal mean CKA: {np.diag(cka_matrix).mean():.4f}" if cka_matrix.shape[0] == cka_matrix.shape[1] else "")
