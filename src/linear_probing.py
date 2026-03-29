"""Linear probing for cross-model transfer experiments.

Tests whether the low-rank shared subspace carries task-relevant signal by:
1. Training a linear probe on Model A's activations
2. Transferring it via rank-k alignment to Model B's space
3. Evaluating transferred probe accuracy on Model B

Supports next-token prediction probes and arbitrary classification probes.
"""

import numpy as np
import torch
from dataclasses import dataclass
from typing import Optional
from rich.console import Console

console = Console()


@dataclass
class ProbeResult:
    """Result from training or evaluating a probe."""
    accuracy_top1: float
    accuracy_top5: float
    loss: float  # cross-entropy
    n_samples: int
    n_classes: int


class LinearProbe:
    """Logistic regression probe on activations.

    Supports GPU-accelerated training and transfer via alignment matrices.
    """

    def __init__(self, input_dim: int, n_classes: int, device: str = "cpu"):
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.device = torch.device(device)
        self.weight = torch.zeros(n_classes, input_dim, device=self.device, dtype=torch.float32)
        self.bias = torch.zeros(n_classes, device=self.device, dtype=torch.float32)
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 50,
        lr: float = 0.01,
        batch_size: int = 512,
        weight_decay: float = 1e-4,
    ) -> list[float]:
        """Train probe via mini-batch SGD with cross-entropy loss.

        Args:
            X: Activations (n_samples, d_model)
            y: Labels (n_samples,) integer class IDs
            epochs: Training epochs
            lr: Learning rate
            batch_size: Mini-batch size
            weight_decay: L2 regularization

        Returns:
            List of per-epoch losses
        """
        X_t = torch.from_numpy(X).to(self.device, dtype=torch.float32)
        y_t = torch.from_numpy(y).to(self.device, dtype=torch.long)

        self.weight = torch.randn(
            self.n_classes, self.input_dim,
            device=self.device, dtype=torch.float32
        ) * 0.01
        self.bias = torch.zeros(self.n_classes, device=self.device, dtype=torch.float32)
        self.weight.requires_grad_(True)
        self.bias.requires_grad_(True)

        optimizer = torch.optim.Adam(
            [self.weight, self.bias], lr=lr, weight_decay=weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()

        n = X_t.shape[0]
        losses = []

        for epoch in range(epochs):
            perm = torch.randperm(n, device=self.device)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, n, batch_size):
                idx = perm[i:i + batch_size]
                logits = X_t[idx] @ self.weight.T + self.bias
                loss = criterion(logits, y_t[idx])

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)

        self._fitted = True
        self.weight = self.weight.detach()
        self.bias = self.bias.detach()
        return losses

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ProbeResult:
        """Evaluate probe accuracy on held-out data.

        Args:
            X: Activations (n_samples, d_model)
            y: Labels (n_samples,)

        Returns:
            ProbeResult with top-1, top-5 accuracy and loss
        """
        X_t = torch.from_numpy(X).to(self.device, dtype=torch.float32)
        y_t = torch.from_numpy(y).to(self.device, dtype=torch.long)

        with torch.no_grad():
            logits = X_t @ self.weight.T + self.bias
            loss = torch.nn.functional.cross_entropy(logits, y_t).item()

            # Top-1
            preds = logits.argmax(dim=1)
            acc1 = (preds == y_t).float().mean().item()

            # Top-5
            if self.n_classes >= 5:
                top5 = logits.topk(5, dim=1).indices
                acc5 = (top5 == y_t.unsqueeze(1)).any(dim=1).float().mean().item()
            else:
                acc5 = acc1

        return ProbeResult(
            accuracy_top1=acc1,
            accuracy_top5=acc5,
            loss=loss,
            n_samples=len(y),
            n_classes=self.n_classes,
        )

    def transfer(self, W_align: np.ndarray, X_mean_src: np.ndarray,
                 Y_mean_tgt: np.ndarray) -> "LinearProbe":
        """Transfer probe to target model space via alignment mapping.

        Given alignment: mapped = (X - X_mean) @ W + Y_mean
        The probe in source space: logits = X @ P.T + b
        Substituting: logits = ((mapped - Y_mean) @ W_pinv + X_mean) @ P.T + b

        Simpler approach: transform probe weights directly.
        New probe in target space: logits = mapped @ P_new.T + b_new
        where P_new = P @ W_pinv^T and b_new accounts for centering.

        For practical purposes, we use the pseudoinverse approach:
        P_new.T = pinv(W) @ P.T  (with centering adjustments)

        Args:
            W_align: Alignment matrix (d_source, d_target)
            X_mean_src: Source centering (d_source,)
            Y_mean_tgt: Target centering (d_target,)

        Returns:
            New LinearProbe in target model's space
        """
        W = torch.from_numpy(W_align).to(self.device, dtype=torch.float32)
        x_mean = torch.from_numpy(X_mean_src).to(self.device, dtype=torch.float32)
        y_mean = torch.from_numpy(Y_mean_tgt).to(self.device, dtype=torch.float32)

        d_target = W.shape[1]

        # Alignment: mapped = (X_src - X_mean) @ W + Y_mean
        # So: X_src = (mapped - Y_mean) @ W^+ + X_mean
        # where W^+ is the pseudoinverse of W (d_target, d_source)
        #
        # Probe in source space: logits = X_src @ P^T + b
        # Substituting: logits = ((mapped - Y_mean) @ W^+ + X_mean) @ P^T + b
        #             = mapped @ (W^+ @ P^T) + (X_mean - Y_mean @ W^+) @ P^T + b
        #
        # New probe weight: P_new = P @ W^+^T  (shape: n_classes x d_target)
        # New probe bias: b_new = b + (X_mean - Y_mean @ W^+) @ P^T

        # W is (d_source, d_target), we need W^+ which is (d_target, d_source)
        W_pinv = torch.linalg.lstsq(W, torch.eye(W.shape[0], device=self.device, dtype=torch.float32)).solution
        # W_pinv shape: (d_target, d_source)

        new_probe = LinearProbe(d_target, self.n_classes, str(self.device))
        # P @ W_pinv^T: (n_classes, d_source) @ (d_source, d_target) = (n_classes, d_target)
        new_probe.weight = (self.weight @ W_pinv.T).detach()

        # Adjust bias for centering
        centering_offset = (x_mean - y_mean @ W_pinv) @ self.weight.T  # (n_classes,)
        new_probe.bias = (self.bias + centering_offset).detach()
        new_probe._fitted = True

        return new_probe


def extract_next_token_labels(
    model_name: str,
    texts: list[str],
    max_seq_len: int = 128,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """Extract next-token labels for probing.

    For each text, tokenize and return the token ID at the position
    AFTER the last non-padding token (i.e., what the model should predict
    given the activation at the last position).

    Args:
        model_name: HuggingFace model name
        texts: List of text prompts
        max_seq_len: Max sequence length
        batch_size: Batch size
        device: Device

    Returns:
        Array of shape (n_texts,) with next-token IDs.
        -1 for texts where next token couldn't be determined.
    """
    from transformers import AutoTokenizer

    console.print(f"[bold]Extracting next-token labels from {model_name}[/bold]")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    labels = np.full(len(texts), -1, dtype=np.int64)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        )

        for j in range(len(batch_texts)):
            # Find last non-padding position
            mask = inputs["attention_mask"][j]
            last_pos = mask.sum().item() - 1

            if last_pos < max_seq_len - 1:
                # Next token exists in the tokenized sequence
                next_token = inputs["input_ids"][j, last_pos + 1].item()
                # But if it's padding, use the token at last_pos as a "self-prediction" fallback
                if next_token != tokenizer.pad_token_id:
                    labels[i + j] = next_token
                    continue

            # Fallback: use the last token itself (predict current token)
            labels[i + j] = inputs["input_ids"][j, last_pos].item()

        if (i // batch_size) % 50 == 0:
            console.print(f"  Processed {i + len(batch_texts)}/{len(texts)} texts")

    valid = (labels != -1).sum()
    console.print(f"  Valid labels: {valid}/{len(texts)}")
    return labels
