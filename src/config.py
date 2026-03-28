"""Configuration loading and validation."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class ModelConfig:
    name: str
    alias: str
    d_model: Optional[int] = None  # Auto-detected from model


@dataclass
class ExtractionConfig:
    dataset: str = "NeelNanda/pile-10k"
    n_prompts: int = 10_000
    max_seq_len: int = 128
    batch_size: int = 32
    token_position: str = "last"
    layer_fractions: list[float] = field(
        default_factory=lambda: [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0]
    )
    dtype: str = "float32"


@dataclass
class CKAConfig:
    kernel: str = "linear"
    rbf_sigma: Optional[float] = None
    debiased: bool = True
    subsample_n: Optional[int] = 5000


@dataclass
class AlignmentConfig:
    method: str = "both"  # "procrustes" | "linear" | "both"
    train_fraction: float = 0.8
    orthogonal: bool = True
    regularization: float = 1e-4
    epochs: int = 100
    lr: float = 1e-3


@dataclass
class OracleConfig:
    adapter_id: Optional[str] = None
    base_model: str = "model_b"
    test_prompts: list[str] = field(default_factory=list)


@dataclass
class Config:
    model_a: ModelConfig
    model_b: ModelConfig
    extraction: ExtractionConfig
    cka: CKAConfig
    alignment: AlignmentConfig
    oracle: OracleConfig
    output_dir: Path = Path("outputs")
    device: str = "cuda"
    seed: int = 42

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with open(path) as f:
            raw = yaml.safe_load(f)

        model_a = ModelConfig(**raw["model_a"])
        model_b = ModelConfig(**raw["model_b"])
        extraction = ExtractionConfig(**raw.get("extraction", {}))
        cka = CKAConfig(**raw.get("cka", {}))

        align_raw = raw.get("alignment", {})
        alignment = AlignmentConfig(
            method=align_raw.get("method", "both"),
            train_fraction=align_raw.get("train_fraction", 0.8),
            orthogonal=align_raw.get("procrustes", {}).get("orthogonal", True),
            regularization=align_raw.get("linear", {}).get("regularization", 1e-4),
            epochs=align_raw.get("linear", {}).get("epochs", 100),
            lr=align_raw.get("linear", {}).get("lr", 1e-3),
        )
        oracle = OracleConfig(**raw.get("oracle", {}))

        output_dir = Path(raw.get("output", {}).get("dir", "outputs"))
        device = raw.get("compute", {}).get("device", "cuda")
        seed = raw.get("compute", {}).get("seed", 42)

        return cls(
            model_a=model_a,
            model_b=model_b,
            extraction=extraction,
            cka=cka,
            alignment=alignment,
            oracle=oracle,
            output_dir=output_dir,
            device=device,
            seed=seed,
        )
