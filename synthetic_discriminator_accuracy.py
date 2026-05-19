from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from mensa_ad.data import TabularAnomalyDataset, load_labeled_frame, split_normal_train_val
from mensa_ad.metrics import compute_binary_metrics, confusion_counts
from mensa_ad.models import DenseBlock, DiscriminatorEncoder, GeneratorDecoder
from mensa_ad.training import noise_uniform, resolve_device, set_seed


@dataclass(frozen=True)
class SyntheticConfig:
    checkpoint: str = "outputs/mensa_ad/model.pt"
    train: str = "data/train.csv"
    train_labels: str = "data/train_label.csv"
    sample_count: int = 512
    batch_size: int = 64
    val_fraction: float = 0.1
    seed: int = 42
    device: str = "auto"
    feature_dim: int | None = None
    noise_dim: int | None = None
    latent_dim: int | None = None
    dropout: float | None = None
    out: str | None = None
    samples_out: str | None = None


@dataclass(frozen=True)
class LoadedCheckpointModels:
    generator: nn.Module
    discriminator: nn.Module
    feature_dim: int
    noise_dim: int
    latent_dim: int
    dropout: float
    architecture: str
    feature_columns: list[str]


class _LegacyGeneratorDecoder(nn.Module):
    def __init__(self, noise_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            DenseBlock(noise_dim, 128, activation="leaky_relu", dropout=dropout, batch_norm=False),
            DenseBlock(128, 256, activation="leaky_relu", dropout=dropout, batch_norm=True),
            DenseBlock(256, 512, activation="leaky_relu", dropout=dropout, batch_norm=True),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class _LegacyDiscriminatorEncoder(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int, dropout: float) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            DenseBlock(input_dim, 600, activation="leaky_relu", dropout=dropout, batch_norm=False),
            DenseBlock(600, 256, activation="leaky_relu", dropout=dropout, batch_norm=True),
            DenseBlock(256, latent_dim, activation="relu", dropout=dropout, batch_norm=True),
        )
        self.validity = nn.Sequential(nn.Linear(latent_dim, 1), nn.Sigmoid())

    def forward(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(samples)
        return self.validity(latent), latent


def _config_value(
    checkpoint_config: dict[str, Any],
    key: str,
    override: int | float | None,
    default: int | float,
) -> int | float:
    if override is not None:
        return override
    return checkpoint_config.get(key, default)


def _looks_like_legacy_checkpoint(generator_state: dict[str, torch.Tensor]) -> bool:
    return "network.3.weight" in generator_state and "network.4.weight" not in generator_state


def load_checkpoint_models(
    checkpoint_path: str | Path,
    config: SyntheticConfig,
    device: torch.device,
) -> LoadedCheckpointModels:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_config = checkpoint.get("config", {})
    feature_columns = list(checkpoint.get("feature_columns", []))
    feature_dim = int(
        _config_value(
            checkpoint_config,
            "input_dim",
            config.feature_dim,
            len(feature_columns) if feature_columns else 15,
        )
    )
    noise_dim = int(_config_value(checkpoint_config, "noise_dim", config.noise_dim, 10))
    latent_dim = int(_config_value(checkpoint_config, "latent_dim", config.latent_dim, 128))
    dropout = float(_config_value(checkpoint_config, "dropout", config.dropout, 0.2))

    generator: nn.Module = GeneratorDecoder(
        noise_dim=noise_dim,
        output_dim=feature_dim,
        dropout=dropout,
    ).to(device)
    discriminator: nn.Module = DiscriminatorEncoder(
        input_dim=feature_dim,
        latent_dim=latent_dim,
        dropout=dropout,
    ).to(device)
    architecture = "current"
    try:
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
    except RuntimeError:
        if not _looks_like_legacy_checkpoint(checkpoint["generator"]):
            raise
        generator = _LegacyGeneratorDecoder(
            noise_dim=noise_dim,
            output_dim=feature_dim,
            dropout=dropout,
        ).to(device)
        discriminator = _LegacyDiscriminatorEncoder(
            input_dim=feature_dim,
            latent_dim=latent_dim,
            dropout=dropout,
        ).to(device)
        generator.load_state_dict(checkpoint["generator"])
        discriminator.load_state_dict(checkpoint["discriminator"])
        architecture = "legacy"

    generator.eval()
    discriminator.eval()
    if not feature_columns:
        feature_columns = [f"f{index}" for index in range(feature_dim)]
    return LoadedCheckpointModels(
        generator=generator,
        discriminator=discriminator,
        feature_dim=feature_dim,
        noise_dim=noise_dim,
        latent_dim=latent_dim,
        dropout=dropout,
        architecture=architecture,
        feature_columns=feature_columns,
    )


def load_training_real_features(
    train_csv: str | Path,
    train_labels_csv: str | Path,
    feature_columns: list[str],
    sample_count: int,
    val_fraction: float,
    seed: int,
) -> np.ndarray:
    train_loaded = load_labeled_frame(train_csv, train_labels_csv, expected_features=len(feature_columns))
    if train_loaded.feature_columns != feature_columns:
        raise ValueError("checkpoint and train feature columns do not match")

    normal_train_df, _ = split_normal_train_val(
        train_loaded.frame,
        val_fraction=val_fraction,
        seed=seed,
    )
    if len(normal_train_df) < sample_count:
        raise ValueError(
            f"not enough normal training rows: requested {sample_count}, available {len(normal_train_df)}"
        )

    sampled = normal_train_df.sample(n=sample_count, random_state=seed).reset_index(drop=True)
    dataset = TabularAnomalyDataset(sampled, feature_columns)
    return dataset.features


@torch.no_grad()
def make_training_style_discriminator_dataset(
    generator: nn.Module,
    real_features: np.ndarray,
    sample_count: int,
    noise_dim: int,
    seed: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    if sample_count < 1:
        raise ValueError("sample_count must be at least 1 per class")
    if real_features.ndim != 2 or real_features.shape[1] < 1:
        raise ValueError("real_features must be a non-empty 2D feature matrix")
    if real_features.shape[0] < sample_count:
        raise ValueError(
            f"not enough real feature rows: requested {sample_count}, available {real_features.shape[0]}"
        )
    if noise_dim < 1:
        raise ValueError("noise_dim must be at least 1")

    set_seed(seed)
    real = np.asarray(real_features[:sample_count], dtype=np.float32)
    generator.to(device).eval()
    fake = generator(noise_uniform(sample_count, noise_dim, device))
    fake_array = fake.detach().cpu().numpy().astype(np.float32)

    features = np.vstack([real, fake_array]).astype(np.float32)
    labels = np.concatenate(
        [
            np.zeros(sample_count, dtype=np.int64),
            np.ones(sample_count, dtype=np.int64),
        ]
    )
    return features, labels


def _loader(features: np.ndarray, labels: np.ndarray, batch_size: int) -> DataLoader:
    tensor_features = torch.as_tensor(features, dtype=torch.float32)
    tensor_labels = torch.as_tensor(labels, dtype=torch.int64)
    return DataLoader(
        TensorDataset(tensor_features, tensor_labels),
        batch_size=batch_size,
        shuffle=False,
    )


@torch.no_grad()
def _evaluate_discriminator(
    discriminator: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    discriminator.eval()
    score_batches: list[np.ndarray] = []
    label_batches: list[np.ndarray] = []
    for features, labels in loader:
        features = features.to(device=device, dtype=torch.float32)
        scores, _ = discriminator(features)
        score_batches.append(scores.detach().cpu().numpy().reshape(-1))
        label_batches.append(labels.detach().cpu().numpy().reshape(-1).astype(np.int64))
    return np.concatenate(score_batches), np.concatenate(label_batches)


def _score_summary(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    real_scores = scores[labels == 0]
    fake_scores = scores[labels == 1]
    return {
        "real_mean": float(np.mean(real_scores)) if real_scores.size else 0.0,
        "fake_mean": float(np.mean(fake_scores)) if fake_scores.size else 0.0,
        "real_std": float(np.std(real_scores)) if real_scores.size else 0.0,
        "fake_std": float(np.std(fake_scores)) if fake_scores.size else 0.0,
    }


def save_synthetic_samples(
    path: str | Path,
    features: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    feature_columns: list[str],
) -> None:
    if features.shape[1] != len(feature_columns):
        raise ValueError("feature column count does not match generated feature width")
    if labels.shape[0] != features.shape[0] or scores.shape[0] != features.shape[0]:
        raise ValueError("features, labels, and discriminator scores must have the same row count")

    frame = pd.DataFrame(features, columns=feature_columns)
    frame.insert(0, "predicted_label", (scores > 0.5).astype(np.int64))
    frame.insert(0, "discriminator_score", scores.astype(np.float64))
    frame.insert(0, "label", labels.astype(np.int64))
    frame.insert(
        0,
        "source",
        np.where(labels == 0, "training_real", "generated_fake"),
    )
    frame.insert(0, "id", np.arange(labels.shape[0], dtype=np.int64))

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_path, index=False)


def run_synthetic_discriminator_experiment(config: SyntheticConfig) -> dict:
    if config.sample_count < 1:
        raise ValueError("sample_count must be at least 1 per class")
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    set_seed(config.seed)
    device = resolve_device(config.device)
    loaded = load_checkpoint_models(config.checkpoint, config, device)
    real_features = load_training_real_features(
        train_csv=config.train,
        train_labels_csv=config.train_labels,
        feature_columns=loaded.feature_columns,
        sample_count=config.sample_count,
        val_fraction=config.val_fraction,
        seed=config.seed,
    )
    features, labels = make_training_style_discriminator_dataset(
        generator=loaded.generator,
        real_features=real_features,
        sample_count=config.sample_count,
        noise_dim=loaded.noise_dim,
        seed=config.seed,
        device=device,
    )
    loader = _loader(features, labels, config.batch_size)
    scores, labels = _evaluate_discriminator(loaded.discriminator, loader, device)
    test_counts = confusion_counts(scores, labels, threshold=0.5)
    test_metrics = compute_binary_metrics(test_counts)
    if config.samples_out:
        save_synthetic_samples(config.samples_out, features, labels, scores, loaded.feature_columns)

    result = {
        "config": asdict(config),
        "loaded_dimensions": {
            "feature_dim": loaded.feature_dim,
            "noise_dim": loaded.noise_dim,
            "latent_dim": loaded.latent_dim,
            "dropout": loaded.dropout,
            "architecture": loaded.architecture,
        },
        "label_convention": {"real": 0, "fake": 1, "threshold": 0.5},
        "source": {
            "checkpoint": str(config.checkpoint),
            "training_real": str(config.train),
            "generated_fake": "loaded_generator",
        },
        "evaluated_rows": int(labels.shape[0]),
        "score_summary": _score_summary(scores, labels),
        "samples_file": str(config.samples_out) if config.samples_out else None,
        "test_counts": test_counts,
        "test_metrics": test_metrics,
    }

    if config.out:
        output_path = Path(config.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")

    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a trained MENSA generator/discriminator checkpoint and report "
            "discriminator accuracy using the same real/fake construction as training."
        )
    )
    parser.add_argument("--checkpoint", default="outputs/mensa_ad/model.pt")
    parser.add_argument("--train", default="data/train.csv")
    parser.add_argument("--train-labels", default="data/train_label.csv")
    parser.add_argument("--sample-count", type=int, default=512, help="Synthetic samples per class.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--feature-dim", type=int)
    parser.add_argument("--noise-dim", type=int)
    parser.add_argument("--latent-dim", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--out")
    parser.add_argument("--samples-out", default="outputs/synthetic_discriminator_samples.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SyntheticConfig(
        checkpoint=args.checkpoint,
        train=args.train,
        train_labels=args.train_labels,
        sample_count=args.sample_count,
        batch_size=args.batch_size,
        val_fraction=args.val_fraction,
        seed=args.seed,
        device=args.device,
        feature_dim=args.feature_dim,
        noise_dim=args.noise_dim,
        latent_dim=args.latent_dim,
        dropout=args.dropout,
        out=args.out,
        samples_out=args.samples_out,
    )
    result = run_synthetic_discriminator_experiment(config)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
