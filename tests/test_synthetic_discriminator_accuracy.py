from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from mensa_ad.models import DenseBlock, DiscriminatorEncoder, GeneratorDecoder
from synthetic_discriminator_accuracy import (
    SyntheticConfig,
    make_training_style_discriminator_dataset,
    run_synthetic_discriminator_experiment,
)


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


def test_make_training_style_discriminator_dataset_uses_normal_training_rows_and_project_labels():
    generator = GeneratorDecoder(noise_dim=2, output_dim=3, dropout=0.0)
    generator.eval()
    normal_train_features = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
            [0.7, 0.8, 0.9],
        ],
        dtype=np.float32,
    )

    features, labels = make_training_style_discriminator_dataset(
        generator=generator,
        real_features=normal_train_features,
        sample_count=2,
        noise_dim=2,
        seed=123,
        device=torch.device("cpu"),
    )

    assert features.shape == (4, 3)
    assert labels.tolist() == [0, 0, 1, 1]
    assert {tuple(row) for row in features[:2]}.issubset({tuple(row) for row in normal_train_features})
    assert np.all(np.isfinite(features))


def test_run_synthetic_discriminator_experiment_loads_checkpoint_without_training(monkeypatch):
    checkpoint_path = Path("outputs") / "_pytest_synthetic_discriminator_checkpoint.pt"
    samples_path = Path("outputs") / "_pytest_synthetic_discriminator_samples.csv"
    train_path = Path("outputs") / "_pytest_synthetic_discriminator_train.csv"
    train_labels_path = Path("outputs") / "_pytest_synthetic_discriminator_train_label.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    generator = GeneratorDecoder(noise_dim=2, output_dim=3, dropout=0.0)
    discriminator = DiscriminatorEncoder(input_dim=3, latent_dim=4, dropout=0.0)

    try:
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "feature_columns": ["f0", "f1", "f2"],
                "config": {
                    "input_dim": 3,
                    "noise_dim": 2,
                    "latent_dim": 4,
                    "dropout": 0.0,
                    "device": "cpu",
                    "seed": 123,
                },
                "history": [],
            },
            checkpoint_path,
        )
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "f0": [0.1, 0.2, 9.0, 0.3],
                "f1": [0.4, 0.5, 9.0, 0.6],
                "f2": [0.7, 0.8, 9.0, 0.9],
            }
        ).to_csv(train_path, index=False)
        pd.DataFrame({"id": [1, 2, 3, 4], "anomaly": [0, 0, 1, 0]}).to_csv(train_labels_path, index=False)

        def fail_if_training_starts(*args, **kwargs):
            raise AssertionError("script should load weights and evaluate without training")

        monkeypatch.setattr(torch.optim, "Adam", fail_if_training_starts)

        result = run_synthetic_discriminator_experiment(
            SyntheticConfig(
                checkpoint=str(checkpoint_path),
                train=str(train_path),
                train_labels=str(train_labels_path),
                sample_count=2,
                batch_size=4,
                val_fraction=0.0,
                seed=123,
                device="cpu",
                samples_out=str(samples_path),
            )
        )

        assert result["evaluated_rows"] == 4
        assert result["config"]["checkpoint"] == str(checkpoint_path)
        assert 0.0 <= result["test_metrics"]["accuracy"] <= 1.0
        assert result["source"]["generated_fake"] == "loaded_generator"
        assert result["samples_file"] == str(samples_path)

        saved_samples = pd.read_csv(samples_path)
        assert saved_samples.columns.tolist() == [
            "id",
            "source",
            "label",
            "discriminator_score",
            "predicted_label",
            "f0",
            "f1",
            "f2",
        ]
        assert saved_samples["label"].tolist() == [0] * 2 + [1] * 2
        assert saved_samples["predicted_label"].isin([0, 1]).all()
        assert (
            saved_samples["predicted_label"].to_numpy()
            == (saved_samples["discriminator_score"].to_numpy() > 0.5).astype(int)
        ).all()
        assert saved_samples["source"].tolist() == ["training_real"] * 2 + ["generated_fake"] * 2
        assert not (saved_samples.loc[saved_samples["label"].eq(0), ["f0", "f1", "f2"]] == 9.0).any().any()
    finally:
        checkpoint_path.unlink(missing_ok=True)
        samples_path.unlink(missing_ok=True)
        train_path.unlink(missing_ok=True)
        train_labels_path.unlink(missing_ok=True)


def test_run_synthetic_discriminator_experiment_loads_legacy_checkpoint_architecture():
    checkpoint_path = Path("outputs") / "_pytest_legacy_synthetic_discriminator_checkpoint.pt"
    train_path = Path("outputs") / "_pytest_legacy_synthetic_discriminator_train.csv"
    train_labels_path = Path("outputs") / "_pytest_legacy_synthetic_discriminator_train_label.csv"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    generator = _LegacyGeneratorDecoder(noise_dim=2, output_dim=3, dropout=0.0)
    discriminator = _LegacyDiscriminatorEncoder(input_dim=3, latent_dim=4, dropout=0.0)

    try:
        torch.save(
            {
                "generator": generator.state_dict(),
                "discriminator": discriminator.state_dict(),
                "feature_columns": ["f0", "f1", "f2"],
                "config": {
                    "input_dim": 3,
                    "noise_dim": 2,
                    "latent_dim": 4,
                    "dropout": 0.0,
                    "device": "cpu",
                    "seed": 123,
                },
                "history": [],
            },
            checkpoint_path,
        )
        pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "f0": [0.1, 0.2, 0.3, 0.4],
                "f1": [0.4, 0.5, 0.6, 0.7],
                "f2": [0.7, 0.8, 0.9, 1.0],
            }
        ).to_csv(train_path, index=False)
        pd.DataFrame({"id": [1, 2, 3, 4], "anomaly": [0, 0, 0, 0]}).to_csv(train_labels_path, index=False)

        result = run_synthetic_discriminator_experiment(
            SyntheticConfig(
                checkpoint=str(checkpoint_path),
                train=str(train_path),
                train_labels=str(train_labels_path),
                sample_count=4,
                batch_size=4,
                val_fraction=0.0,
                seed=123,
                device="cpu",
            )
        )

        assert result["evaluated_rows"] == 8
        assert result["loaded_dimensions"]["architecture"] == "legacy"
    finally:
        checkpoint_path.unlink(missing_ok=True)
        train_path.unlink(missing_ok=True)
        train_labels_path.unlink(missing_ok=True)
