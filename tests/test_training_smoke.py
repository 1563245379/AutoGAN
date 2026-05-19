import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from mensa_ad.data import TabularAnomalyDataset
from mensa_ad.training import TrainConfig, score_dataset, train_mensa


class _ZeroGenerator(nn.Module):
    def __init__(self, output_dim: int) -> None:
        super().__init__()
        self.output_dim = output_dim

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return torch.zeros(noise.shape[0], self.output_dim, device=noise.device)


class _IdentityLatentDiscriminator(nn.Module):
    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        validity = torch.zeros(features.shape[0], 1, device=features.device)
        return validity, features


def test_score_dataset_uses_mse_for_latent_adversarial_distance():
    frame = pd.DataFrame(
        {
            "id": [1],
            "f0": [1.0],
            "f1": [0.5],
            "anomaly": [0],
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])
    config = TrainConfig(
        input_dim=2,
        noise_dim=2,
        batch_size=1,
        score_samples=1,
        device="cpu",
    )

    scores = score_dataset(dataset, _ZeroGenerator(output_dim=2), _IdentityLatentDiscriminator(), config)

    assert np.allclose(scores, np.array([0.625]))


def test_train_mensa_runs_one_epoch_and_scores_samples():
    frame = pd.DataFrame(
        {
            "id": list(range(12)),
            "f0": np.linspace(-0.2, 0.2, 12),
            "f1": np.linspace(0.2, -0.2, 12),
            "anomaly": [0] * 12,
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])
    config = TrainConfig(
        input_dim=2,
        noise_dim=10,
        latent_dim=16,
        batch_size=4,
        epochs=1,
        score_samples=2,
        device="cpu",
        seed=123,
    )

    state = train_mensa(train_dataset=dataset, val_dataset=dataset, config=config)
    scores = score_dataset(dataset, state.generator, state.discriminator, config)

    assert len(state.history) == 1
    assert "val_score_mean" not in state.history[0]
    assert scores.shape == (12,)
    assert np.all(np.isfinite(scores))


def test_train_mensa_processes_dataset_smaller_than_batch_size():
    frame = pd.DataFrame(
        {
            "id": list(range(3)),
            "f0": np.linspace(-0.1, 0.1, 3),
            "f1": np.linspace(0.1, -0.1, 3),
            "anomaly": [0] * 3,
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])
    config = TrainConfig(
        input_dim=2,
        noise_dim=10,
        latent_dim=16,
        batch_size=8,
        epochs=1,
        score_samples=2,
        device="cpu",
        seed=123,
    )

    state = train_mensa(train_dataset=dataset, val_dataset=dataset, config=config)
    scores = score_dataset(dataset, state.generator, state.discriminator, config)

    row = state.history[0]
    assert not (row["discriminator_loss"] == 0.0 and row["generator_loss"] == 0.0)
    assert scores.shape == (3,)
    assert np.all(np.isfinite(scores))


def test_train_config_rejects_invalid_score_samples():
    frame = pd.DataFrame(
        {
            "id": list(range(4)),
            "f0": np.linspace(-0.1, 0.1, 4),
            "f1": np.linspace(0.1, -0.1, 4),
            "anomaly": [0] * 4,
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])
    config = TrainConfig(
        input_dim=2,
        noise_dim=10,
        latent_dim=16,
        batch_size=4,
        epochs=1,
        score_samples=0,
        device="cpu",
        seed=123,
    )

    with pytest.raises(ValueError, match="score_samples"):
        train_mensa(train_dataset=dataset, val_dataset=dataset, config=config)


def test_train_config_rejects_invalid_batch_size():
    frame = pd.DataFrame(
        {
            "id": list(range(4)),
            "f0": np.linspace(-0.1, 0.1, 4),
            "f1": np.linspace(0.1, -0.1, 4),
            "anomaly": [0] * 4,
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])
    config = TrainConfig(
        input_dim=2,
        noise_dim=10,
        latent_dim=16,
        batch_size=0,
        epochs=1,
        score_samples=2,
        device="cpu",
        seed=123,
    )

    with pytest.raises(ValueError, match="batch_size"):
        train_mensa(train_dataset=dataset, val_dataset=dataset, config=config)


def test_train_config_rejects_invalid_epochs():
    frame = pd.DataFrame(
        {
            "id": list(range(4)),
            "f0": np.linspace(-0.1, 0.1, 4),
            "f1": np.linspace(0.1, -0.1, 4),
            "anomaly": [0] * 4,
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])
    config = TrainConfig(
        input_dim=2,
        noise_dim=10,
        latent_dim=16,
        batch_size=4,
        epochs=0,
        score_samples=2,
        device="cpu",
        seed=123,
    )

    with pytest.raises(ValueError, match="epochs"):
        train_mensa(train_dataset=dataset, val_dataset=dataset, config=config)


def test_train_config_rejects_negative_reconstruction_weight():
    frame = pd.DataFrame(
        {
            "id": list(range(4)),
            "f0": np.linspace(-0.1, 0.1, 4),
            "f1": np.linspace(0.1, -0.1, 4),
            "anomaly": [0] * 4,
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])
    config = TrainConfig(
        input_dim=2,
        noise_dim=10,
        latent_dim=16,
        batch_size=4,
        epochs=1,
        score_samples=2,
        reconstruction_weight=-0.1,
        device="cpu",
        seed=123,
    )

    with pytest.raises(ValueError, match="reconstruction_weight"):
        train_mensa(train_dataset=dataset, val_dataset=dataset, config=config)


def test_train_mensa_records_validation_score_when_interval_enabled():
    frame = pd.DataFrame(
        {
            "id": list(range(8)),
            "f0": np.linspace(-0.2, 0.2, 8),
            "f1": np.linspace(0.2, -0.2, 8),
            "anomaly": [0] * 8,
        }
    )
    dataset = TabularAnomalyDataset(frame, ["f0", "f1"])
    config = TrainConfig(
        input_dim=2,
        noise_dim=10,
        latent_dim=16,
        batch_size=4,
        epochs=1,
        score_samples=2,
        val_score_interval=1,
        device="cpu",
        seed=123,
    )

    state = train_mensa(train_dataset=dataset, val_dataset=dataset, config=config)

    assert "val_score_mean" in state.history[0]
    assert np.isfinite(state.history[0]["val_score_mean"])
