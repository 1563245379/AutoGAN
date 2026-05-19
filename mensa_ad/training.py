from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from mensa_ad.models import DiscriminatorEncoder, GeneratorDecoder


@dataclass(frozen=True)
class TrainConfig:
    input_dim: int = 15
    noise_dim: int = 10
    latent_dim: int = 128
    batch_size: int = 1024
    epochs: int = 50
    lr: float = 0.0002
    dropout: float = 0.2
    score_samples: int = 4
    val_score_interval: int = 0
    device: str = "auto"
    seed: int = 42


@dataclass
class TrainState:
    generator: GeneratorDecoder
    discriminator: DiscriminatorEncoder
    history: list[dict[str, float]]
    config: TrainConfig

    def checkpoint(self, feature_columns: list[str]) -> dict[str, Any]:
        return {
            "generator": self.generator.state_dict(),
            "discriminator": self.discriminator.state_dict(),
            "feature_columns": feature_columns,
            "config": asdict(self.config),
            "history": self.history,
        }


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def noise_uniform(batch_size: int, noise_dim: int, device: torch.device) -> torch.Tensor:
    return torch.empty(batch_size, noise_dim, device=device).uniform_(-1.0, 1.0)


def validate_config(config: TrainConfig, *, training: bool) -> None:
    if config.batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    if training and config.epochs < 1:
        raise ValueError("epochs must be at least 1")
    if config.score_samples < 1:
        raise ValueError("score_samples must be at least 1")
    if config.val_score_interval < 0:
        raise ValueError("val_score_interval must be non-negative")


def _loader(dataset: Dataset, batch_size: int, shuffle: bool, drop_last: bool) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)


def train_mensa(
    train_dataset: Dataset,
    val_dataset: Dataset | None,
    config: TrainConfig,
) -> TrainState:
    validate_config(config, training=True)
    set_seed(config.seed)
    device = resolve_device(config.device)
    generator = GeneratorDecoder(
        noise_dim=config.noise_dim,
        output_dim=config.input_dim,
        dropout=config.dropout,
    ).to(device)
    discriminator = DiscriminatorEncoder(
        input_dim=config.input_dim,
        latent_dim=config.latent_dim,
        dropout=config.dropout,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=config.lr)
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=config.lr)
    train_loader = _loader(train_dataset, config.batch_size, shuffle=True, drop_last=False)
    history: list[dict[str, float]] = []

    bar = tqdm(range(1, config.epochs + 1), desc="Training AutoGAN")
    for epoch in bar:
        generator.train()
        discriminator.train()
        d_losses: list[float] = []
        g_losses: list[float] = []

        for real, _, _ in train_loader:
            real = real.to(device=device, dtype=torch.float32)
            batch_size = real.shape[0]
            real_targets = torch.full((batch_size, 1), 0.1, device=device)
            fake_targets = torch.full((batch_size, 1), 0.9, device=device)
            g_targets = torch.zeros(batch_size, 1, device=device)

            discriminator_optimizer.zero_grad(set_to_none=True)
            fake = generator(noise_uniform(batch_size, config.noise_dim, device)).detach()
            combined = torch.cat([real, fake], dim=0)
            combined_targets = torch.cat([real_targets, fake_targets], dim=0)
            validity, _ = discriminator(combined)
            d_loss = criterion(validity, combined_targets)
            d_loss.backward()
            discriminator_optimizer.step()

            generator_optimizer.zero_grad(set_to_none=True)
            fake = generator(noise_uniform(batch_size, config.noise_dim, device))
            fake_validity, _ = discriminator(fake)
            g_loss = criterion(fake_validity, g_targets)
            g_loss.backward()
            generator_optimizer.step()

            d_losses.append(float(d_loss.detach().cpu().item()))
            g_losses.append(float(g_loss.detach().cpu().item()))

            bar.set_postfix({"d_loss": float(d_loss.detach().cpu().item()), "g_loss": float(g_loss.detach().cpu().item())})

        row = {
            "epoch": float(epoch),
            "discriminator_loss": float(np.mean(d_losses)) if d_losses else 0.0,
            "generator_loss": float(np.mean(g_losses)) if g_losses else 0.0,
        }
        should_score_val = (
            config.val_score_interval > 0
            and epoch % config.val_score_interval == 0
            and val_dataset is not None
            and len(val_dataset) > 0
        )
        if should_score_val:
            val_scores = score_dataset(val_dataset, generator, discriminator, config)
            row["val_score_mean"] = float(np.mean(val_scores))
        history.append(row)

    return TrainState(generator=generator, discriminator=discriminator, history=history, config=config)


@torch.no_grad()
def score_dataset(
    dataset: Dataset,
    generator: GeneratorDecoder,
    discriminator: DiscriminatorEncoder,
    config: TrainConfig,
) -> np.ndarray:
    validate_config(config, training=False)
    device = resolve_device(config.device)
    generator.to(device).eval()
    discriminator.to(device).eval()
    loader = _loader(dataset, config.batch_size, shuffle=False, drop_last=False)
    all_scores: list[np.ndarray] = []

    for real, _, _ in tqdm(loader, desc="Scoring dataset"):
        real = real.to(device=device, dtype=torch.float32)
        _, real_latent = discriminator(real)
        sample_scores: list[torch.Tensor] = []
        for _ in range(config.score_samples):
            fake = generator(noise_uniform(real.shape[0], config.noise_dim, device))
            _, fake_latent = discriminator(fake)
            adversarial = torch.norm(real_latent - fake_latent, p=2, dim=1)
            sample_scores.append(adversarial)
        stacked = torch.stack(sample_scores, dim=0)
        batch_scores = torch.min(stacked, dim=0).values
        all_scores.append(batch_scores.detach().cpu().numpy())

    if not all_scores:
        return np.empty((0,), dtype=np.float64)
    return np.concatenate(all_scores).astype(np.float64)
