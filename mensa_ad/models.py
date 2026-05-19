from __future__ import annotations

import torch
from torch import nn


def _activation(name: str) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "leaky_relu":
        return nn.LeakyReLU(0.2, inplace=True)
    raise ValueError(f"unsupported activation: {name}")


class DenseBlock(nn.Sequential):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        activation: str = "leaky_relu",
        dropout: float = 0.2,
        normalize: bool = True,
    ) -> None:
        layers: list[nn.Module] = [nn.Linear(input_dim, output_dim)]
        if normalize:
            layers.append(nn.LayerNorm(output_dim))
        layers.append(_activation(activation))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        super().__init__(*layers)


class GeneratorDecoder(nn.Module):
    """MENSA Generator-Decoder for anomaly detection."""

    def __init__(
        self,
        noise_dim: int = 10,
        output_dim: int = 15,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.network = nn.Sequential(
            DenseBlock(noise_dim, 128, activation="leaky_relu", dropout=dropout, normalize=True),
            DenseBlock(128, 256, activation="leaky_relu", dropout=dropout, normalize=True),
            DenseBlock(256, 512, activation="leaky_relu", dropout=dropout, normalize=True),
            DenseBlock(512, 512, activation="relu", dropout=dropout, normalize=True),
            nn.Linear(512, output_dim),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor) -> torch.Tensor:
        return self.network(noise)


class DiscriminatorEncoder(nn.Module):
    """MENSA Discriminator-Encoder with an exported latent model."""

    def __init__(
        self,
        input_dim: int = 15,
        latent_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = nn.Sequential(
            DenseBlock(input_dim, 600, activation="leaky_relu", dropout=dropout, normalize=True),
            DenseBlock(600, 256, activation="leaky_relu", dropout=dropout, normalize=True),
            DenseBlock(256, latent_dim, activation="relu", dropout=dropout, normalize=True),
        )
        self.validity = nn.Linear(latent_dim, 1)

    def encode(self, samples: torch.Tensor) -> torch.Tensor:
        return self.encoder(samples)

    def forward(self, samples: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        latent = self.encode(samples)
        validity = self.validity(latent)
        return validity, latent
