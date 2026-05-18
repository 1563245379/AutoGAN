import torch

from mensa_ad.models import DiscriminatorEncoder, GeneratorDecoder


def test_generator_maps_noise_to_normalized_feature_space():
    generator = GeneratorDecoder(noise_dim=10, output_dim=15)
    generator.eval()
    noise = torch.zeros(4, 10)

    generated = generator(noise)

    assert generated.shape == (4, 15)
    assert torch.max(generated).item() <= 1.0
    assert torch.min(generated).item() >= -1.0


def test_discriminator_returns_validity_and_latent_representation():
    discriminator = DiscriminatorEncoder(input_dim=15, latent_dim=128)
    discriminator.eval()
    samples = torch.zeros(4, 15)

    validity, latent = discriminator(samples)

    assert validity.shape == (4, 1)
    assert latent.shape == (4, 128)
    assert torch.max(validity).item() <= 1.0
    assert torch.min(validity).item() >= 0.0


def test_generator_train_mode_accepts_single_sample():
    generator = GeneratorDecoder(noise_dim=10, output_dim=15)
    generator.train()
    noise = torch.zeros(1, 10)

    generated = generator(noise)

    assert generated.shape == (1, 15)


def test_discriminator_train_mode_accepts_single_sample():
    discriminator = DiscriminatorEncoder(input_dim=15, latent_dim=128)
    discriminator.train()
    samples = torch.zeros(1, 15)

    validity, latent = discriminator(samples)

    assert validity.shape == (1, 1)
    assert latent.shape == (1, 128)
