"""Synthetic sine wave data for regression.

Generates data from the function:
    y = 1.0 * sin(4*pi*x) + 0.5 * sin(10*pi*x) + 1.5 * x^2 + noise

where x is in [0, 1] with slight jitter, and noise ~ N(0, 0.05).
"""

import torch
from torch.utils.data import TensorDataset, random_split
from math import pi


def load_data(n=10000, split_ratio=0.8, seed=42):
    """Generate synthetic sine wave regression data.

    Args:
        n: Number of data points.
        split_ratio: Fraction of data used for training.
        seed: Random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset) as TensorDataset subsets.
    """
    torch.manual_seed(seed)

    x_noise = torch.rand(n) * 0.02
    x = torch.linspace(0, 1, n) + x_noise
    x = x.clamp(0, 1)

    noise_level = 0.05
    y = (
        1.0 * torch.sin(4 * pi * x)
        + 0.5 * torch.sin(10 * pi * x)
        + 1.5 * (x ** 2)
        + torch.randn(n) * noise_level
    )

    x = x.view(-1, 1)
    y = y.view(-1, 1)

    full_dataset = TensorDataset(x, y)

    train_size = int(n * split_ratio)
    val_size = n - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    return train_dataset, val_dataset
