"""FashionMNIST data loading for classification.

Downloads FashionMNIST via torchvision and splits into train/val sets.
The images are converted to tensors (1x28x28) with values in [0, 1].
"""

import torch
from torch.utils.data import random_split


def load_data(split_ratio=0.8, seed=42):
    """Load FashionMNIST and split into train/val datasets.

    Args:
        split_ratio: Fraction of training data used for training (rest is validation).
        seed: Random seed for reproducibility.

    Returns:
        (train_dataset, val_dataset) as random subsets of the FashionMNIST training set.

    Raises:
        ImportError: If torchvision is not installed.
    """
    try:
        from torchvision import datasets, transforms
    except ImportError:
        raise ImportError(
            "torchvision is required for the classification recipe.\n"
            "Install it with: uv pip install torchvision\n"
            "Or: pip install torchvision"
        )

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_dataset = datasets.FashionMNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    train_size = int(len(full_dataset) * split_ratio)
    val_size = len(full_dataset) - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    return train_dataset, val_dataset
