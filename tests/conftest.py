"""Shared fixtures for PyTorch template test suite."""

import os
import sys
import shutil
import tempfile

import pytest
import torch

# Ensure the project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import default_run_config
from util import load_data


@pytest.fixture
def sample_config():
    """Return a default RunConfig for testing."""
    return default_run_config()


@pytest.fixture
def tiny_model(sample_config):
    """Create a small MLP using the sample config."""
    torch.manual_seed(42)
    model = sample_config.create_model()
    return model


@pytest.fixture
def tiny_dataset():
    """Create a tiny synthetic dataset (100 samples)."""
    torch.manual_seed(42)
    train_dataset, val_dataset = load_data(n=100)
    return train_dataset, val_dataset


@pytest.fixture
def tmp_run_dir():
    """Create a temporary directory for run outputs, cleaned up after test."""
    dirpath = tempfile.mkdtemp(prefix="pytest_run_")
    yield dirpath
    shutil.rmtree(dirpath, ignore_errors=True)
