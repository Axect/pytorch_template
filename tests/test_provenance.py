"""Tests for provenance.py — environment capture, config hashing, file output."""

import os
import sys
import time

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from provenance import capture_environment, compute_config_hash, save_provenance
from config import default_run_config


# ---------------------------------------------------------------------------
# Environment capture
# ---------------------------------------------------------------------------

def test_capture_environment():
    """Returns dict with expected keys."""
    env = capture_environment()
    assert isinstance(env, dict)
    assert "python_version" in env
    assert "platform" in env
    assert "hostname" in env
    assert "torch_version" in env


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

def test_config_hash_determinism():
    """Same config produces same hash."""
    config_a = default_run_config()
    config_b = default_run_config()
    assert compute_config_hash(config_a) == compute_config_hash(config_b)


def test_config_hash_changes():
    """Different config produces different hash."""
    config_a = default_run_config()
    config_b = config_a.with_overrides(epochs=999)
    assert compute_config_hash(config_a) != compute_config_hash(config_b)


# ---------------------------------------------------------------------------
# Save provenance files
# ---------------------------------------------------------------------------

def test_save_provenance(tmp_path):
    """Creates env_snapshot.yaml and run_metadata.yaml files."""
    torch.manual_seed(42)
    config = default_run_config()
    model = config.create_model()

    start_time = time.time()
    end_time = start_time + 1.0  # Simulate 1 second of training

    run_path = str(tmp_path)
    save_provenance(run_path, config, model, "cpu", start_time, end_time)

    assert os.path.isfile(os.path.join(run_path, "env_snapshot.yaml"))
    assert os.path.isfile(os.path.join(run_path, "run_metadata.yaml"))

    # Verify the metadata file contains expected fields
    import yaml
    with open(os.path.join(run_path, "run_metadata.yaml"), "r") as f:
        metadata = yaml.safe_load(f)
    assert "config_hash" in metadata
    assert "training_time_seconds" in metadata
    assert "total_parameters" in metadata
    assert "trainable_parameters" in metadata
