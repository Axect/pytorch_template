"""Tests for config.py — RunConfig creation, validation, serialization."""

import os
import sys
import tempfile

import pytest
import yaml

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import RunConfig, default_run_config


# ---------------------------------------------------------------------------
# Loading / validation
# ---------------------------------------------------------------------------

def test_valid_config_loads():
    """Load configs/run_template.yaml successfully."""
    config_path = os.path.join(PROJECT_ROOT, "configs", "run_template.yaml")
    config = RunConfig.from_yaml(config_path)
    assert isinstance(config, RunConfig)
    assert config.epochs > 0


def test_invalid_epochs():
    """epochs=0 raises ValueError."""
    with pytest.raises(ValueError, match="epochs must be > 0"):
        RunConfig(
            project="test", device="cpu", seeds=[1],
            net="model.MLP", optimizer="torch.optim.Adam",
            scheduler="torch.optim.lr_scheduler.StepLR",
            epochs=0, batch_size=32,
            net_config={"nodes": 8, "layers": 1},
            optimizer_config={"lr": 1e-3},
            scheduler_config={"step_size": 10},
        )


def test_invalid_batch_size():
    """batch_size=0 raises ValueError."""
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        RunConfig(
            project="test", device="cpu", seeds=[1],
            net="model.MLP", optimizer="torch.optim.Adam",
            scheduler="torch.optim.lr_scheduler.StepLR",
            epochs=10, batch_size=0,
            net_config={"nodes": 8, "layers": 1},
            optimizer_config={"lr": 1e-3},
            scheduler_config={"step_size": 10},
        )


def test_empty_seeds():
    """Empty seeds list raises ValueError."""
    with pytest.raises(ValueError, match="seeds must be non-empty"):
        RunConfig(
            project="test", device="cpu", seeds=[],
            net="model.MLP", optimizer="torch.optim.Adam",
            scheduler="torch.optim.lr_scheduler.StepLR",
            epochs=10, batch_size=32,
            net_config={"nodes": 8, "layers": 1},
            optimizer_config={"lr": 1e-3},
            scheduler_config={"step_size": 10},
        )


def test_invalid_net_format():
    """net without '.' raises ValueError."""
    with pytest.raises(ValueError, match="must be in module.Class format"):
        RunConfig(
            project="test", device="cpu", seeds=[1],
            net="MLP", optimizer="torch.optim.Adam",
            scheduler="torch.optim.lr_scheduler.StepLR",
            epochs=10, batch_size=32,
            net_config={"nodes": 8, "layers": 1},
            optimizer_config={"lr": 1e-3},
            scheduler_config={"step_size": 10},
        )


def test_frozen_config(sample_config):
    """Setting attribute after creation raises AttributeError."""
    with pytest.raises(AttributeError, match="frozen"):
        sample_config.epochs = 999


def test_with_overrides(sample_config):
    """Creates new config with changed fields, original unchanged."""
    original_epochs = sample_config.epochs
    new_config = sample_config.with_overrides(epochs=100)
    assert new_config.epochs == 100
    assert sample_config.epochs == original_epochs


def test_with_overrides_deep_merge(sample_config):
    """Nested dict fields are merged, not replaced."""
    original_lr = sample_config.optimizer_config["lr"]
    new_config = sample_config.with_overrides(
        optimizer_config={"weight_decay": 0.01}
    )
    # The new config should have both the original lr and the new weight_decay
    assert new_config.optimizer_config["lr"] == original_lr
    assert new_config.optimizer_config["weight_decay"] == 0.01
    # Original should be untouched
    assert "weight_decay" not in sample_config.optimizer_config


def test_config_yaml_roundtrip(sample_config):
    """from_yaml then to_yaml produces equivalent config."""
    with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False, mode="w") as f:
        tmp_path = f.name

    try:
        sample_config.to_yaml(tmp_path)
        reloaded = RunConfig.from_yaml(tmp_path)
        assert reloaded.project == sample_config.project
        assert reloaded.epochs == sample_config.epochs
        assert reloaded.batch_size == sample_config.batch_size
        assert reloaded.seeds == sample_config.seeds
        assert reloaded.net == sample_config.net
        assert reloaded.optimizer_config == sample_config.optimizer_config
        assert reloaded.scheduler_config == sample_config.scheduler_config
    finally:
        os.unlink(tmp_path)


def test_duplicate_yaml_key():
    """YAML with duplicate keys raises yaml.YAMLError."""
    duplicate_yaml = """\
project: Test1
project: Test2
device: cpu
seeds: [1]
net: model.MLP
optimizer: torch.optim.Adam
scheduler: torch.optim.lr_scheduler.StepLR
epochs: 10
batch_size: 32
net_config:
  nodes: 8
  layers: 1
optimizer_config:
  lr: 0.001
scheduler_config:
  step_size: 10
"""
    with tempfile.NamedTemporaryFile(
        suffix=".yaml", delete=False, mode="w"
    ) as f:
        f.write(duplicate_yaml)
        tmp_path = f.name

    try:
        with pytest.raises(yaml.YAMLError, match="Duplicate key"):
            RunConfig.from_yaml(tmp_path)
    finally:
        os.unlink(tmp_path)


def test_create_criterion(sample_config):
    """create_criterion() returns a callable loss function."""
    criterion = sample_config.create_criterion()
    assert callable(criterion)


def test_default_run_config():
    """default_run_config() creates a valid RunConfig."""
    config = default_run_config()
    assert isinstance(config, RunConfig)
    assert config.project == "PyTorch_Template"
    assert config.device == "cpu"
    assert config.epochs > 0
    assert config.batch_size > 0
    assert len(config.seeds) > 0
