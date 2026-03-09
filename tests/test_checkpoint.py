"""Tests for checkpoint.py — CheckpointManager and SeedManifest."""

import os
import sys

import pytest
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from checkpoint import CheckpointManager, SeedManifest
from config import default_run_config


def _make_model_optimizer_scheduler():
    """Helper: create a simple model, optimizer, and scheduler."""
    torch.manual_seed(42)
    config = default_run_config()
    model = config.create_model()
    optimizer = config.create_optimizer(model)
    scheduler = config.create_scheduler(optimizer)
    return model, optimizer, scheduler


# ---------------------------------------------------------------------------
# CheckpointManager
# ---------------------------------------------------------------------------

def test_save_load_checkpoint(tmp_path):
    """Save then load checkpoint restores model state."""
    model, optimizer, scheduler = _make_model_optimizer_scheduler()
    manager = CheckpointManager(run_dir=str(tmp_path))

    # Capture original state
    original_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Save
    ckpt_path = os.path.join(str(tmp_path), "test_ckpt.pt")
    manager.save_checkpoint(
        ckpt_path, model, optimizer, scheduler,
        epoch=5, val_loss=0.123, metrics={"mse": 0.123},
        config_hash="abc123",
    )

    # Modify model weights
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(999.0)

    # Verify weights changed
    for key in original_state:
        assert not torch.equal(model.state_dict()[key], original_state[key])

    # Load
    model2, optimizer2, scheduler2 = _make_model_optimizer_scheduler()
    checkpoint = manager.load_checkpoint(
        ckpt_path, model2, optimizer2, scheduler2,
        device="cpu", config_hash="abc123",
    )

    # Restored state should match original
    for key in original_state:
        assert torch.equal(model2.state_dict()[key], original_state[key])
    assert checkpoint["epoch"] == 5
    assert checkpoint["val_loss"] == pytest.approx(0.123)


def test_checkpoint_config_hash_mismatch(tmp_path):
    """Mismatched config hash raises ValueError."""
    model, optimizer, scheduler = _make_model_optimizer_scheduler()
    manager = CheckpointManager(run_dir=str(tmp_path))

    ckpt_path = os.path.join(str(tmp_path), "test_ckpt.pt")
    manager.save_checkpoint(
        ckpt_path, model, optimizer, scheduler,
        epoch=1, val_loss=0.5, metrics={},
        config_hash="hash_A",
    )

    model2, optimizer2, scheduler2 = _make_model_optimizer_scheduler()
    with pytest.raises(ValueError, match="Config hash mismatch"):
        manager.load_checkpoint(
            ckpt_path, model2, optimizer2, scheduler2,
            device="cpu", config_hash="hash_B",
        )


def test_checkpoint_cleanup(tmp_path):
    """Old periodic checkpoints are removed, keeping only keep_last_k."""
    model, optimizer, scheduler = _make_model_optimizer_scheduler()
    manager = CheckpointManager(
        run_dir=str(tmp_path), save_every_n=1, keep_last_k=2,
    )

    # Save 5 periodic checkpoints
    for epoch in range(5):
        path = os.path.join(str(tmp_path), f"checkpoint_epoch_{epoch}.pt")
        manager.save_checkpoint(
            path, model, optimizer, scheduler,
            epoch=epoch, val_loss=0.5, metrics={},
        )
        manager._cleanup_old_checkpoints()

    # Only keep_last_k (2) should remain
    remaining = [f for f in os.listdir(str(tmp_path)) if f.startswith("checkpoint_epoch_")]
    assert len(remaining) == 2
    # The last two should survive
    assert "checkpoint_epoch_3.pt" in remaining
    assert "checkpoint_epoch_4.pt" in remaining


# ---------------------------------------------------------------------------
# SeedManifest
# ---------------------------------------------------------------------------

def test_seed_manifest_complete(tmp_path):
    """mark_complete and is_complete work correctly."""
    manifest = SeedManifest(str(tmp_path))
    assert not manifest.is_complete(42)

    manifest.mark_complete(42, val_loss=0.05, metrics={"mse": 0.05})
    assert manifest.is_complete(42)
    assert not manifest.is_complete(99)
    assert manifest.get_complete_count() == 1


def test_seed_manifest_persistence(tmp_path):
    """Manifest persists to disk and reloads."""
    manifest = SeedManifest(str(tmp_path))
    manifest.mark_complete(42, val_loss=0.05)
    manifest.mark_complete(123, val_loss=0.03)

    # Reload from disk
    manifest2 = SeedManifest(str(tmp_path))
    assert manifest2.is_complete(42)
    assert manifest2.is_complete(123)
    assert manifest2.get_complete_count() == 2
    assert manifest2.get_total_loss() == pytest.approx(0.08)
