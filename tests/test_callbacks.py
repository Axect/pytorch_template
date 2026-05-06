"""Tests for callbacks.py — priority ordering, event dispatch, callback logic."""

import os
import sys
import math
from unittest.mock import MagicMock

import pytest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch

from callbacks import (
    TrainingCallback,
    CallbackRunner,
    NaNDetectionCallback,
    EarlyStoppingCallback,
    LossPredictionCallback,
    LatestModelCallback,
)
from config import default_run_config


# ---------------------------------------------------------------------------
# Priority & dispatch
# ---------------------------------------------------------------------------

class LowPriorityCB(TrainingCallback):
    priority = 10
    def __init__(self):
        self.called = False
    def on_train_begin(self, trainer, **kwargs):
        self.called = True


class HighPriorityCB(TrainingCallback):
    priority = 200
    def __init__(self):
        self.called = False
    def on_train_begin(self, trainer, **kwargs):
        self.called = True


def test_callback_priority_order():
    """Callbacks are sorted by priority (lower number first)."""
    high = HighPriorityCB()
    low = LowPriorityCB()
    runner = CallbackRunner([high, low])
    assert runner.callbacks[0].priority < runner.callbacks[1].priority
    assert runner.callbacks[0] is low
    assert runner.callbacks[1] is high


def test_callback_fire_event():
    """fire() calls the correct method on each callback."""
    order = []

    class CB_A(TrainingCallback):
        priority = 1
        def on_train_begin(self, trainer, **kwargs):
            order.append("A")

    class CB_B(TrainingCallback):
        priority = 2
        def on_train_begin(self, trainer, **kwargs):
            order.append("B")

    runner = CallbackRunner([CB_B(), CB_A()])
    runner.fire("on_train_begin", trainer=MagicMock())
    assert order == ["A", "B"]


# ---------------------------------------------------------------------------
# NaN detection
# ---------------------------------------------------------------------------

def test_nan_detection():
    """NaNDetectionCallback sets flag on NaN loss."""
    cb = NaNDetectionCallback()
    trainer = MagicMock()
    # Normal loss should not trigger
    cb.on_epoch_end(trainer=trainer, epoch=0, train_loss=0.5, val_loss=0.4, metrics={})
    assert not cb.nan_detected
    # NaN loss should trigger
    cb.on_epoch_end(trainer=trainer, epoch=1, train_loss=float("nan"), val_loss=0.4, metrics={})
    assert cb.nan_detected


# ---------------------------------------------------------------------------
# Early stopping
# ---------------------------------------------------------------------------

def test_early_stopping_min():
    """EarlyStoppingCallback stops when no improvement (mode=min)."""
    cb = EarlyStoppingCallback(patience=3, mode="min", min_delta=0.0001)
    trainer = MagicMock()
    # First call sets baseline
    cb.on_val_end(trainer=trainer, epoch=0, val_loss=1.0, metrics={})
    assert not cb.should_stop
    # No improvement (worse loss) for 3 epochs
    for i in range(1, 4):
        cb.on_val_end(trainer=trainer, epoch=i, val_loss=1.1, metrics={})
    assert cb.should_stop


def test_early_stopping_improvement_resets():
    """Counter resets on improvement."""
    cb = EarlyStoppingCallback(patience=3, mode="min", min_delta=0.0001)
    trainer = MagicMock()
    cb.on_val_end(trainer=trainer, epoch=0, val_loss=1.0, metrics={})
    cb.on_val_end(trainer=trainer, epoch=1, val_loss=1.1, metrics={})
    cb.on_val_end(trainer=trainer, epoch=2, val_loss=1.1, metrics={})
    assert cb.counter == 2
    # Improvement should reset counter
    cb.on_val_end(trainer=trainer, epoch=3, val_loss=0.5, metrics={})
    assert cb.counter == 0
    assert not cb.should_stop


def test_early_stopping_state_dict():
    """state_dict / load_state_dict round-trip."""
    cb = EarlyStoppingCallback(patience=5, mode="min", min_delta=0.001)
    trainer = MagicMock()
    cb.on_val_end(trainer=trainer, epoch=0, val_loss=1.0, metrics={})
    cb.on_val_end(trainer=trainer, epoch=1, val_loss=1.0, metrics={})

    state = cb.state_dict()
    assert "counter" in state
    assert "best_value" in state
    assert "should_stop" in state

    cb2 = EarlyStoppingCallback(patience=5, mode="min", min_delta=0.001)
    cb2.load_state_dict(state)
    assert cb2.counter == cb.counter
    assert cb2.best_value == cb.best_value
    assert cb2.should_stop == cb.should_stop


# ---------------------------------------------------------------------------
# Loss prediction
# ---------------------------------------------------------------------------

def test_loss_prediction():
    """LossPredictionCallback sets _loss_prediction after 10 epochs."""
    cb = LossPredictionCallback(max_epochs=50)
    trainer = MagicMock()
    trainer._loss_prediction = None

    # Simulate 15 epochs with decreasing loss
    for epoch in range(15):
        val_loss = 1.0 / (epoch + 1)
        cb.on_val_end(trainer=trainer, epoch=epoch, val_loss=val_loss, metrics={})

    # After epoch >= 10, _loss_prediction should be set
    assert trainer._loss_prediction is not None


# ---------------------------------------------------------------------------
# LatestModelCallback — full-state save for resume
# ---------------------------------------------------------------------------

def test_latest_model_callback_writes_full_checkpoint(tmp_path):
    """LatestModelCallback persists model + optimizer + scheduler + RNG on every epoch."""
    config = default_run_config()
    model = config.create_model()
    optimizer = config.create_optimizer(model)
    scheduler = config.create_scheduler(optimizer)

    save_path = str(tmp_path / "latest_model.pt")
    cb = LatestModelCallback(save_path, config_hash="abc")

    # Provide a minimal trainer surface the callback uses.
    trainer = MagicMock()
    trainer.model = model
    trainer.optimizer = optimizer
    trainer.scheduler = scheduler
    trainer.callbacks.callbacks = []  # no early-stopping callback

    cb.on_epoch_end(trainer=trainer, epoch=3, train_loss=0.1,
                    val_loss=0.2, metrics={"mse": 0.2})

    assert os.path.exists(save_path)
    ckpt = torch.load(save_path, map_location="cpu", weights_only=False)
    assert "model_state_dict" in ckpt
    assert "optimizer_state_dict" in ckpt
    assert "scheduler_state_dict" in ckpt
    assert "rng_states" in ckpt
    assert ckpt["epoch"] == 3
    assert ckpt["config_hash"] == "abc"
