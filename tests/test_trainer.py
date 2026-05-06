"""Tests for util.Trainer — smoke test and callback integration."""

import os
import sys
from unittest.mock import MagicMock

import pytest
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import default_run_config
from callbacks import CallbackRunner, TrainingCallback
from util import Trainer, load_data


def _build_trainer_components():
    """Helper: build a tiny model, optimizer, scheduler, criterion, and data loaders."""
    torch.manual_seed(42)
    config = default_run_config().with_overrides(
        epochs=2,
        batch_size=32,
        net_config={"nodes": 16, "layers": 2},
    )
    model = config.create_model()
    optimizer = config.create_optimizer(model)
    scheduler = config.create_scheduler(optimizer)
    criterion = config.create_criterion()

    train_ds, val_ds = load_data(n=100)
    dl_train = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True)
    dl_val = DataLoader(val_ds, batch_size=config.batch_size)

    return model, optimizer, scheduler, criterion, dl_train, dl_val


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def test_trainer_smoke():
    """Trainer completes 2 epochs with tiny model and returns a loss."""
    model, optimizer, scheduler, criterion, dl_train, dl_val = _build_trainer_components()

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        device="cpu",
    )

    val_loss = trainer.train(dl_train, dl_val, epochs=2)
    assert isinstance(val_loss, float)
    assert val_loss > 0


# ---------------------------------------------------------------------------
# Callback integration
# ---------------------------------------------------------------------------

def test_trainer_with_callbacks():
    """Trainer fires callback events correctly."""
    model, optimizer, scheduler, criterion, dl_train, dl_val = _build_trainer_components()

    events_fired = []

    class RecorderCallback(TrainingCallback):
        priority = 50

        def on_train_begin(self, trainer, **kwargs):
            events_fired.append("on_train_begin")

        def on_train_epoch_begin(self, trainer, epoch, **kwargs):
            events_fired.append(f"on_train_epoch_begin:{epoch}")

        def on_val_end(self, trainer, epoch, val_loss, metrics, **kwargs):
            events_fired.append(f"on_val_end:{epoch}")

        def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
            events_fired.append(f"on_epoch_end:{epoch}")

        def on_train_end(self, trainer, **kwargs):
            events_fired.append("on_train_end")

    callback_runner = CallbackRunner([RecorderCallback()])
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        callbacks=callback_runner,
        device="cpu",
    )

    trainer.train(dl_train, dl_val, epochs=2)

    assert "on_train_begin" in events_fired
    assert "on_train_epoch_begin:0" in events_fired
    assert "on_train_epoch_begin:1" in events_fired
    assert "on_val_end:0" in events_fired
    assert "on_val_end:1" in events_fired
    assert "on_epoch_end:0" in events_fired
    assert "on_epoch_end:1" in events_fired
    assert "on_train_end" in events_fired


# ---------------------------------------------------------------------------
# Resume / start_epoch
# ---------------------------------------------------------------------------

def test_trainer_resumes_from_start_epoch():
    """Trainer.train(start_epoch=k) skips epochs [0..k-1] and only fires
    epoch events for [k..epochs-1]."""
    model, optimizer, scheduler, criterion, dl_train, dl_val = _build_trainer_components()

    seen_epochs: list[int] = []

    class EpochRecorder(TrainingCallback):
        priority = 50

        def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
            seen_epochs.append(epoch)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, callbacks=CallbackRunner([EpochRecorder()]),
        device="cpu",
    )
    trainer.train(dl_train, dl_val, epochs=4, start_epoch=2)
    assert seen_epochs == [2, 3]


def test_trainer_no_op_when_start_epoch_at_or_past_end():
    """If start_epoch >= epochs, training is a no-op but on_train_end still fires."""
    model, optimizer, scheduler, criterion, dl_train, dl_val = _build_trainer_components()

    seen_epochs: list[int] = []
    end_fired = []

    class Recorder(TrainingCallback):
        priority = 50

        def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
            seen_epochs.append(epoch)

        def on_train_end(self, trainer, **kwargs):
            end_fired.append(True)

    trainer = Trainer(
        model=model, optimizer=optimizer, scheduler=scheduler,
        criterion=criterion, callbacks=CallbackRunner([Recorder()]),
        device="cpu",
    )
    trainer.train(dl_train, dl_val, epochs=2, start_epoch=2)
    assert seen_epochs == []
    assert end_fired == [True]
