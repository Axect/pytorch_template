---
title: Callback System
nav_order: 3
---

# Callback System

The callback system lets you extend training behavior without touching the core training loop. Instead of a monolithic training function with flags scattered throughout, the `Trainer` fires named events at specific moments, and callbacks respond to the events they care about.

---

## How Callbacks Work

The architecture has three pieces:

**`TrainingCallback`** — the base class. Subclass it and override whichever hook methods you need. All hooks default to no-ops, so you only implement what matters.

**`CallbackRunner`** — holds a list of callbacks sorted by `priority` (lower number = runs earlier). When `Trainer` fires an event, `CallbackRunner.fire()` iterates through all callbacks in priority order and calls the matching method.

**`Trainer`** — the training loop. It calls `self.callbacks.fire(event_name, ...)` at seven defined points:

```
on_train_begin
    └─ for each epoch:
        on_train_epoch_begin
            └─ for each batch:
                on_train_step_end
        on_val_begin
            └─ (validation loop)
        on_val_end
        on_epoch_end
on_train_end
```

The event flow for a single epoch looks like this:

```
Trainer.train_epoch()
    │
    ├─► fire("on_train_epoch_begin")   # set optimizer to train mode
    │
    ├─► for each batch:
    │       forward → backward → optimizer.step()
    │       fire("on_train_step_end")  # gradient monitoring
    │
Trainer.val_epoch()
    │
    ├─► fire("on_val_begin")           # set optimizer to eval mode
    │
    ├─► (validation forward pass)
    │
    └─► fire("on_val_end")             # loss prediction, pruner report, early stopping check
    │
    fire("on_epoch_end")               # overfit detection, W&B logging, checkpointing
    │
    Trainer checks stop signals        # NaN or early stopping → break
```

**Priority** determines execution order within each event. A callback with priority 5 runs before one with priority 80 when both respond to the same hook. This ordering is essential when callbacks communicate — the producer must run before the consumer.

---

## All 9 Built-in Callbacks

### 1. `NaNDetectionCallback` — priority 5

**What it does:** After each epoch, checks whether `train_loss` or `val_loss` is `NaN`. If so, sets `self.nan_detected = True`.

**When it fires:** `on_epoch_end`

**Why it matters:** NaN losses are silent killers. Without detection, training simply continues, logging `nan` to W&B, and you only discover the failure when you look at the results. With this callback, training halts immediately and the run returns `val_loss = inf`, which Optuna will correctly treat as a failed trial.

**Hook used:** After `nan_detected` is set, `Trainer.train()` checks all callbacks and breaks the epoch loop if any `NaNDetectionCallback` has `nan_detected = True`.

---

### 2. `OptimizerModeCallback` — priority 10

**What it does:** Calls `optimizer.train()` at the start of each training epoch and `optimizer.eval()` before validation.

**When it fires:** `on_train_epoch_begin`, `on_val_begin`

**Why it matters:** The `SPlus` optimizer (from `pytorch_optimizer`) and `ScheduleFree` optimizers from `torch.optim` maintain internal EMA state that must be switched between train and eval modes — otherwise validation loss is computed with gradient-contaminated weights. For standard optimizers like `AdamW`, these methods don't exist and the callback is a no-op.

**Implementation note:** The callback guards against false positives by checking both `hasattr(optimizer, 'train')` and that the optimizer is not the model itself (which also has a `.train()` method).

---

### 3. `GradientMonitorCallback` — priority 12

**What it does:** After each training batch, computes the L2 norm of all parameter gradients. Tracks the per-epoch maximum. Warns to the console if a batch's norm exceeds `warn_threshold` (default: `1e4`). Stores the epoch's maximum norm in `trainer._max_grad_norm`.

**When it fires:** `on_train_epoch_begin` (reset state), `on_train_step_end` (compute norm), `on_epoch_end` (store result)

**Why it matters:** Gradient explosion is hard to diagnose from loss curves alone — loss may not go NaN immediately, or may oscillate suspiciously. Monitoring gradient norms gives you an early warning and a continuous signal that W&B can visualize. It also provides data for diagnosing whether a learning rate is too high.

**Configuration:**
```python
GradientMonitorCallback(warn_threshold=1e4)  # default
GradientMonitorCallback(warn_threshold=1e2)  # stricter, for sensitive architectures
```

---

### 4. `LossPredictionCallback` — priority 70

**What it does:** Accumulates validation losses epoch by epoch. After epoch 10, fits an exponential curve `L(t) = A * exp(K * t)` to the loss history and extrapolates to `max_epochs`. Stores the predicted final loss (as `-log10`) in `trainer._loss_prediction`.

**When it fires:** `on_val_end`

**Why it matters:** During HPO, you often cannot afford to run every trial to completion. The loss prediction gives a forward-looking signal: if the predicted final loss is already worse than the current best, the trial is likely a waste of compute even before the pruner acts. The prediction is logged to W&B as `predicted_final_loss`.

**Note:** The prediction is `None` for the first 10 epochs (insufficient data for a reliable fit). If the exponential fit fails or produces a non-finite result, the callback falls back to `-log10(last_val_loss)`.

---

### 5. `OverfitDetectionCallback` — priority 75

**What it does:** After a configurable warmup period, examines the last `window_size` epochs of both training and validation loss. If training loss is **monotonically decreasing** while validation loss is **monotonically increasing** across the entire window, overfitting is flagged. The gap ratio (`val_loss / train_loss`) is stored in `trainer._overfit_gap_ratio` and a warning is printed.

**When it fires:** `on_epoch_end`

**Why it matters:** Standard early stopping reacts to validation loss rising above its best — but only after the damage is done. Overfit detection catches the **trend** before it peaks, giving you earlier signal. The gap ratio is also a useful normalization: a gap ratio of 2.0 means validation loss is twice the training loss, which is interpretable regardless of the loss scale.

**Configuration:**
```python
OverfitDetectionCallback(warmup_epochs=5, window_size=5)  # defaults
```

Setting `window_size=1` makes detection hair-trigger (any single-epoch divergence fires). A larger window requires sustained divergence and reduces false positives.

---

### 6. `WandbLoggingCallback` — priority 80

**What it does:** At the end of each epoch, logs a dict to W&B containing:
- `train_loss`, `val_loss`, `lr` (always)
- Any extra `metrics` passed from the trainer
- `max_grad_norm` (if `GradientMonitorCallback` has run)
- `overfit_gap_ratio` (if `OverfitDetectionCallback` fired)
- `predicted_final_loss` (if `LossPredictionCallback` has a prediction)

Also prints a compact summary to the console every 10 epochs.

**When it fires:** `on_epoch_end`

**Why it matters:** Centralizes all W&B logging in one place. Higher-priority callbacks compute their results (gradient norms, overfit ratio, loss predictions) and store them as trainer attributes. `WandbLoggingCallback` reads those attributes and bundles everything into a single `wandb.log()` call — one log event per epoch, consistent keys, no duplicates.

---

### 7. `PrunerCallback` — priority 85

**What it does:** After each validation, reports the current `val_loss` to the Optuna pruner. If the pruner signals `should_prune()`, raises `optuna.TrialPruned`.

**When it fires:** `on_val_end`

**Why it matters:** Optuna pruning cuts unpromising trials early during HPO, saving significant compute. This callback is only added to the callback list when a `pruner` and `trial` object are present (i.e., during HPO, not during final training runs).

---

### 8. `EarlyStoppingCallback` — priority 90

**What it does:** Tracks the monitored metric (default: `val_loss`) and counts consecutive epochs without meaningful improvement (threshold: `min_delta`). When the counter reaches `patience`, sets `self.should_stop = True`.

**When it fires:** `on_val_end`

**Why it matters:** Prevents wasted compute when a model has clearly converged. The `min_delta` threshold avoids stopping on noise — an improvement of `0.001%` does not reset the counter if `min_delta = 0.0001` (which is 0.01%).

**Configuration** (via `early_stopping_config` in `RunConfig`):
```yaml
early_stopping_config:
  enabled: true
  patience: 20
  mode: min
  min_delta: 0.0001
```

Supports `state_dict()` / `load_state_dict()` for checkpoint-resume compatibility.

---

### 9. `CheckpointCallback` — priority 95

**What it does:** Delegates to a `CheckpointManager` to save the model, optimizer, scheduler, and early stopping state. Saves periodically (every `save_every_n_epochs`) and keeps the best-performing checkpoint by `monitor` metric. Maintains a sliding window of the last `keep_last_k` periodic checkpoints to bound disk usage.

**When it fires:** `on_epoch_end`

**Why it matters:** Running at priority 95, this is the last callback to execute each epoch — after all diagnostics, logging, and stop signals have been processed. This guarantees that a checkpoint always reflects a fully-evaluated epoch state. It also captures the `EarlyStoppingCallback` state, enabling exact resume from mid-training interruptions.

---

## Writing Custom Callbacks

A custom callback needs only three things: a `priority`, and overrides for whichever hooks matter. Here is a complete example that tracks the best validation loss and prints a summary at the end of training:

```python
from callbacks import TrainingCallback

class BestLossTrackerCallback(TrainingCallback):
    priority = 60  # runs before W&B logging (80) so the value is available

    def __init__(self):
        self.best_val_loss = float("inf")
        self.best_epoch = -1

    def on_val_end(self, trainer, epoch, val_loss, metrics, **kwargs):
        if val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            self.best_epoch = epoch
            # Store on trainer so WandbLoggingCallback can pick it up
            trainer._best_val_loss = self.best_val_loss

    def on_train_end(self, trainer, **kwargs):
        from tqdm import tqdm
        tqdm.write(
            f"Training complete. Best val_loss: {self.best_val_loss:.4e} "
            f"at epoch {self.best_epoch}"
        )
```

To add it to a training run, append it to the callbacks list before constructing `CallbackRunner`:

```python
callbacks_list = [
    OptimizerModeCallback(),
    NaNDetectionCallback(),
    GradientMonitorCallback(),
    BestLossTrackerCallback(),   # <-- add here
    LossPredictionCallback(run_config.epochs),
    OverfitDetectionCallback(),
    WandbLoggingCallback(),
]
callback_runner = CallbackRunner(callbacks_list)
```

The `CallbackRunner` sorts by priority on construction, so insertion order does not matter.

---

## Callback Communication

Callbacks communicate by writing to and reading from `trainer` attributes. This is an intentional design choice: callbacks are decoupled from each other (they do not hold references to each other), but they share a common carrier — the `Trainer` instance.

The convention is that producer callbacks write to `trainer._<name>` and consumer callbacks check for those attributes with `hasattr`:

| Attribute | Written by | Read by | Semantics |
|---|---|---|---|
| `trainer._max_grad_norm` | `GradientMonitorCallback` (priority 12) | `WandbLoggingCallback` (priority 80) | Max gradient L2 norm for the current epoch |
| `trainer._overfit_gap_ratio` | `OverfitDetectionCallback` (priority 75) | `WandbLoggingCallback` (priority 80) | `val_loss / train_loss` when divergence is detected; `None` otherwise |
| `trainer._loss_prediction` | `LossPredictionCallback` (priority 70) | `WandbLoggingCallback` (priority 80) | Predicted final loss as `-log10(L)` |

Priority ordering enforces the producer-before-consumer contract. `GradientMonitorCallback` at priority 12 always finishes its `on_epoch_end` before `WandbLoggingCallback` at priority 80 starts its `on_epoch_end`. The `hasattr` guard on the consumer side means that if the producer callback is not present (e.g., in a minimal test setup), the consumer silently skips that metric rather than raising `AttributeError`.

This pattern also makes it easy to add new metrics: write to a `trainer._my_metric` attribute from a high-priority callback, then read it in `WandbLoggingCallback` — no changes to `WandbLoggingCallback`'s constructor or the `CallbackRunner` setup are required.
