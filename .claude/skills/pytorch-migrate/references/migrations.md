# Migration Reference

Detailed steps for each migration. Apply in order, skip migrations that are already present.

---

## M1: PFL Pruner (v1 → v2)

**Detect:** `grep -c "class PFLPruner" pruner.py` returns 0

### pruner.py

Create the file from scratch:

```python
import numpy as np
from typing import List, Dict, Tuple, Optional
import bisect
from dataclasses import dataclass, field
from util import predict_final_loss


@dataclass
class Trial:
    """Trial class to hold intermediate state."""

    trial_id: int
    current_epoch: int = 0
    seed_values: Dict[int, List[float]] = field(default_factory=dict)

    def add_value(self, seed: int, value: float) -> None:
        """Add a new intermediate value for a given seed."""
        if seed not in self.seed_values:
            self.seed_values[seed] = []
        self.seed_values[seed].append(value)
        self.current_epoch = len(self.seed_values[seed])


class BasePruner:
    """
    Pruner base class with Optuna-like interface.
    """

    def __init__(self):
        self._trials: Dict[int, Trial] = {}
        self._current_trial: Optional[Trial] = None

    def register_trial(self, trial_id: int) -> None:
        """Register a new trial."""
        self._trials[trial_id] = Trial(trial_id=trial_id)

    def complete_trial(self, trial_id: int) -> None:
        """Mark a trial as finished and clean up."""
        if trial_id in self._trials:
            if self._current_trial and self._current_trial.trial_id == trial_id:
                self._current_trial = None
            del self._trials[trial_id]

    def report(self, trial_id: int, seed: int, epoch: int, value: float) -> None:
        if trial_id not in self._trials:
            self.register_trial(trial_id)
        trial = self._trials[trial_id]
        trial.add_value(seed, value)
        self._current_trial = trial

    def should_prune(self) -> bool:
        if not self._current_trial:
            return False
        return self._should_prune_trial(self._current_trial)

    def _should_prune_trial(self, trial: Trial) -> bool:
        raise NotImplementedError


class PFLPruner(BasePruner):
    """Predicted Final Loss (PFL) based pruner with Optuna-like interface."""

    def __init__(
        self,
        n_startup_trials: int = 10,
        n_warmup_epochs: int = 10,
        top_k: int = 10,
        target_epoch: int = 50,
    ):
        super().__init__()
        self.n_startup_trials = n_startup_trials
        self.n_warmup_epochs = n_warmup_epochs
        self.top_k = top_k
        self.target_epoch = target_epoch
        self.top_pairs: List[Tuple[float, float]] = []
        self.completed_trials = 0

    def complete_trial(self, trial_id: int) -> None:
        if trial_id in self._trials:
            self.completed_trials += 1
            self._check_and_insert(self._trials[trial_id])
            super().complete_trial(trial_id)

    def _check_and_insert(self, trial: Trial) -> None:
        train_loss, pfl = self._compute_trial_metrics(trial)
        if self._should_insert_pair(train_loss):
            self._insert_pair(train_loss, pfl)

    def _compute_trial_metrics(self, trial: Trial) -> Tuple[float, float]:
        if not trial.seed_values:
            return float("inf"), -float("inf")
        avg_train_loss = 0.0
        avg_pfl = 0.0
        n_seeds = len(trial.seed_values)
        for loss_vec in trial.seed_values.values():
            if loss_vec:
                avg_train_loss += loss_vec[-1]
                avg_pfl += self._predict_final_loss(loss_vec)
        avg_train_loss /= n_seeds
        avg_pfl /= n_seeds
        return avg_train_loss, avg_pfl

    def _predict_final_loss(self, losses: List[float]) -> float:
        if len(losses) < 2:
            return float("inf")
        try:
            return predict_final_loss(losses, self.target_epoch)
        except Exception:
            return float("inf")

    def _should_insert_pair(self, train_loss: float) -> bool:
        if len(self.top_pairs) < self.top_k:
            return True
        return train_loss < self.top_pairs[-1][0]

    def _insert_pair(self, train_loss: float, pfl: float) -> None:
        pair = (train_loss, pfl)
        idx = bisect.bisect_left(self.top_pairs, pair)
        if len(self.top_pairs) < self.top_k:
            self.top_pairs.insert(idx, pair)
        elif idx < self.top_k:
            self.top_pairs.insert(idx, pair)
            self.top_pairs.pop()

    def _should_prune_trial(self, trial: Trial) -> bool:
        for losses in trial.seed_values.values():
            if not losses or not np.isfinite(losses[-1]):
                return True
        if (
            self.completed_trials < self.n_startup_trials
            or trial.current_epoch <= self.n_warmup_epochs
        ):
            return False
        _, curr_pfl = self._compute_trial_metrics(trial)
        if self.top_pairs:
            worst_pred = max(pair[1] for pair in self.top_pairs)
            return curr_pfl > worst_pred
        return False
```

### callbacks.py

Add `PrunerCallback` class (priority=85) after `WandbLoggingCallback`:

```python
class PrunerCallback(TrainingCallback):
    """Reports metrics to Optuna pruner and raises TrialPruned if needed."""
    priority = 85

    def __init__(self, pruner, trial, seed):
        self.pruner = pruner
        self.trial = trial
        self.seed = seed

    def on_val_end(self, trainer, epoch, val_loss, metrics, **kwargs):
        import optuna
        if self.pruner is not None and self.trial is not None:
            self.pruner.report(
                trial_id=self.trial.number,
                seed=self.seed,
                epoch=epoch,
                value=val_loss,
            )
            if self.pruner.should_prune():
                raise optuna.TrialPruned()
```

### util.py

1. Add `predict_final_loss()` function (before the `Trainer` class):

```python
def predict_final_loss(losses, max_epochs):
    """Predict the final validation loss using shifted exponential decay.

    Fits L(t) = a * exp(-b * t) + c to EMA-smoothed losses.
    Returns the predicted raw loss value at max_epochs.
    Works with positive and negative losses.
    """
    n = len(losses)
    if n < 10:
        return float(losses[-1])

    y = np.array(losses, dtype=np.float64)

    # EMA smoothing — adaptive span
    span = min(n // 3, 20)
    alpha = 2.0 / (span + 1)
    ema = np.empty(n)
    ema[0] = y[0]
    for i in range(1, n):
        ema[i] = alpha * y[i] + (1 - alpha) * ema[i - 1]

    # Three equally-spaced anchor points from smoothed curve
    i1, i2, i3 = n // 3, 2 * n // 3, n - 1
    y1, y2, y3 = ema[i1], ema[i2], ema[i3]

    d12 = y1 - y2
    d23 = y2 - y3

    if abs(d12) < 1e-15 or abs(d23) < 1e-15:
        return float(ema[-1])

    r = d23 / d12

    if r <= 0 or r >= 1:
        window = min(10, n - 1)
        recent_rate = (ema[-1] - ema[-1 - window]) / window
        remaining = max(max_epochs - n, 0)
        predicted = ema[-1] + recent_rate * remaining * 0.5
        return float(predicted) if np.isfinite(predicted) else float(ema[-1])

    d = float(i2 - i1)
    b = -np.log(r) / d
    t1 = float(i1)
    t2 = float(i2)

    denom = np.exp(-b * t1) - np.exp(-b * t2)
    if abs(denom) < 1e-30:
        return float(ema[-1])

    a = d12 / denom
    c = y1 - a * np.exp(-b * t1)

    predicted = a * np.exp(-b * max_epochs) + c

    if np.isfinite(predicted):
        return float(predicted)

    return float(ema[-1])
```

2. Add `LossPredictionCallback` import and `PrunerCallback` import.

3. Wire into `run()`: register trial, add `PrunerCallback` to callbacks, and complete trial in `finally`:

```python
def run(run_config, dl_train, dl_val, group_name=None, trial=None, pruner=None):
    ...
    # Register trial at the beginning if pruner exists
    if pruner is not None and trial is not None and hasattr(pruner, "register_trial"):
        pruner.register_trial(trial.number)

    ...
    callbacks_list = [
        OptimizerModeCallback(),
        LossPredictionCallback(run_config.epochs),  # NEW
        WandbLoggingCallback(),
    ]
    if pruner is not None and trial is not None:
        callbacks_list.append(PrunerCallback(pruner, trial, seed))  # NEW
    ...
    finally:
        # Call trial_finished only once after all seeds are done
        if (
            pruner is not None
            and trial is not None
            and hasattr(pruner, "complete_trial")
        ):
            pruner.complete_trial(trial.number)
```

### config.py

1. Add `pruner` field to `OptimizeConfig` dataclass:

```python
@dataclass
class OptimizeConfig:
    study_name: str
    trials: int
    seed: int
    metric: str
    direction: str
    sampler: dict = field(default_factory=dict)
    pruner: dict = field(default_factory=dict)    # NEW
    search_space: dict = field(default_factory=dict)
```

2. Add `create_pruner()` method to `OptimizeConfig` (after `_create_sampler()`):

```python
def create_pruner(self):
    if not self.pruner:
        return None
    module_name, class_name = self.pruner["name"].rsplit(".", 1)
    module = importlib.import_module(module_name)
    pruner_class = getattr(module, class_name)
    pruner_kwargs = self.pruner.get("kwargs", {})
    return pruner_class(**pruner_kwargs)
```

### configs/optimize_template.yaml

Add pruner section:

```yaml
pruner:
  name: pruner.PFLPruner
  kwargs:
    n_startup_trials: 10
    n_warmup_epochs: 10
    top_k: 10
    target_epoch: 50
```

### cli.py / main.py

In the `train` command's HPO branch, create the pruner from opt_config and pass it to `run()`:

```python
opt_config = OptimizeConfig.from_yaml(optimize_config)
pruner = opt_config.create_pruner()   # NEW

def objective(trial, base_config, opt_config, dl_train, dl_val):
    ...
    return run(
        trial_config, dl_train, dl_val, group_name, trial=trial, pruner=pruner  # NEW
    )
```

---

## M2: NaN Detection + Checkpoint (v2 → v3)

**Detect:** `grep -c "class NaNDetectionCallback" callbacks.py` returns 0

### callbacks.py

1. Add `NaNDetectionCallback` (priority=5) early in the file, before `EarlyStoppingCallback`:

```python
class NaNDetectionCallback(TrainingCallback):
    """Detects NaN losses and signals training to stop."""
    priority = 5  # Run very early

    def __init__(self):
        self.nan_detected = False

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        if math.isnan(train_loss) or math.isnan(val_loss):
            from tqdm import tqdm
            tqdm.write("Early stopping due to NaN loss")
            self.nan_detected = True
```

Add `import math` at the top if not already present.

2. Add `CheckpointCallback` (priority=95) after `EarlyStoppingCallback`:

```python
class CheckpointCallback(TrainingCallback):
    """Saves periodic and best-model checkpoints during training."""
    priority = 95  # After early stopping check

    def __init__(self, checkpoint_manager, config_hash: str = ""):
        self.checkpoint_manager = checkpoint_manager
        self.config_hash = config_hash

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        early_stopping_state = None
        for cb in trainer.callbacks.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                early_stopping_state = cb.state_dict()
                break
        self.checkpoint_manager.maybe_save(
            epoch, trainer.model, trainer.optimizer, trainer.scheduler,
            val_loss, metrics, early_stopping_state, self.config_hash,
        )
```

### config.py

1. Add `EarlyStoppingConfig` dataclass (before `RunConfig`):

```python
@dataclass
class EarlyStoppingConfig:
    enabled: bool = False
    patience: int = 10
    mode: str = "min"  # "min" or "max"
    min_delta: float = 0.0001
```

2. Add `CheckpointConfig` dataclass (after `EarlyStoppingConfig`):

```python
@dataclass
class CheckpointConfig:
    enabled: bool = False
    save_every_n_epochs: int = 10
    keep_last_k: int = 3
    save_best: bool = True
    monitor: str = "val_loss"
    mode: str = "min"
```

3. Add both fields to `RunConfig`:

```python
@dataclass
class RunConfig:
    ...
    early_stopping_config: EarlyStoppingConfig = field(
        default_factory=lambda: EarlyStoppingConfig()
    )
    ...
    checkpoint_config: CheckpointConfig = field(
        default_factory=lambda: CheckpointConfig()
    )
```

4. Add dict-to-dataclass conversion in `RunConfig.__post_init__`:

```python
def __post_init__(self):
    if isinstance(self.early_stopping_config, dict):
        self.early_stopping_config = EarlyStoppingConfig(
            **self.early_stopping_config
        )
    if isinstance(self.checkpoint_config, dict):
        self.checkpoint_config = CheckpointConfig(
            **self.checkpoint_config
        )
    ...
```

### checkpoint.py

Create the file from scratch:

```python
import os
import json
import glob
import random
import torch
import numpy as np
from typing import Any, Optional


class CheckpointManager:
    """Manages saving and loading of training checkpoints."""

    CHECKPOINT_VERSION = 1

    def __init__(self, run_dir: str, save_every_n: int = 10,
                 keep_last_k: int = 3, save_best: bool = True,
                 monitor: str = "val_loss", mode: str = "min"):
        self.run_dir = run_dir
        self.save_every_n = save_every_n
        self.keep_last_k = keep_last_k
        self.save_best = save_best
        self.monitor = monitor
        self.mode = mode
        self.best_value: Optional[float] = None

    def _is_better(self, current: float) -> bool:
        if self.best_value is None:
            return True
        if self.mode == "min":
            return current < self.best_value
        return current > self.best_value

    def _capture_rng_states(self) -> dict:
        rng = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch_cpu": torch.random.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng["torch_cuda"] = torch.cuda.get_rng_state_all()
        return rng

    def _restore_rng_states(self, rng: dict) -> None:
        random.setstate(rng["python"])
        np.random.set_state(rng["numpy"])
        torch.random.set_rng_state(rng["torch_cpu"])
        if "torch_cuda" in rng and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng["torch_cuda"])

    def save_checkpoint(self, path: str, model, optimizer, scheduler,
                        epoch: int, val_loss: float, metrics: dict,
                        early_stopping_state: Optional[dict] = None,
                        config_hash: str = "") -> None:
        """Save a complete training checkpoint."""
        checkpoint = {
            "version": self.CHECKPOINT_VERSION,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "metrics": metrics,
            "config_hash": config_hash,
            "rng_states": self._capture_rng_states(),
        }
        if hasattr(scheduler, 'state_dict'):
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()
        if early_stopping_state:
            checkpoint["early_stopping_state"] = early_stopping_state
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, model, optimizer, scheduler,
                        device: str = "cpu", config_hash: str = "") -> dict:
        """Load a checkpoint and restore all state."""
        checkpoint = torch.load(path, map_location=device, weights_only=False)
        saved_hash = checkpoint.get("config_hash", "")
        if config_hash and saved_hash and saved_hash != config_hash:
            raise ValueError(
                f"Config hash mismatch: checkpoint={saved_hash[:12]}... "
                f"vs current={config_hash[:12]}... "
                "The config has changed since this checkpoint was saved."
            )
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if "scheduler_state_dict" in checkpoint and hasattr(scheduler, 'load_state_dict'):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if "rng_states" in checkpoint:
            self._restore_rng_states(checkpoint["rng_states"])
        return checkpoint

    def maybe_save(self, epoch: int, model, optimizer, scheduler,
                   val_loss: float, metrics: dict,
                   early_stopping_state: Optional[dict] = None,
                   config_hash: str = "") -> None:
        """Conditionally save checkpoint based on epoch and best-model policy."""
        if self.save_every_n > 0 and (epoch + 1) % self.save_every_n == 0:
            path = os.path.join(self.run_dir, f"checkpoint_epoch_{epoch}.pt")
            self.save_checkpoint(path, model, optimizer, scheduler,
                               epoch, val_loss, metrics, early_stopping_state, config_hash)
            self._cleanup_old_checkpoints()
        if self.save_best and self._is_better(val_loss):
            self.best_value = val_loss
            path = os.path.join(self.run_dir, "best.pt")
            self.save_checkpoint(path, model, optimizer, scheduler,
                               epoch, val_loss, metrics, early_stopping_state, config_hash)
        path = os.path.join(self.run_dir, "latest.pt")
        self.save_checkpoint(path, model, optimizer, scheduler,
                           epoch, val_loss, metrics, early_stopping_state, config_hash)

    def _cleanup_old_checkpoints(self) -> None:
        pattern = os.path.join(self.run_dir, "checkpoint_epoch_*.pt")
        checkpoints = sorted(glob.glob(pattern))
        while len(checkpoints) > self.keep_last_k:
            os.remove(checkpoints.pop(0))

    def find_latest_checkpoint(self) -> Optional[str]:
        latest = os.path.join(self.run_dir, "latest.pt")
        if os.path.exists(latest):
            return latest
        return None


class SeedManifest:
    """Tracks completed seeds for multi-seed resume support."""

    def __init__(self, group_path: str):
        self.path = os.path.join(group_path, "seed_manifest.json")
        self.completed_seeds: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.path):
            with open(self.path, "r") as f:
                data = json.load(f)
                self.completed_seeds = data.get("completed_seeds", {})

    def save(self) -> None:
        with open(self.path, "w") as f:
            json.dump({"completed_seeds": self.completed_seeds}, f, indent=2)

    def mark_complete(self, seed: int, val_loss: float,
                      wandb_run_id: str = "", metrics: dict | None = None) -> None:
        self.completed_seeds[str(seed)] = {
            "val_loss": val_loss,
            "wandb_run_id": wandb_run_id,
            "metrics": metrics or {},
        }
        self.save()

    def is_complete(self, seed: int) -> bool:
        return str(seed) in self.completed_seeds

    def get_total_loss(self) -> float:
        return sum(s["val_loss"] for s in self.completed_seeds.values())

    def get_complete_count(self) -> int:
        return len(self.completed_seeds)
```

### util.py

1. Add imports:

```python
from checkpoint import CheckpointManager, SeedManifest
from callbacks import (..., NaNDetectionCallback, CheckpointCallback)
```

2. Add NaN check in `Trainer.train()` (in the per-epoch callback signal check):

```python
for cb in self.callbacks.callbacks:
    if isinstance(cb, NaNDetectionCallback) and cb.nan_detected:
        val_loss = math.inf
        break_flag = True
        break
    if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
        tqdm.write(f"Early stopping triggered at epoch {epoch}")
        break_flag = True
        break
```

3. Wire `SeedManifest` and `CheckpointManager` into `run()`:

```python
# Create seed manifest for multi-seed resume support
manifest = SeedManifest(group_path)

...
for seed in seeds:
    # Skip already-completed seeds
    if manifest.is_complete(seed):
        tqdm.write(f"Seed {seed} already complete, skipping")
        continue
    ...
    # Create CheckpointManager if enabled
    if run_config.checkpoint_config.enabled:
        from provenance import compute_config_hash
        config_hash = compute_config_hash(run_config)
        ckpt_manager = CheckpointManager(
            run_dir=run_path,
            save_every_n=run_config.checkpoint_config.save_every_n_epochs,
            keep_last_k=run_config.checkpoint_config.keep_last_k,
            save_best=run_config.checkpoint_config.save_best,
            monitor=run_config.checkpoint_config.monitor,
            mode=run_config.checkpoint_config.mode,
        )
        callbacks_list.append(CheckpointCallback(ckpt_manager, config_hash))
    ...
    # Mark seed as complete
    manifest.mark_complete(seed, val_loss)

    # Early stopping if loss becomes inf
    if math.isinf(val_loss):
        break

return manifest.get_total_loss() / (complete_count if complete_count > 0 else 1)
```

### configs/run_template.yaml

Add `early_stopping_config` and `checkpoint_config` sections:

```yaml
early_stopping_config:
  enabled: false
  patience: 10
  mode: min # 'min' or 'max'
  min_delta: 0.0001 # Percent
checkpoint_config:
  enabled: false
  save_every_n_epochs: 10
  keep_last_k: 3
  save_best: true
  monitor: val_loss
  mode: min
```

---

## M3: Modular CLI (v3 → v4)

**Detect:** `test -f cli.py` fails (cli.py does not exist)

### cli.py

Create `cli.py` using the typer framework. The file provides a structured CLI with subcommands that supersede the argparse-based `main.py`.

**Dependencies:** `typer`, `rich`, `beaupy` (add to project requirements if not present).

Key structure:

```python
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="PyTorch Template CLI")
console = Console()


@app.command()
def train(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
    device: str = typer.Option(None, help="Device override (e.g. 'cuda:0' or 'cpu')"),
    optimize_config: str = typer.Option(None, help="Path to optimization config for HPO"),
):
    """Train a model using the given run configuration."""
    ...


@app.command()
def validate(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
):
    """Validate a run configuration for structural and runtime correctness."""
    ...


@app.command()
def preview(
    run_config: str = typer.Argument(..., help="Path to the YAML config file"),
):
    """Preview the model, optimizer, scheduler, and criterion from a config."""
    ...


@app.command()
def analyze(
    project: str = typer.Option(None, help="Project name (non-interactive)"),
    group: str = typer.Option(None, help="Group name (non-interactive)"),
    seed: str = typer.Option(None, help="Seed (non-interactive)"),
    device: str = typer.Option("cpu", help="Device for analysis"),
):
    """Analyze a trained model — interactive or non-interactive."""
    ...


if __name__ == "__main__":
    app()
```

Usage:
```bash
python cli.py train configs/run_template.yaml
python cli.py train configs/run_template.yaml --optimize-config configs/optimize_template.yaml
python cli.py validate configs/run_template.yaml
python cli.py preview configs/run_template.yaml
python cli.py analyze --project MyProject --group exp1 --seed 42
```

**Recommendation:** Copy the full implementation from the upstream template's `cli.py` rather than writing it from scratch — the `train` command's HPO branch logic and the `analyze` command's interactive fallback are non-trivial.

### main.py

`main.py` remains as a **legacy argparse entry point**. It is not replaced — existing scripts that call `python main.py --run_config ...` continue to work. No changes are required to `main.py` during this migration.

---

## M4: Data Decoupling (v4 → v5)

**Detect:** `grep -c "data: str" config.py` returns 0

### config.py

1. Add `data` field to `RunConfig` dataclass (after `criterion_config`):

```python
data: str = "util.load_data"
```

2. Add format validation in `__post_init__` (after the existing `"."` checks for net/optimizer/scheduler):

```python
if self.data and "." not in self.data:
    raise ValueError(
        f"data must be in module.function format (contain at least one '.'), "
        f"got '{self.data}'"
    )
```

3. Add `("data", self.data)` to the import path check list in `validate_for_execution()`.

4. Add `load_data()` method (after `create_criterion()`):

```python
def load_data(self):
    """Load data using the configured data module path."""
    module_name, func_name = self.data.rsplit(".", 1)
    module = importlib.import_module(module_name)
    load_fn = getattr(module, func_name)
    return load_fn()
```

5. Add `validate_semantics()` method (after `validate_for_execution()`):

```python
def validate_semantics(self) -> list[str]:
    """Tier 3 — semantic validation: check logical relationships."""
    issues = []
    lr = self.optimizer_config.get("lr")
    if lr is not None and lr <= 0:
        issues.append(f"optimizer_config.lr must be positive, got {lr}")
    if len(self.seeds) != len(set(self.seeds)):
        issues.append(f"seeds contains duplicates: {self.seeds}")
    scheduler_class = self.scheduler.rsplit(".", 1)[-1]
    if scheduler_class in ("ExpHyperbolicLRScheduler", "HyperbolicLRScheduler"):
        total_steps = self.scheduler_config.get("total_steps")
        upper_bound = self.scheduler_config.get("upper_bound")
        if total_steps is not None and upper_bound is not None:
            if upper_bound < total_steps:
                issues.append(
                    f"scheduler upper_bound ({upper_bound}) must be >= "
                    f"total_steps ({total_steps}) for {scheduler_class}"
                )
    if scheduler_class == "CosineAnnealingLR":
        t_max = self.scheduler_config.get("T_max")
        if t_max is not None and t_max != self.epochs:
            issues.append(f"CosineAnnealingLR T_max ({t_max}) != epochs ({self.epochs})")
    if self.early_stopping_config.enabled:
        if self.early_stopping_config.patience >= self.epochs:
            issues.append(
                f"early_stopping patience ({self.early_stopping_config.patience}) >= "
                f"epochs ({self.epochs}) — early stopping will never trigger"
            )
    return issues
```

### cli.py / main.py

Replace all `from util import load_data` + `load_data()` calls with `config.load_data()`:

```python
# Before
from util import load_data, run
ds_train, ds_val = load_data()

# After
from util import run
ds_train, ds_val = base_config.load_data()
```

In the `analyze` command, replace `load_data()` with `config.load_data()`.

### YAML configs

Add `data` field to all existing run configs. Scan and patch:

```bash
# For each YAML config that has criterion_config but not data:
for f in $(grep -rl "criterion_config" configs/ --include="*.yaml"); do
    if ! grep -q "^data:" "$f"; then
        sed -i '/^criterion_config:/a data: util.load_data' "$f"
    fi
done
```

**Note:** If the project has recipe-specific data loaders (e.g., `recipes/regression/data.py`), recipe configs should use `data: recipes.regression.data.load_data` instead.

---

## M5: Diagnostics + Preflight + HPO Report (v5 → v6)

**Detect:** `grep -c "class GradientMonitorCallback" callbacks.py` returns 0

### callbacks.py

Add two new callback classes and update WandbLoggingCallback.

1. Add `GradientMonitorCallback` (priority=12):

```python
class GradientMonitorCallback(TrainingCallback):
    """Monitors gradient norms to detect exploding gradients."""
    priority = 12

    def __init__(self, warn_threshold: float = 1e4):
        self.warn_threshold = warn_threshold
        self._step_grad_norms: list[float] = []
        self.epoch_max_grad_norms: list[float] = []
        self._current_epoch_max = 0.0

    def on_train_epoch_begin(self, trainer, epoch, **kwargs):
        self._step_grad_norms = []
        self._current_epoch_max = 0.0

    def on_train_step_end(self, trainer, batch_idx, loss, **kwargs):
        total_norm = 0.0
        for param in trainer.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5
        self._step_grad_norms.append(total_norm)
        self._current_epoch_max = max(self._current_epoch_max, total_norm)
        if total_norm > self.warn_threshold:
            from tqdm import tqdm
            tqdm.write(
                f"[GradientMonitor] grad norm {total_norm:.2e} "
                f"exceeds threshold {self.warn_threshold:.2e} at batch {batch_idx}"
            )

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        self.epoch_max_grad_norms.append(self._current_epoch_max)
        trainer._max_grad_norm = self._current_epoch_max
```

2. Add `OverfitDetectionCallback` (priority=75):

```python
class OverfitDetectionCallback(TrainingCallback):
    """Detects overfitting by monitoring train/val loss divergence."""
    priority = 75

    def __init__(self, warmup_epochs: int = 5, window_size: int = 5):
        self.warmup_epochs = warmup_epochs
        self.window_size = window_size
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.overfit_detected = False

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if epoch < self.warmup_epochs or len(self.train_losses) < self.window_size:
            trainer._overfit_gap_ratio = None
            return
        recent_train = self.train_losses[-self.window_size:]
        recent_val = self.val_losses[-self.window_size:]
        train_decreasing = all(
            recent_train[i] >= recent_train[i + 1] for i in range(len(recent_train) - 1)
        )
        val_increasing = all(
            recent_val[i] <= recent_val[i + 1] for i in range(len(recent_val) - 1)
        )
        if train_decreasing and val_increasing:
            self.overfit_detected = True
            gap_ratio = recent_val[-1] / recent_train[-1] if recent_train[-1] > 0 else float("inf")
            from tqdm import tqdm
            tqdm.write(
                f"[OverfitDetection] epoch {epoch}: "
                f"train_loss decreasing, val_loss increasing (gap ratio: {gap_ratio:.2f})"
            )
            trainer._overfit_gap_ratio = gap_ratio
        else:
            trainer._overfit_gap_ratio = None
```

3. Update `WandbLoggingCallback.on_epoch_end` — add after `log_dict.update(metrics)`:

```python
if hasattr(trainer, '_max_grad_norm') and trainer._max_grad_norm is not None:
    log_dict["max_grad_norm"] = trainer._max_grad_norm
if hasattr(trainer, '_overfit_gap_ratio') and trainer._overfit_gap_ratio is not None:
    log_dict["overfit_gap_ratio"] = trainer._overfit_gap_ratio
```

### util.py

1. Add imports:
```python
from callbacks import (..., GradientMonitorCallback, OverfitDetectionCallback)
```

2. Add to Trainer `__init__`:
```python
self._max_grad_norm: float | None = None
self._overfit_gap_ratio: float | None = None
```

3. Add to callbacks list in `run()`:
```python
callbacks_list = [
    OptimizerModeCallback(),
    NaNDetectionCallback(),
    GradientMonitorCallback(),       # NEW
    LossPredictionCallback(run_config.epochs),
    OverfitDetectionCallback(),      # NEW
    WandbLoggingCallback(),
]
```

### cli.py

Add two new commands: `preflight` and `hpo-report`.

These are substantial additions (~120 lines each). Read the current template's `cli.py` for the exact implementation. The key signatures are:

```python
@app.command()
def preflight(
    run_config: str = typer.Argument(...),
    device: str = typer.Option(None),
    json_output: bool = typer.Option(False, "--json"),
): ...

@app.command(name="hpo-report")
def hpo_report(
    db: str = typer.Option(None),
    study_name: str = typer.Option(None),
    opt_config: str = typer.Option(None),
    top_k: int = typer.Option(5),
    json_output: bool = typer.Option(False, "--json"),
): ...
```

**Recommendation:** For these large additions, it is safest to copy the implementations from the upstream template's `cli.py` rather than writing them from scratch.

---

## M6: Dual Logging + TUI Monitor + Provenance (v6 → current)

**Detect:** `grep -c "class CSVLoggingCallback" callbacks.py` returns 0

### config.py

Add `logging` field to `RunConfig` and validate it in `__post_init__`:

```python
@dataclass
class RunConfig:
    ...
    logging: str = "wandb"  # "wandb" or "tui"
```

In `__post_init__`, add validation after the existing format checks:

```python
if self.logging not in ("wandb", "tui"):
    raise ValueError(
        f"logging must be 'wandb' or 'tui', got '{self.logging}'"
    )
```

### callbacks.py

1. Add `CSVLoggingCallback` (priority=81) — always active, handles dynamic columns:

```python
class CSVLoggingCallback(TrainingCallback):
    """Logs metrics to a CSV file every epoch (always active).

    Handles dynamic columns: if new metrics appear mid-training,
    the CSV is rewritten with the expanded header and all prior rows.
    """
    priority = 81

    def __init__(self, csv_path: str):
        self.csv_path = csv_path
        self._fieldnames: list[str] = []
        self._rows: list[dict] = []

    def _collect_metrics(self, trainer, epoch, train_loss, val_loss, metrics):
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": trainer.optimizer.param_groups[0]["lr"],
        }
        row.update(metrics)
        grad = getattr(trainer, '_max_grad_norm', None)
        row["max_grad_norm"] = grad if grad is not None else ""
        gap = getattr(trainer, '_overfit_gap_ratio', None)
        row["overfit_gap_ratio"] = gap if gap is not None else ""
        pred = getattr(trainer, '_loss_prediction', None)
        row["predicted_final_loss"] = pred if pred is not None else ""
        return row

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        row = self._collect_metrics(trainer, epoch, train_loss, val_loss, metrics)
        new_keys = [k for k in row if k not in self._fieldnames]
        self._rows.append(row)

        if new_keys:
            # New columns appeared — expand header and rewrite everything
            self._fieldnames.extend(new_keys)
            self._flush_all()
        else:
            # Fast path — append one row
            with open(self.csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=self._fieldnames, restval="")
                writer.writerow(row)

    def _flush_all(self):
        """Rewrite the entire CSV with the current (possibly expanded) header."""
        with open(self.csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, restval="")
            writer.writeheader()
            writer.writerows(self._rows)
```

Add `import csv` at the top of `callbacks.py`.

2. Add `TUILoggingCallback` (priority=80) — agent-friendly terminal logging (replaces wandb when `logging: tui`):

```python
class TUILoggingCallback(TrainingCallback):
    """Agent-friendly terminal logging (replaces wandb)."""
    priority = 80

    def on_train_begin(self, trainer, **kwargs):
        from tqdm import tqdm
        epochs = kwargs.get("epochs", "?")
        tqdm.write(f"{'='*60}")
        tqdm.write(f"Training started  |  epochs: {epochs}")
        tqdm.write(f"{'='*60}")

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        from tqdm import tqdm
        parts = [
            f"[{epoch+1:>4d}/{trainer._total_epochs}]",
            f"train: {train_loss:.4e}",
            f"val: {val_loss:.4e}",
            f"lr: {trainer.optimizer.param_groups[0]['lr']:.4e}",
        ]
        for k, v in metrics.items():
            if isinstance(v, float):
                parts.append(f"{k}: {v:.4e}")

        if hasattr(trainer, '_max_grad_norm') and trainer._max_grad_norm is not None:
            parts.append(f"grad: {trainer._max_grad_norm:.2e}")
        if hasattr(trainer, '_loss_prediction') and trainer._loss_prediction is not None:
            parts.append(f"pred: {trainer._loss_prediction:.4e}")

        tqdm.write(" | ".join(parts))

    def on_train_end(self, trainer, **kwargs):
        from tqdm import tqdm
        tqdm.write(f"{'='*60}")
        tqdm.write("Training complete")
        tqdm.write(f"{'='*60}")
```

3. Add `LatestModelCallback` (priority=96) — saves model state dict every epoch:

```python
class LatestModelCallback(TrainingCallback):
    """Saves latest model state_dict every epoch."""
    priority = 96

    def __init__(self, save_path: str):
        self.save_path = save_path

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        import torch
        torch.save(trainer.model.state_dict(), self.save_path)
```

### provenance.py

Create the file from scratch:

```python
import platform
import sys
import os
import hashlib
import json
import time
from dataclasses import asdict
from typing import Any


def capture_environment() -> dict[str, Any]:
    """Capture full execution environment for reproducibility."""
    env = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "hostname": platform.node(),
    }

    try:
        import torch
        env["torch_version"] = torch.__version__
        env["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            env["cuda_version"] = torch.version.cuda or "N/A"
            env["gpu_count"] = torch.cuda.device_count()
            env["gpu_devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_mb": round(torch.cuda.get_device_properties(i).total_memory / 1024**2),
                }
                for i in range(torch.cuda.device_count())
            ]
        env["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
    except ImportError:
        env["torch_version"] = "not installed"

    try:
        import numpy
        env["numpy_version"] = numpy.__version__
    except ImportError:
        pass

    try:
        import subprocess
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            env["git_commit"] = result.stdout.strip()
        result_dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5
        )
        if result_dirty.returncode == 0:
            env["git_dirty"] = len(result_dirty.stdout.strip()) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    env_vars = {}
    for var in ["PYTHONHASHSEED", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "CUDA_VISIBLE_DEVICES", "CUBLAS_WORKSPACE_CONFIG"]:
        val = os.environ.get(var)
        if val is not None:
            env_vars[var] = val
    if env_vars:
        env["env_vars"] = env_vars

    return env


def compute_config_hash(config) -> str:
    """Compute SHA-256 hash of canonicalized config for deduplication."""
    config_dict = asdict(config) if hasattr(config, '__dataclass_fields__') else dict(config)
    config_dict.pop('_frozen', None)
    canonical = json.dumps(config_dict, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()


def capture_run_metadata(model, device: str, start_time: float, end_time: float) -> dict[str, Any]:
    """Capture metadata about a completed training run."""
    import torch
    metadata = {
        "training_time_seconds": round(end_time - start_time, 2),
        "device": device,
    }
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metadata["total_parameters"] = total_params
    metadata["trainable_parameters"] = trainable_params
    if device.startswith("cuda") and torch.cuda.is_available():
        device_idx = int(device.split(":")[-1]) if ":" in device else 0
        metadata["peak_gpu_memory_mb"] = round(
            torch.cuda.max_memory_allocated(device_idx) / 1024**2, 2
        )
    return metadata


def save_provenance(run_path: str, config, model, device: str,
                    start_time: float, end_time: float) -> None:
    """Save all provenance files to the run directory."""
    import yaml

    env = capture_environment()
    config_hash = compute_config_hash(config)
    run_meta = capture_run_metadata(model, device, start_time, end_time)

    with open(os.path.join(run_path, "env_snapshot.yaml"), "w") as f:
        yaml.dump(env, f, sort_keys=False, default_flow_style=False)

    run_meta["config_hash"] = config_hash
    with open(os.path.join(run_path, "run_metadata.yaml"), "w") as f:
        yaml.dump(run_meta, f, sort_keys=False, default_flow_style=False)
```

### cli.py

1. Add `doctor` command (checks system environment):

```python
@app.command()
def doctor():
    """Check system environment: Python, PyTorch, CUDA, packages, and wandb."""
    from provenance import capture_environment

    env = capture_environment()

    table = Table(title="System Doctor", show_lines=True)
    table.add_column("Check", style="bold cyan")
    table.add_column("Status")

    table.add_row("Python version", f"[green]{env['python_version']}[/green]")

    torch_ver = env.get("torch_version", "not installed")
    if torch_ver == "not installed":
        table.add_row("PyTorch", "[red]not installed[/red]")
    else:
        table.add_row("PyTorch", f"[green]{torch_ver}[/green]")

    cuda_available = env.get("cuda_available", False)
    if cuda_available:
        cuda_ver = env.get("cuda_version", "N/A")
        table.add_row("CUDA", f"[green]available (v{cuda_ver})[/green]")
        gpu_devices = env.get("gpu_devices", [])
        for i, gpu in enumerate(gpu_devices):
            table.add_row(
                f"  GPU {i}",
                f"[green]{gpu['name']} ({gpu['memory_total_mb']} MB)[/green]",
            )
    else:
        table.add_row("CUDA", "[yellow]not available[/yellow]")

    try:
        import wandb
        api_key = wandb.api.api_key
        if api_key:
            table.add_row("wandb", "[green]logged in[/green]")
        else:
            table.add_row("wandb", "[red]not logged in[/red]")
    except Exception:
        table.add_row("wandb", "[red]not installed or error[/red]")

    required_packages = [
        "torch", "numpy", "optuna", "wandb", "tqdm", "rich", "beaupy", "scienceplots",
    ]
    for pkg in required_packages:
        try:
            mod = __import__(pkg)
            version = getattr(mod, "__version__", "ok")
            table.add_row(f"  {pkg}", f"[green]{version}[/green]")
        except ImportError:
            table.add_row(f"  {pkg}", "[red]missing[/red]")

    console.print(table)
```

2. Add `monitor` command (launches the Rust TUI binary):

```python
@app.command()
def monitor(
    path: str = typer.Argument(
        None, help="Path to metrics.csv or its parent directory"
    ),
    interval: int = typer.Option(500, help="Refresh interval in milliseconds"),
):
    """Launch the real-time TUI training monitor (Rust binary)."""
    import os
    import subprocess
    import glob as glob_mod

    monitor_bin = os.path.join(os.path.dirname(__file__), "tools", "monitor", "target", "release", "training-monitor")

    if not os.path.exists(monitor_bin):
        console.print("[yellow]Monitor binary not found. Building...[/yellow]")
        cargo_dir = os.path.join(os.path.dirname(__file__), "tools", "monitor")
        result = subprocess.run(["cargo", "build", "--release"], cwd=cargo_dir)
        if result.returncode != 0:
            console.print("[red]Failed to build monitor. Install Rust: https://rustup.rs[/red]")
            raise typer.Exit(code=1)

    if path is None:
        candidates = glob_mod.glob("runs/**/metrics.csv", recursive=True)
        if not candidates:
            console.print("[red]No metrics.csv found under runs/. Specify a path.[/red]")
            raise typer.Exit(code=1)
        path = max(candidates, key=os.path.getmtime)
        console.print(f"[dim]Auto-detected: {path}[/dim]")

    try:
        subprocess.run([monitor_bin, path, "--interval", str(interval)])
    except KeyboardInterrupt:
        pass
```

### util.py

Update `run()` to:

1. Read `use_wandb` from config:

```python
use_wandb = run_config.logging == "wandb"
```

2. Conditionally add `WandbLoggingCallback` vs `TUILoggingCallback`, and always add `CSVLoggingCallback` and `LatestModelCallback`:

```python
callbacks_list = [
    OptimizerModeCallback(),
    NaNDetectionCallback(),
    GradientMonitorCallback(),
    LossPredictionCallback(run_config.epochs),
    OverfitDetectionCallback(),
]
if use_wandb:
    callbacks_list.append(WandbLoggingCallback())
else:
    callbacks_list.append(TUILoggingCallback())
# Always-on callbacks: CSV logging + latest model save
run_path = f"{group_path}/{run_name}"
if not os.path.exists(run_path):
    os.makedirs(run_path)
callbacks_list.append(CSVLoggingCallback(f"{run_path}/metrics.csv"))
callbacks_list.append(LatestModelCallback(f"{run_path}/latest_model.pt"))
```

3. Call `save_provenance()` after training completes:

```python
from provenance import save_provenance, compute_config_hash

start_time = time.time()
val_loss = trainer.train(dl_train, dl_val, epochs=run_config.epochs)
end_time = time.time()

torch.save(model.state_dict(), f"{run_path}/model.pt")

# Save provenance
save_provenance(run_path, run_config, model, device, start_time, end_time)
```

4. Update imports:

```python
from callbacks import (
    ..., CSVLoggingCallback, TUILoggingCallback, LatestModelCallback,
)
from provenance import save_provenance, compute_config_hash
```

### configs/run_template.yaml

Add `logging` field (typically right after `device`):

```yaml
logging: wandb  # 'wandb' or 'tui' (agent-friendly terminal logging)
```

### tools/monitor/ (optional Rust TUI)

The `tools/monitor/` directory contains a Rust binary (`training-monitor`) that reads `metrics.csv` in real time and renders a live terminal dashboard with loss curves, learning rate, and gradient norm plots. It is **optional** — the `monitor` CLI command builds it automatically via `cargo build --release` on first use.

To use it manually:
```bash
cd tools/monitor && cargo build --release
./target/release/training-monitor runs/MyProject/exp1/42/metrics.csv --interval 500
```

If Rust is not installed, the `monitor` command will print a link to `https://rustup.rs`. The CSV-based logging (`CSVLoggingCallback`) that feeds the monitor is always active regardless of whether the Rust binary is present.

---

## How to Apply Migrations to a Real Project

For projects that forked/diverged significantly from the template (custom util.py, custom callbacks, etc.):

1. **DO NOT** overwrite `util.py` wholesale — the user likely has custom `load_data()`, analysis functions, etc.
2. **DO** apply targeted edits to `config.py` (add fields, methods)
3. **DO** append new callbacks to `callbacks.py` (don't remove existing custom ones)
4. **DO** add new CLI commands to `cli.py` (append, don't replace)
5. **DO** add `data` field to YAML configs
6. **DO** update imports in `util.py` and wire new callbacks into `run()`

### Divergence handling

If the user has heavily modified a file:
1. Read both the user's version and the template's version
2. Identify which template changes are missing
3. Apply only the missing changes, preserving user's customizations
4. Verify with `pytest` and `preflight` after migration
