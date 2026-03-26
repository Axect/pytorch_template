# Migration Reference

Detailed steps for each migration. Apply in order, skip migrations that are already present.

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

## M5: Diagnostics + Preflight + HPO Report (v5 → current)

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
