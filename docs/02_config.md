---
title: Configuration Deep Dive
nav_order: 2
---

# Configuration Deep Dive

The configuration system is the backbone of reproducible research. Every experiment is fully described by one or two YAML files, and `RunConfig` enforces correctness before a single training step runs.

---

## The Three Tiers of Validation

`RunConfig` validates your config in three distinct stages, each catching a different class of error.

### Tier 1: Structural (in `__post_init__`)

Runs immediately when the dataclass is constructed — before any imports, before any hardware is checked. It catches obviously malformed configs:

- `epochs > 0`, `batch_size > 0`, `seeds` is non-empty
- `net`, `optimizer`, `scheduler`, and `data` must be in `module.Class` (dot-separated) format
- If early stopping is enabled: `patience > 0`, `mode` is `"min"` or `"max"`

After Tier 1 passes, the config is **frozen**. Any attempt to mutate a field raises `AttributeError` — use `with_overrides()` instead.

### Tier 2: Runtime (`validate_for_execution()`)

Call this just before training begins. It verifies that the environment can actually run the config:

- If `device` starts with `"cuda"`, checks that `torch.cuda.is_available()`
- Attempts `importlib.import_module` for each dotted path (`net`, `optimizer`, `scheduler`, `criterion`, `data`) and confirms the named attribute exists

This separation matters: you can load and inspect configs without a GPU present, only failing at execution time.

### Tier 3: Semantic (`validate_semantics()`)

Returns a list of warning strings (empty = all clear). This tier catches bugs that Tiers 1 and 2 miss — ones that only surface when **values interact**:

- `optimizer_config.lr` must be positive
- `seeds` must be unique (duplicate seeds waste compute by running identical experiments)
- For `ExpHyperbolicLRScheduler` / `HyperbolicLRScheduler`: `upper_bound >= total_steps`
- For `CosineAnnealingLR`: `T_max` should equal `epochs`
- If early stopping is enabled: `patience < epochs` (otherwise early stopping can never trigger)

> **Key insight:** Tier 3 catches the bugs that Tiers 1 and 2 miss — the ones that only matter when components interact. A config where `patience = 100` and `epochs = 50` is structurally valid and importable, but logically broken.

---

## RunConfig Fields

All fields are required unless a default is shown.

| Field | Type | Default | Controls |
|---|---|---|---|
| `project` | `str` | — | W&B project name; also names the SQLite DB for HPO |
| `device` | `str` | — | `"cpu"`, `"cuda:0"`, etc. |
| `seeds` | `list[int]` | — | One training run per seed; results are averaged |
| `net` | `str` | — | `module.ClassName` of the model class |
| `optimizer` | `str` | — | `module.ClassName` of the optimizer |
| `scheduler` | `str` | — | `module.ClassName` of the LR scheduler |
| `epochs` | `int` | — | Total training epochs per seed |
| `batch_size` | `int` | — | Dataloader batch size |
| `net_config` | `dict` | — | Kwargs forwarded to the model constructor |
| `optimizer_config` | `dict` | — | Kwargs forwarded to the optimizer (e.g., `lr`) |
| `scheduler_config` | `dict` | — | Kwargs forwarded to the scheduler |
| `criterion` | `str` | `"torch.nn.MSELoss"` | Loss function, in `module.ClassName` format |
| `criterion_config` | `dict` | `{}` | Kwargs forwarded to the criterion constructor |
| `data` | `str` | `"util.load_data"` | `module.function` path for the data loader function |
| `early_stopping_config` | `EarlyStoppingConfig` | disabled | Early stopping settings (see below) |
| `checkpoint_config` | `CheckpointConfig` | disabled | Checkpoint saving settings (see below) |
| `monitor` | `str` | `"val_loss"` | Metric to track for best-model selection |
| `logging` | `str` | `"wandb"` | `"wandb"` for W&B logging, `"tui"` for agent-friendly terminal logging |

**`EarlyStoppingConfig` sub-fields:** `enabled` (bool, default `False`), `patience` (int, default `10`), `mode` (`"min"`/`"max"`, default `"min"`), `min_delta` (float, default `0.0001`).

**`CheckpointConfig` sub-fields:** `enabled` (bool, default `False`), `save_every_n_epochs` (int, default `10`), `keep_last_k` (int, default `3`), `save_best` (bool, default `True`), `monitor` (str, default `"val_loss"`), `mode` (`"min"`/`"max"`, default `"min"`).

### Frozen configs and `with_overrides()`

After `__post_init__`, the config is immutable. To create a variant (e.g., for HPO trial overrides):

```python
hpo_config = base_config.with_overrides(
    epochs=10,
    optimizer_config={"lr": trial_lr},
)
```

`with_overrides()` does a **deep merge** for dict fields: keys you specify are updated, keys you omit are preserved. It then constructs and returns a new validated `RunConfig`.

---

## The `data` Field — Pluggable Data Loading

The `data` field specifies a `module.function` path that `RunConfig.load_data()` will resolve and call via `importlib`. The function must return `(train_dataset, val_dataset)`.

Before this field existed, switching datasets required editing `cli.py` or `util.py` directly. Now it is a single YAML line:

```yaml
# Default: use the built-in synthetic regression dataset
data: util.load_data

# Use a recipe's dataset
data: recipes.regression.data.load_data

# Use a project-specific custom loader
data: my_project.data.load_custom_data
```

The resolution chain is:

```python
module_name, func_name = self.data.rsplit(".", 1)
module = importlib.import_module(module_name)
load_fn = getattr(module, func_name)
return load_fn()
```

Tier 2 validation (`validate_for_execution`) will catch any `ImportError` or missing attribute before training begins, so typos in the path fail fast with a clear error message.

> **Key insight:** Before this field, every project had to edit `cli.py` to change datasets. Now it is one YAML line. Your experiment YAML is fully self-describing — another researcher can reproduce your exact data pipeline just from the config file.

---

## OptimizeConfig

`OptimizeConfig` is loaded from a separate `opt.yaml` file and drives Optuna HPO.

```yaml
study_name: Optimize_Template
trials: 20
seed: 42
metric: val_loss
direction: minimize

sampler:
  name: optuna.samplers.TPESampler
  # kwargs:               # optional; forwarded to sampler constructor
  #   n_startup_trials: 10

pruner:
  name: pruner.PFLPruner
  kwargs:
    n_startup_trials: 10
    n_warmup_epochs: 10
    top_k: 10
    target_epoch: 50

search_space:
  net_config:
    nodes:
      type: categorical
      choices: [32, 64, 128]
    layers:
      type: int
      min: 3
      max: 5
  optimizer_config:
    lr:
      type: float
      min: 1.e-3
      max: 1.e-2
      log: true        # log-uniform sampling — essential for learning rates
  scheduler_config:
    upper_bound:
      type: int
      min: 300
      max: 400
      step: 50
    min_lr:
      type: float
      min: 1.e-7
      max: 1.e-4
      log: true
```

**`search_space` structure:** Top-level keys (`net_config`, `optimizer_config`, `scheduler_config`) map to the corresponding `RunConfig` dict fields. Each parameter entry has a `type` (`int`, `float`, or `categorical`) plus range bounds or `choices`. Float parameters support `log: true` for log-uniform sampling, which is strongly recommended for learning rates and regularization coefficients.

**Sampler:** Any `optuna.samplers.*` class. `TPESampler` (Tree-structured Parzen Estimator) is the default and appropriate for most use cases. `GridSampler` is also supported — when selected, `OptimizeConfig` automatically constructs the grid from `categorical` entries in the search space.

**Pruner:** Optional. If omitted, all trials run to completion. The built-in `PFLPruner` prunes underperforming trials based on projected final loss across multiple seeds.

---

## Config Naming Convention

Configs live under `configs/` and follow this convention:

```
configs/<CONTRIBUTION>_v<VERSION>/<MODEL>_{run,opt,best}.yaml
```

- `<CONTRIBUTION>` — the experiment name or project shorthand (e.g., `SolarFlux`, `WavePredict`)
- `<VERSION>` — integer version, incremented when the search space or architecture changes significantly
- `<MODEL>` — the model or configuration variant being tested

Examples:

```
configs/SolarFlux_v1/MLP_run.yaml       # base run config for HPO
configs/SolarFlux_v1/MLP_opt.yaml       # HPO search space
configs/SolarFlux_v1/MLP_best.yaml      # best config after HPO

configs/WavePredict_v2/HNN_run.yaml
configs/WavePredict_v2/HNN_opt.yaml
configs/WavePredict_v2/HNN_best.yaml
```

This convention makes it immediately clear which phase of the workflow each file belongs to, and version numbers let you track search space evolution without losing old configs.

---

## Config Lifecycle

A complete experiment follows three config states:

```
run.yaml  ──────►  HPO (opt.yaml)  ──────►  best.yaml
(base)              (search space)            (final run)
```

**`run.yaml` (HPO base):** Sets fixed hyperparameters and HPO-trial settings. Use `epochs: 10` and `seeds: [42]` here — HPO trials should be short and use a single seed to keep compute manageable.

**`opt.yaml` (search space):** Defines what Optuna will vary. Points back to `run.yaml` as the base; trial overrides are merged via `with_overrides()`.

**`best.yaml` (final run):** Generated after HPO completes. It is `run.yaml` with:
- `epochs` set to the full training budget (e.g., 200)
- `scheduler_config.total_steps` (or `T_max`) updated to match new `epochs`
- `seeds` expanded to the full multi-seed list (e.g., `[89, 231, 928, 814, 269]`)
- `optimizer_config.lr`, `net_config.layers`, and any other searched params set to the best trial's values

The result is a single file that fully describes a reproducible, optimized experiment — no code changes required.
