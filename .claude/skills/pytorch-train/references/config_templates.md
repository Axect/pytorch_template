# Config Templates Reference

Full annotated YAML templates for each config type. Copy and modify as needed.

---

## Regression: run.yaml (HPO Base)

```yaml
# ── Project Identification ──
project: <PROJECT>_v<VERSION>_<MODEL>    # e.g., OSPREY_v0.10_DeepONet
device: cuda:0                            # cuda:N or cpu

# ── Model ──
net: <module.ClassName>                   # e.g., model.MLP, recipes.regression.model.MLP
net_config:
  nodes: 64                               # FIXED during HPO — set wide enough
  layers: 4                               # HPO tunes this

# ── Optimizer: SPlus ──
optimizer: pytorch_optimizer.SPlus
optimizer_config:
  lr: 1.e-1                               # HPO tunes this in [1e-3, 1e+0]
  eps: 1.e-10

# ── Scheduler: ExpHyperbolicLR ──
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
scheduler_config:
  total_steps: 150                        # Full training scale, NOT synced to HPO epochs
  upper_bound: 300                        # Must be >= total_steps
  min_lr: 1.e-6                           # HPO tunes this

# ── Training ──
epochs: 10                                # Short for HPO (epoch-insensitive scheduler)
batch_size: 256
seeds: [42]                               # Single seed for HPO speed

# ── Loss ──
criterion: torch.nn.MSELoss
criterion_config: {}

# ── Early Stopping (disabled for HPO) ──
early_stopping_config:
  enabled: false
  patience: 10
  mode: min
  min_delta: 0.0001

# ── Checkpointing (disabled for HPO) ──
checkpoint_config:
  enabled: false
  save_every_n_epochs: 10
  keep_last_k: 3
  save_best: true
  monitor: val_loss
  mode: min
```

---

## Regression: opt.yaml (HPO Search)

```yaml
# ── Study ──
study_name: <MODEL>_TPE                   # e.g., DeepONet_TPE
trials: 50                                # 30-100 depending on search space
seed: 42
metric: val_loss
direction: minimize

# ── Sampler ──
sampler:
  name: optuna.samplers.TPESampler
  # kwargs:
  #   n_startup_trials: 10               # Uncomment to customize

# ── Pruner ──
pruner:
  name: pruner.PFLPruner
  kwargs:
    n_startup_trials: 10                  # Run first N trials without pruning
    n_warmup_epochs: 5                    # Train N epochs before pruning is enabled
    top_k: 10                             # Compare against top K trials
    target_epoch: 10                      # Must match run.yaml epochs

# ── Search Space ──
# Only tune: layers, lr, min_lr
# Fix: nodes (width), upper_bound, total_steps
search_space:
  net_config:
    layers:
      type: int
      min: 3
      max: 6
  optimizer_config:
    lr:
      type: float
      min: 1.e-3                          # SPlus range: [1e-3, 1e+0]
      max: 1.e+0                          # NEVER use [1e-5, 1e-2] for SPlus
      log: true
  scheduler_config:
    min_lr:
      type: float
      min: 1.e-7
      max: 1.e-3
      log: true
```

---

## Regression: best.yaml (Post-HPO Final Training)

```yaml
# ── Project Identification ──
project: <PROJECT>_v<VERSION>_<MODEL>     # Remove _Opt suffix
device: cuda:0

# ── Model ──
net: <module.ClassName>
net_config:
  nodes: 64                               # Same as run.yaml
  layers: <FROM_HPO>                      # Best trial value

# ── Optimizer: SPlus ──
optimizer: pytorch_optimizer.SPlus
optimizer_config:
  lr: <FROM_HPO>                          # Best trial value (e.g., 3.42e-1)
  eps: 1.e-10

# ── Scheduler: ExpHyperbolicLR ──
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
scheduler_config:
  total_steps: 150                        # Now total_steps == epochs
  upper_bound: 300
  min_lr: <FROM_HPO>                      # Best trial value (e.g., 2.15e-5)

# ── Training ──
epochs: 150                               # Full training (task-dependent: 100-500)
batch_size: 256
seeds: [58, 89, 231, 928, 814]           # Multi-seed for statistical significance

# ── Loss ──
criterion: torch.nn.MSELoss
criterion_config: {}

# ── Early Stopping (recommended for long training) ──
early_stopping_config:
  enabled: true
  patience: 30
  mode: min
  min_delta: 0.0001

# ── Checkpointing ──
checkpoint_config:
  enabled: true
  save_every_n_epochs: 25
  keep_last_k: 3
  save_best: true
  monitor: val_loss
  mode: min
```

---

## Classification: run.yaml (HPO Base)

```yaml
project: <PROJECT>_v<VERSION>_<MODEL>
device: cuda:0

net: <module.ClassName>                   # e.g., recipes.classification.model.SimpleCNN
net_config:
  channels: 32                            # FIXED during HPO
  num_classes: 10                         # FIXED — dataset dependent

optimizer: pytorch_optimizer.SPlus
optimizer_config:
  lr: 1.e-1
  eps: 1.e-10

scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
scheduler_config:
  total_steps: 150
  upper_bound: 300
  min_lr: 1.e-6

epochs: 10
batch_size: 128                           # Smaller for image data
seeds: [42]

criterion: torch.nn.CrossEntropyLoss      # Classification loss
criterion_config: {}

early_stopping_config:
  enabled: false
  patience: 10
  mode: min
  min_delta: 0.0001

checkpoint_config:
  enabled: false
  save_every_n_epochs: 10
  keep_last_k: 3
  save_best: true
  monitor: val_loss
  mode: min
```

---

## Classification: opt.yaml (HPO Search)

```yaml
study_name: <MODEL>_TPE
trials: 50
seed: 42
metric: val_loss
direction: minimize

sampler:
  name: optuna.samplers.TPESampler

pruner:
  name: pruner.PFLPruner
  kwargs:
    n_startup_trials: 10
    n_warmup_epochs: 5
    top_k: 10
    target_epoch: 10

search_space:
  net_config:
    channels:
      type: categorical
      choices: [16, 32, 64]
  optimizer_config:
    lr:
      type: float
      min: 1.e-3
      max: 1.e+0
      log: true
  scheduler_config:
    min_lr:
      type: float
      min: 1.e-7
      max: 1.e-3
      log: true
```

---

## Non-SPlus Alternative: AdamW + CosineAnnealing

For users who prefer standard optimizers:

### run.yaml

```yaml
optimizer: torch.optim.AdamW
optimizer_config:
  lr: 1.e-3
  weight_decay: 1.e-4

scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_config:
  T_max: 50                              # Must equal epochs
  eta_min: 1.e-5

epochs: 50
```

### opt.yaml search_space

```yaml
search_space:
  net_config:
    layers:
      type: int
      min: 3
      max: 6
  optimizer_config:
    lr:
      type: float
      min: 1.e-5                          # Standard range for AdamW
      max: 1.e-2
      log: true
    weight_decay:
      type: float
      min: 1.e-6
      max: 1.e-2
      log: true
  scheduler_config:
    eta_min:
      type: float
      min: 1.e-7
      max: 1.e-4
      log: true
```

**Note**: With CosineAnnealing, `T_max` MUST equal `epochs`. Adjust both together when switching between HPO and final training.

---

## Field Reference

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `project` | str | Yes | Project identifier. Convention: `<NAME>_v<VER>_<MODEL>` |
| `device` | str | Yes | `cuda:0`, `cuda:1`, or `cpu` |
| `net` | str | Yes | Module.Class path for model |
| `net_config` | dict | Yes | Kwargs passed to model constructor as `hparams` |
| `optimizer` | str | Yes | Module.Class path for optimizer |
| `optimizer_config` | dict | Yes | Kwargs passed to optimizer (must include `lr`) |
| `scheduler` | str | Yes | Module.Class path for scheduler |
| `scheduler_config` | dict | Yes | Kwargs passed to scheduler |
| `epochs` | int | Yes | Number of training epochs |
| `batch_size` | int | Yes | Batch size for DataLoader |
| `seeds` | list[int] | Yes | Random seeds for reproducibility |
| `criterion` | str | Yes | Module.Class path for loss function |
| `criterion_config` | dict | Yes | Kwargs passed to loss constructor (often `{}`) |
| `early_stopping_config` | dict | No | See below |
| `checkpoint_config` | dict | No | See below |

### early_stopping_config

| Sub-field | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable early stopping |
| `patience` | int | 10 | Epochs without improvement before stopping |
| `mode` | str | "min" | "min" (lower is better) or "max" |
| `min_delta` | float | 0.0001 | Minimum improvement threshold (percent) |

### checkpoint_config

| Sub-field | Type | Default | Description |
|-----------|------|---------|-------------|
| `enabled` | bool | false | Enable checkpointing |
| `save_every_n_epochs` | int | 10 | Save checkpoint every N epochs |
| `keep_last_k` | int | 3 | Keep only last K periodic checkpoints |
| `save_best` | bool | true | Save best model separately |
| `monitor` | str | "val_loss" | Metric to monitor for best model |
| `mode` | str | "min" | "min" or "max" for monitor metric |
