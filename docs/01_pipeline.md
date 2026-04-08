---
title: The Full Pipeline
nav_order: 1
---

# The Full Pipeline

This chapter walks through a complete experiment lifecycle — from writing your first config to analyzing the best model. Every command shown here is the same command the AI agent uses.

---

## Phase 1: Write Your Config

One YAML file defines everything about an experiment: model architecture, optimizer, scheduler, dataset, training length, seeds. This is intentional — one file per experiment means you can reproduce any result by re-running one command with one config.

Here is a complete annotated `run.yaml` for a regression task with SPlus + ExpHyperbolicLR:

```yaml
# ── Project Identification ──
project: SolarFlux_v0.3_FluxNet   # Convention: <NAME>_v<VERSION>_<MODEL>
device: cuda:0                    # cuda:N or cpu

# ── Model ──
net: model.MLP                    # importlib path: <module>.<ClassName>
net_config:
  nodes: 64                       # Width — fix during HPO, tune manually
  layers: 4                       # Depth — HPO tunes this

# ── Optimizer: SPlus ──
optimizer: pytorch_optimizer.SPlus
optimizer_config:
  lr: 1.e-1                       # Starting lr — HPO tunes this in [1e-3, 1e+0]
  eps: 1.e-10

# ── Scheduler: ExpHyperbolicLR ──
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
scheduler_config:
  total_steps: 150                # Full-training scale, NOT synced to HPO epochs
  upper_bound: 300                # Must be >= total_steps
  min_lr: 1.e-6                   # HPO tunes this

# ── Training ──
epochs: 10                        # 10 epochs for HPO; increase to 150 for final run
batch_size: 256
seeds: [42]                       # Single seed during HPO; expand for final run

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

# ── Data Loader ──
data: util.load_data              # importlib path to a load_data() function
```

Change datasets by changing the `data:` field — one line. The function must return `(train_dataset, val_dataset)` as PyTorch Dataset objects.

Config files live under `configs/<MAIN_CONTRIBUTION>_v<VERSION>/`:

```
configs/SolarFlux_v0.3/
├── fluxnet_run.yaml    # HPO base config
├── fluxnet_opt.yaml    # HPO search config
└── fluxnet_best.yaml   # Final training config (created after HPO)
```

See **[Chapter 2: Configuration Deep Dive](02_config.html)** for every field explained.

---

## Phase 2: Pre-flight Check

A single shape mismatch can waste hours of GPU time. Preflight catches it in seconds by running one batch forward and backward through the full stack — data loading, model instantiation, optimizer step, gradient check.

```bash
python -m cli preflight configs/SolarFlux_v0.3/fluxnet_run.yaml --device cuda:0
```

The output is a table of checks:

```
┌──────────────────────────────┬────────┬────────────────────────────────┐
│ Check                        │ Status │ Detail                         │
├──────────────────────────────┼────────┼────────────────────────────────┤
│ Import paths                 │ PASS   │ model.MLP, pytorch_optimizer.. │
│ Device availability          │ PASS   │ cuda:0                         │
│ Semantic validation          │ PASS   │ upper_bound >= total_steps     │
│ Model instantiation          │ PASS   │ MLP: 12,481 parameters         │
│ Data loading                 │ PASS   │ train=8000, val=2000           │
│ Forward pass                 │ PASS   │ output (256, 1) vs target (256,│
│ Backward pass                │ PASS   │ grad_norm=0.0234               │
│ GPU memory estimate          │ PASS   │ ~47 MB                         │
└──────────────────────────────┴────────┴────────────────────────────────┘
All pre-flight checks passed.
```

Fix every FAIL and investigate every WARN before proceeding. Use `--json` for machine-readable output when scripting or using agent tools.

Also available before preflight:

```bash
python -m cli validate configs/SolarFlux_v0.3/fluxnet_run.yaml  # Schema + semantic checks only
python -m cli preview configs/SolarFlux_v0.3/fluxnet_run.yaml   # Print model architecture
```

---

## Phase 3: Training

```bash
python -m cli train configs/SolarFlux_v0.3/fluxnet_run.yaml --device cuda:0
```

What happens when you run this:

1. The config is loaded and validated.
2. `load_data()` is called via the `data:` field — returns train and val datasets.
3. For each seed in `seeds`, a complete training run executes:
   - Model, optimizer, scheduler, criterion are instantiated.
   - All active callbacks are attached (see Chapter 3).
   - The training loop runs for `epochs` epochs.
   - Results are saved to `runs/<project>/<group_name>/<seed>/`.
4. W&B logs `train_loss`, `val_loss`, `lr`, `max_grad_norm`, and `overfit_gap_ratio` every epoch.

Two diagnostic callbacks run automatically during every training session:

- **GradientMonitorCallback** — Computes gradient L2 norm per step. Logs `max_grad_norm` per epoch. Warns in the console if grad norm exceeds 10,000 (potential exploding gradients).
- **OverfitDetectionCallback** — After 5 warmup epochs, detects sustained divergence where training loss decreases while validation loss increases over 5 consecutive epochs. Logs `overfit_gap_ratio` when detected. Warning only — does not stop training.

For long runs, queue with pueue so they survive session termination:

```bash
pueue group add SolarFlux
pueue add -g SolarFlux -- bash -c \
  "cd /path/to/project && .venv/bin/python -m cli train \
   configs/SolarFlux_v0.3/fluxnet_run.yaml --device cuda:0"
pueue status -g SolarFlux
```

---

## Phase 4: HPO with Optuna

HPO finds the best hyperparameters by running many short training trials and using the results to guide the search. The optimizer config (`opt.yaml`) defines the search space:

```yaml
study_name: FluxNet_TPE
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
    target_epoch: 10        # Must match run.yaml epochs

search_space:
  net_config:
    layers:
      type: int
      min: 3
      max: 6
  optimizer_config:
    lr:
      type: float
      min: 1.e-3            # SPlus range: NEVER use [1e-5, 1e-2]
      max: 1.e+0
      log: true
  scheduler_config:
    min_lr:
      type: float
      min: 1.e-7
      max: 1.e-3
      log: true
```

Run HPO:

```bash
python -m cli train configs/SolarFlux_v0.3/fluxnet_run.yaml \
  --optimize-config configs/SolarFlux_v0.3/fluxnet_opt.yaml \
  --device cuda:0
```

Key insight: With SPlus + ExpHyperbolicLR, **10 epochs is enough for HPO**. The scheduler is epoch-insensitive — the lr curve shape is set by `upper_bound / total_steps`, so 10 epochs of a 150-step schedule gives a reliable training signal. Trial ordering is preserved when you later train for 150 epochs. You do not need to retune.

The PFLPruner stops unpromising trials early by predicting their final loss from the first few epochs. This can cut HPO wall time by 30-60%.

While HPO runs, you can monitor progress in real time with the TUI's `--hpo` mode:

```bash
# In a separate terminal
python -m cli monitor --hpo
```

This shows a live scatter of trial objective values, parameter scatter grids, and the best trial's training curves — all updated as new trials complete. See **[Chapter 4: Monitoring HPO in Real-time](04_hpo.html#monitoring-hpo-in-real-time)** for the full tab reference.

See **[Chapter 4: Hyperparameter Optimization](04_hpo.html)** for search space design, sampler choice, and pruner tuning.

---

## Phase 5: HPO Analysis

After HPO completes, analyze the results before creating `best.yaml`:

```bash
# Auto-detects the .db file if only one exists
python -m cli hpo-report

# Explicit — use when multiple studies exist
python -m cli hpo-report --db SolarFlux_v0.3_FluxNet_Opt.db --study-name FluxNet_TPE

# With boundary warnings — recommended
python -m cli hpo-report \
  --db SolarFlux_v0.3_FluxNet_Opt.db \
  --opt-config configs/SolarFlux_v0.3/fluxnet_opt.yaml
```

The report shows:

1. **Completion stats** — How many trials completed, were pruned, or failed.
2. **Best trial** — The winning parameter set and its validation loss.
3. **Parameter importance** — fANOVA-based ranking of which hyperparameters matter most.
4. **Boundary warnings** — If your best parameter sits at the edge of its search range, you need to widen that range and re-run.
5. **Top-K trials** — A table of the best trials for comparison.

The boundary warning is the most important check. If the best `lr` is `1.0` (the top of the range `[1e-3, 1e+0]`), the true optimum is outside your search space. Widen the range, change the study name in `opt.yaml` to avoid conflicts with the previous run, and re-run HPO.

---

## Phase 6: Final Training

Create `best.yaml` by copying `run.yaml` and applying the HPO results. The key differences:

```yaml
# From run.yaml → best.yaml

epochs: 150              # Was: 10
seeds: [58, 89, 231, 928, 814]   # Was: [42]

# HPO-found values (examples)
net_config:
  layers: 5              # Was: 4 (whatever HPO found)
optimizer_config:
  lr: 3.42e-1            # Was: 1.e-1
scheduler_config:
  total_steps: 150       # Now synced to epochs
  min_lr: 2.15e-5        # Was: 1.e-6

# Enable for long training
early_stopping_config:
  enabled: true
  patience: 30
  mode: min
  min_delta: 0.0001

checkpoint_config:
  enabled: true
  save_every_n_epochs: 25
  keep_last_k: 3
  save_best: true
  monitor: val_loss
  mode: min
```

Then validate and run:

```bash
python -m cli validate configs/SolarFlux_v0.3/fluxnet_best.yaml
python -m cli preflight configs/SolarFlux_v0.3/fluxnet_best.yaml
python -m cli train configs/SolarFlux_v0.3/fluxnet_best.yaml --device cuda:0
```

With 5 seeds and 150 epochs, this is a long run. Use pueue.

---

## Phase 7: Analysis

After training completes, check the run directories and analyze results:

```bash
ls runs/SolarFlux_v0.3_FluxNet/     # One subdirectory per group name
python -m cli analyze               # Interactive model loading and evaluation
```

The `analyze` command guides you through selecting a project, group, and seed interactively, then loads the saved model and evaluates it on the validation set.

The `runs/` directory structure after training:

```
runs/
└── SolarFlux_v0.3_FluxNet/
    └── MLP_n_64_l_5_SPlus_lr_3.42e-01.../
        ├── config.yaml              # Exact config used for this group
        ├── 58/
        │   └── model.pt             # Trained weights for seed 58
        ├── 89/
        │   └── model.pt
        └── ...
```

---

## What's Next

- Deep dive into configs → [Chapter 2: Configuration Deep Dive](02_config.html)
- Understanding callbacks → [Chapter 3: Callback System](03_callbacks.html)
- HPO strategies → [Chapter 4: Hyperparameter Optimization](04_hpo.html)
- Adding custom models/data → [Chapter 5: Customization Guide](05_customization.html)
