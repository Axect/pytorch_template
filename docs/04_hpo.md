---
title: Hyperparameter Optimization
nav_order: 4
---

# Hyperparameter Optimization

This chapter explains how to run HPO, how to design a good search space, and — most importantly — **why** the choices made here work.

---

## The HPO Philosophy

The temptation in HPO is to tune everything. Resist it.

The guiding principle here is: **fix capacity, search expressiveness.** In the default MLP architecture, `nodes` controls how wide the network is — that is its capacity to store information. `layers` controls how many nonlinear transformations it applies — that is its expressiveness. These two parameters have very different roles.

Fixing `nodes` and tuning `layers` makes HPO trials comparable to each other: every trial has the same number of parameters per layer, so differences in validation loss are due to depth (expressiveness), not raw parameter count. If you tune both simultaneously, a shallow-but-wide network and a deep-but-narrow network can reach the same loss for completely different reasons, and the optimizer cannot learn a useful signal about either dimension.

The same reasoning applies to `total_steps` and `upper_bound` in the scheduler: these define the shape of the learning rate schedule and should remain fixed relative to the training duration. Let HPO search for the right learning rate magnitude and floor (`lr`, `min_lr`) rather than the schedule geometry.

**Rule of thumb:** tune at most 3–5 parameters per HPO run. More parameters than that causes combinatorial explosion — you need exponentially more trials to cover the space.

---

## Search Space Design

The search space is defined in a separate YAML file (e.g., `configs/optimize_template.yaml`). It is grouped by config category (`net_config`, `optimizer_config`, `scheduler_config`) and each parameter has a type and bounds.

**Supported types:**

| Type | Required fields | Optional fields | Notes |
|---|---|---|---|
| `int` | `min`, `max` | `step` | Discrete integer range |
| `float` | `min`, `max` | `log: true` | Continuous; use `log` for orders-of-magnitude ranges |
| `categorical` | `choices` | — | Explicit list; use for architecture choices |

Here is a complete, well-designed search space for the `SPlus` + `ExpHyperbolicLR` combination (HPO run config):

```yaml
# configs/hpo.yaml
study_name: MyProject_HPO
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
    n_warmup_epochs: 10
    top_k: 10
    target_epoch: 10  # match HPO epochs

search_space:
  net_config:
    layers:
      type: int
      min: 3
      max: 6
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

**What to tune, and what not to:**

- `layers`: tune. Depth has a large effect on whether the network can capture the target function's complexity.
- `lr`: tune, with a wide log-scale range. SPlus has internal eigenvalue-based scaling (see next section), so `1e-3` to `1e+0` is the correct range — not `1e-5` to `1e-2`.
- `min_lr`: tune. The learning rate floor determines how much fine-tuning the scheduler provides at convergence.
- `nodes`: **do not tune**. Fix it based on your memory budget and desired model size.
- `upper_bound`: **do not tune**. This is a schedule geometry parameter. Set it once based on how quickly you expect the model to converge.
- `total_steps`: **do not tune**. Fix it to match the number of HPO epochs (10).

---

## SPlus + ExpHyperbolicLR: Why 10 Epochs Is Enough

This section addresses the most important practical insight in this template.

### SPlus: wide lr range is correct

SPlus (from `pytorch_optimizer`) is a second-order-inspired optimizer that applies an internal eigenvalue-based scaling to each parameter update. The effective step size is much smaller than the nominal `lr` you pass to it. A `lr=0.1` with SPlus behaves more like `lr=1e-4` with Adam in terms of actual parameter movement.

This is why the HPO range for `lr` must be `[1e-3, 1e+0]` (log scale). If you narrow it to `[1e-5, 1e-2]` — a reasonable range for Adam — you will miss the effective region entirely and HPO will return a suboptimal result. Do not change this range.

### ExpHyperbolicLR: epoch-insensitive ordering

`ExpHyperbolicLR` (from `pytorch_scheduler`) has an unusual property: **the relative ordering of two configurations by validation loss after 10 epochs is the same as their ordering after 150 epochs**, provided the schedule shapes are comparable.

This happens because the scheduler's decay is hyperbolic — it drops quickly early and then flattens. The first 10 epochs capture the important part of the loss descent. After that, additional epochs refine the result but do not change which configuration wins.

This means you can set `epochs: 10` in the HPO run config and get reliable results that transfer to the full 150-epoch run.

**The critical detail:** set `total_steps: 10` (matching HPO epochs) in the HPO config, but `upper_bound` stays fixed. The `target_epoch` in PFLPruner should also match HPO epochs. Here is how the two configs relate:

| Parameter | HPO run config | Best run config |
|---|---|---|
| `epochs` | 10 | 150 |
| `scheduler_config.total_steps` | 10 | 150 |
| `scheduler_config.upper_bound` | (fixed, e.g. 250) | same |
| `seeds` | `[42]` | `[58, 89, 231, 928, 814]` |

The `upper_bound` stays the same in both runs because it controls the asymptotic behavior of the schedule, not the number of steps.

---

## PFL Pruner

Running every HPO trial to completion is wasteful. Most bad configurations reveal themselves early. The `PFLPruner` (Predicted Final Loss Pruner) exploits this by predicting where a trial will end up and stopping it early if the prediction is poor.

### How it works

1. **Startup phase**: the first `n_startup_trials` trials run to completion unconditionally. These populate the "top-K" reference set.
2. **Warmup phase**: within each trial, the first `n_warmup_epochs` epochs are never pruned. This gives the loss enough data to fit.
3. **Prediction**: after the warmup period, PFLPruner fits an exponential decay $L(t) = A e^{Kt}$ to the validation loss history using `numpy.polyfit` on the log-transformed losses. It then extrapolates to `target_epoch` to get a Predicted Final Loss (PFL).
4. **Pruning decision**: if the current trial's PFL is worse than the worst PFL in the top-K set, the trial is pruned.

The PFL is computed as $-\log_{10}(L_{\text{predicted}})$, so higher PFL is better (lower predicted loss). A trial is pruned if its PFL falls below the minimum PFL of the top-K reference trials.

The result: **approximately 40% GPU time savings** compared to running all trials to completion, with no loss in the quality of the best trial found.

### Configuration parameters

```yaml
pruner:
  name: pruner.PFLPruner
  kwargs:
    n_startup_trials: 10   # Run these unconditionally to build a reference
    n_warmup_epochs: 10    # Within each trial, never prune before this epoch
    top_k: 10              # Size of the reference set
    target_epoch: 10       # Predict loss at this epoch (match HPO epochs)
```

Setting `target_epoch` to match `epochs` in the run config ensures the pruner predicts to the actual end of training, not beyond it.

---

## Reading HPO Results with hpo-report

After the HPO run completes, a SQLite database file is created (e.g., `MyProject_Opt.db`). Use `hpo-report` to analyze it:

```bash
# Auto-detect the .db file if only one exists
python cli.py hpo-report

# Explicitly specify database and optimization config for boundary warnings
python cli.py hpo-report --db MyProject_Opt.db --opt-config configs/hpo.yaml --top-k 5
```

Example output:

```
Study: MyProject_HPO (MyProject_Opt.db)
Trials: 50 total, 28 completed, 19 pruned, 3 failed

Best Trial #17
  Value: 0.000312
  Group: MLP_n_64_l_4_SP_l_3.4210e-02_EHLRS_t_10_u_250_m_1.2345e-06[17]
  net_config_layers: 4
  optimizer_config_lr: 0.03421
  scheduler_config_min_lr: 1.23e-06

┌─────────────────────────────────────────────────────┐
│               Parameter Importance                  │
├─────────────────────────┬───────────────────────────┤
│ optimizer_config_lr     │ 0.6231 ███████████████████ │
│ net_config_layers       │ 0.2847 █████████           │
│ scheduler_config_min_lr │ 0.0922 ███                 │
└─────────────────────────┴───────────────────────────┘

Boundary Warnings:
  scheduler_config_min_lr=1.23e-06 at LOWER boundary [1e-07, 1e-03]

┌──────────────────────────────────────────────────┐
│                  Top 5 Trials                    │
├────┬──────────┬─────────┬───────────┬────────────┤
│ #  │ Value    │ layers  │ lr        │ min_lr     │
├────┼──────────┼─────────┼───────────┼────────────┤
│ 17 │ 0.000312 │ 4       │ 3.4210e-2 │ 1.2345e-6  │
│ 23 │ 0.000389 │ 4       │ 2.9876e-2 │ 8.9123e-7  │
│  8 │ 0.000401 │ 5       │ 4.1234e-2 │ 1.5678e-6  │
│ 31 │ 0.000445 │ 3       │ 3.8765e-2 │ 2.3456e-6  │
│ 12 │ 0.000512 │ 4       │ 5.6789e-2 │ 9.8765e-7  │
└────┴──────────┴─────────┴───────────┴────────────┘
```

**How to read each section:**

- **Stats**: the ratio of pruned to completed trials shows pruner effectiveness. A 40–60% prune rate is healthy. Over 80% may indicate the search space is too wide.
- **Best trial**: the winning configuration. Use this as the basis for `best.yaml`.
- **Parameter importance** (fANOVA): which parameters explain the most variance in trial outcomes. If one parameter dominates (>0.7), concentrate your next search around it. If all parameters are roughly equal, the search space is well-balanced.
- **Boundary warnings**: if the best value is within 5% of a search bound, the optimum may lie outside your current range. Widen the bound in that direction and re-run HPO.
- **Top-K table**: compare the top trials. If they cluster tightly in one parameter (e.g., all top trials have `layers=4`), that parameter is well-determined. Use it as a fixed value in future runs.

---

## Monitoring HPO in Real-time

While HPO runs, you can watch its progress in a live terminal dashboard — the same Rust TUI used for training monitoring, now with an `--hpo` mode that reads the Optuna SQLite database directly.

```bash
# In a separate terminal, while HPO is running (or after it finishes)
python -m cli monitor --hpo
```

The command auto-detects the `.db` file. If multiple databases exist, it presents a selection menu.

### Tabs

The HPO monitor has four tabs, navigable with `←→` or `Tab`:

| Tab | What it shows |
|-----|--------------|
| **Overview** | Trial scatter plot (yellow dots) overlaid with best-so-far convergence line (green). Status bar shows completed/pruned/failed/running counts. |
| **Parameters** | Grid of scatter plots — one per search-space parameter. X-axis is parameter value, Y-axis is objective. Points colored by trial state (green=complete, yellow=pruned, red=failed). Reveals which parameter regions yield good results. |
| **Best Trial** | Training curves (loss, lr, gradient norm) of the current best trial, identical to the training monitor view. Automatically switches when a new best trial appears. |
| **Trials** | Sortable table of all trials with parameters and duration. Select a row with `↑↓` and press `Enter` to see that trial's full training curves. `Esc` returns to the table. |

### Key bindings

| Key | Action | Where |
|-----|--------|-------|
| `q` | Quit | All tabs |
| `←→` / `Tab` | Switch tabs | All tabs |
| `l` | Toggle Y-axis log scale | All tabs |
| `x` | Toggle X-axis log scale | Parameters tab |
| `+` / `=` | Zoom in on Y axis | Overview, Parameters, Best Trial |
| `-` | Zoom out on Y axis | Overview, Parameters, Best Trial |
| `↑` / `↓` | Pan Y axis | Overview, Parameters, Best Trial |
| `r` | Reset Y axis to auto range | All tabs |
| `↑` / `↓` | Select trial row | Trials table |
| `Enter` | View selected trial's training curves | Trials table |
| `Esc` | Return to table | Trials detail view |

### When to use

- **During HPO**: watch the Overview scatter to see if trials are converging. If the scatter is narrowing around a region, the sampler is learning. If it stays spread out, your search space may be too large or your budget too small.
- **After HPO**: switch to the Parameters tab to visually confirm which parameters matter. The `hpo-report` command gives you numbers (fANOVA importance); the scatter plots give you intuition about the landscape shape.
- **Debugging bad runs**: use the Trials tab to inspect individual trials. Select a pruned trial to see at which epoch the loss diverged, or a failed trial to check for NaN patterns.

---

## From HPO to best.yaml

Once HPO completes and you have analyzed the results, create `best.yaml` for the final multi-seed run.

Start from your HPO run config and make the following changes:

```yaml
# best.yaml — final training run after HPO
project: MyProject_Best
device: cuda:0
net: model.MLP
optimizer: pytorch_optimizer.SPlus
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
criterion: torch.nn.MSELoss
criterion_config: {}
data: util.load_data

# 1. Increase epochs from 10 to full training duration
epochs: 150

# 2. Expand seeds for statistical robustness
seeds: [58, 89, 231, 928, 814]

batch_size: 256

net_config:
  nodes: 64            # unchanged — fixed capacity
  layers: 4            # from HPO best trial

optimizer_config:
  lr: 3.421e-2         # from HPO best trial
  eps: 1.e-10

scheduler_config:
  # 3. Set total_steps to match full epochs
  total_steps: 150
  upper_bound: 250     # unchanged — fixed geometry
  min_lr: 1.234e-6     # from HPO best trial

# 4. Enable early stopping and checkpointing for the full run
early_stopping_config:
  enabled: true
  patience: 20
  mode: min
  min_delta: 0.0001

checkpoint_config:
  enabled: true
  save_every_n_epochs: 10
  keep_last_k: 3
  save_best: true
  monitor: val_loss
  mode: min
```

Then run:

```bash
python cli.py train best.yaml
```

The five seeds will each train independently and log to W&B under the same group name, giving you a distribution of final validation losses to report mean ± std.

{: .note }
If the boundary warning showed that `min_lr` was at the lower edge of its search range, widen the range to `[1e-8, 1e-3]` and run HPO again before creating `best.yaml`.
