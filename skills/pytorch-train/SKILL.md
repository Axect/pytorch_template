---
name: pytorch-train
description: >
  Guide for training PyTorch models using this template's config-driven pipeline.
  Use this skill whenever the user wants to: train a model, create experiment configs
  (run.yaml / opt.yaml / best.yaml), run hyperparameter optimization (HPO) with Optuna,
  extract best parameters from an HPO study, set up a new experiment version,
  or scaffold a new model/data recipe. Triggers on: "train", "HPO", "optimize",
  "create config", "best params", "new experiment", "run training", "set up model",
  "preflight check", "HPO analysis".
allowed-tools: Bash, Write, Read, Glob, Grep, Edit
---

# pytorch-train

Train models through this template's full pipeline: config creation, HPO, best-param extraction, and final training.

## Usage

```
/pytorch-train [phase] [options]
```

**Phases** (auto-detected from user intent if omitted):
- `full` — Complete pipeline (Phases 1-5)
- `config` — Create config files only (Phases 1-2)
- `hpo` — Run HPO (Phase 3, assumes configs exist)
- `extract` — Extract best params from completed HPO (Phase 4)
- `train` — Run final training with best.yaml (Phase 5)

---

## Phase 1: Gather Requirements

Before creating any files, confirm these with the user:

| Item | Example | Notes |
|------|---------|-------|
| Project name | `SolarFlux`, `WavePredict` | Used in `project:` field and directory names |
| Version | `v0.3`, `v1.32` | Determines config subdirectory |
| Model name | `fluxnet`, `wavenet`, `mlp` | File prefix and `net:` path |
| Task type | regression / classification | Determines criterion, metric direction |
| Model module path | `model.MLP`, `recipes.regression.model.MLP` | Importlib path for `net:` field |
| net_config | `{nodes: 64, layers: 4}` | Architecture hyperparameters |
| Data loader | existing `load_data()` or needs new one | Recipe scaffolding if needed |
| Device | `cuda:0` (default if available) | GPU index |

### Task Type Defaults

| Setting | Regression | Classification |
|---------|-----------|----------------|
| `criterion` | `torch.nn.MSELoss` | `torch.nn.CrossEntropyLoss` |
| `criterion_config` | `{}` | `{}` |
| `metric` (HPO) | `val_loss` | `val_loss` |
| `direction` (HPO) | `minimize` | `minimize` |

---

## Pre-flight Check (Before HPO)

After creating configs, always run preflight before HPO:

```bash
python -m cli preflight configs/<DIR>/<model>_run.yaml --device cuda:0
```

Preflight runs 1 batch forward+backward and checks:
- Import paths and device availability
- Semantic validation (upper_bound >= total_steps, lr > 0, etc.)
- Model/optimizer/scheduler/criterion instantiation
- Data loading via the config's `data` field
- Forward pass (output shape vs target shape)
- Backward pass (gradient NaN/Inf detection, grad norm)
- GPU memory estimate

Fix any FAIL/WARN issues before proceeding to HPO. Use `--json` for machine-readable output.

---

## Phase 2: Create Config Files

### Naming Convention (CRITICAL)

```
configs/<MAIN_CONTRIBUTION>_v<VERSION>/<MODEL_NAME>_{run,opt,best}.yaml
```

Examples:
- `configs/SolarFlux_v0.3/fluxnet_run.yaml`
- `configs/SolarFlux_v0.3/fluxnet_opt.yaml`
- `configs/SolarFlux_v0.3/fluxnet_best.yaml`

The `project:` field follows: `<PROJECT>_v<VERSION>_<MODEL>` (e.g., `SolarFlux_v0.3_FluxNet`).

### 2a: run.yaml (HPO Base Config)

Key rules:
- `epochs: 10` — Default for HPO. Sufficient due to epoch-insensitive schedulers.
- `seeds: [42]` — Single seed during HPO for speed.
- `scheduler_config.total_steps: 150` — Fixed at full-training scale, NOT synced to HPO epochs.
- `scheduler_config.upper_bound: 300` — Must be >= total_steps.
- `scheduler_config.min_lr: 1.e-6` — Will be tuned by HPO.
- `data: <module.load_data>` — Data loader function path (default: `util.load_data`)

Read `references/config_templates.md` for full annotated templates.

### 2b: opt.yaml (HPO Search Config)

Key rules for SPlus + ExpHyperbolicLR combo:
- **SPlus lr range: `[1.e-3, 1.e+0]` (log scale)** — SPlus has internal eigenvalue-based scaling. The effective lr is much smaller than the nominal value. NEVER use standard ranges like `[1e-5, 1e-2]`.
- **HPO targets**: Only tune `layers`, `lr`, `min_lr`. Fix `nodes` (width) and `upper_bound`.
- `trials: 50` default (adjust 30-100 based on search space size).
- Pruner: `pruner.PFLPruner` with `target_epoch` matching run.yaml `epochs`.

### 2c: best.yaml (After HPO)

Created in Phase 4. Transformations from run.yaml:
- `epochs` → full training count (typically 150, task-dependent)
- `scheduler_config.total_steps` → match new epochs (e.g., 150)
- `seeds` → expand to multi-seed list (e.g., `[58, 89, 231, 928, 814]`)
- HPO-found params filled in from study results
- `project:` field updated (remove `_Opt` suffix if present)

---

## Phase 3: Run HPO

### Command

```bash
python -m cli train configs/<DIR>/<model>_run.yaml \
  --optimize-config configs/<DIR>/<model>_opt.yaml \
  --device cuda:0
```

### Long-running tasks: use pueue

```bash
pueue group add <PROJECT_NAME>
pueue add -g <PROJECT_NAME> -- bash -c \
  "cd $(pwd) && .venv/bin/python -m cli train configs/<DIR>/<model>_run.yaml \
   --optimize-config configs/<DIR>/<model>_opt.yaml --device cuda:0"
```

### Monitor

```bash
pueue status -g <PROJECT_NAME>
pueue log -g <PROJECT_NAME>
```

---

## Phase 4: Extract Best Parameters

Read `references/hpo_extraction.md` for the full extraction procedure.

Summary:
1. Load Optuna study from `<PROJECT>_Opt.db`
2. Get `study.best_trial.params`
3. Map flattened param names back to YAML structure
4. Copy run.yaml → best.yaml with overrides
5. Update epochs, total_steps, seeds

### Quick analysis with hpo-report

```bash
# Auto-detects DB and study if only one exists
python -m cli hpo-report

# Explicit
python -m cli hpo-report --db <PROJECT>_Opt.db --study-name <STUDY_NAME>

# With boundary check (warns if best param is at search space edge)
python -m cli hpo-report --db <PROJECT>_Opt.db --opt-config configs/<DIR>/<model>_opt.yaml

# JSON output for agent parsing
python -m cli hpo-report --json
```

The report includes:
- Completion stats (completed/pruned/failed)
- Best trial details (value, params, group_name)
- Parameter importance (fANOVA-based)
- Boundary warnings (suggests widening search space)
- Top-K trials comparison table

---

## Phase 5: Final Training

```bash
python -m cli train configs/<DIR>/<model>_best.yaml --device cuda:0
```

For long training, use pueue (same pattern as Phase 3).

After training completes, verify:
```bash
ls runs/<PROJECT>/                    # Check run directories
python -m cli analyze --project <PROJECT>  # Analyze results
```

---

## Domain Knowledge: SPlus + HyperbolicLR/ExpHyperbolicLR

### Why lr range is [1e-3, 1e+0] for SPlus

SPlus uses eigenvalue-based internal scaling. The nominal lr (what you set in config) is divided by an estimated spectral norm, making the effective step size much smaller. Standard lr ranges like [1e-5, 1e-2] result in extremely slow convergence with SPlus.

### Why total_steps != epochs in HPO

ExpHyperbolicLR / HyperbolicLR is **epoch-insensitive**: the learning rate curve shape is controlled by the `upper_bound / total_steps` ratio. Setting `total_steps=150` with `epochs=10` means you only traverse the first 10 steps of a 150-step schedule. This produces a reliable training signal for HPO because:
1. The lr trajectory is identical to full training (just truncated).
2. Relative ordering of trials is preserved even with short training.
3. After HPO, switching to `epochs=150` gives the full schedule without retuning.

### Parameter Reference

| pytorch_scheduler param | Equivalent in local projects | Description |
|------------------------|------------------------------|-------------|
| `total_steps` | `max_iter` | Total number of scheduler steps |
| `min_lr` | `infimum_lr` | Lower bound of learning rate |
| `upper_bound` | `upper_bound` | Hyperbolic curve parameter (must be >= total_steps) |

### Recommended Defaults

| Parameter | HPO run.yaml | Final best.yaml |
|-----------|-------------|-----------------|
| `epochs` | 10 (up to 30) | 150 (task-dependent) |
| `total_steps` | 150 | 150 (= epochs) |
| `upper_bound` | 300 | 300 |
| `min_lr` | 1.e-6 (HPO tunes this) | from HPO result |
| SPlus `lr` | 1.e-1 (HPO tunes this) | from HPO result |

---

## Training Diagnostics

Two callbacks are always active during training:

**GradientMonitorCallback** — Computes gradient L2 norm per step. Logs `max_grad_norm` per epoch to W&B. Warns if gradient norm exceeds 10,000 (potential exploding gradients).

**OverfitDetectionCallback** — Tracks train/val loss per epoch. After 5 warmup epochs, checks for sustained divergence (train_loss decreasing while val_loss increasing over 5 consecutive epochs). Logs `overfit_gap_ratio` to W&B when detected. Warning only — does not stop training.

Both are logged to W&B automatically. Check the W&B dashboard for `max_grad_norm` and `overfit_gap_ratio` metrics.

---

## Edge Cases

### Non-SPlus Optimizers (AdamW, Adam, etc.)

- lr search range: `[1.e-5, 1.e-2]` (standard log scale)
- No `OptimizerModeCallback` needed (no train/eval mode switching)
- Consider `CosineAnnealingLR` scheduler with `T_max` = epochs, `eta_min` = min_lr
- HPO epochs can stay at 10 but may need more (20-30) for reliable signal

### Adding a New Model to Existing Version

1. Read existing configs in `configs/<DIR>/` to infer naming pattern
2. Create new `<new_model>_run.yaml` and `<new_model>_opt.yaml`
3. Reuse the same project name prefix and version

### Re-running HPO with Different Search Space

1. Edit only `<model>_opt.yaml`
2. Change `study_name` to avoid conflicts with previous study
3. Previous Optuna DB is preserved (multiple studies can coexist)

### Recipe Scaffolding (New Model + Data)

If the user needs a new model or data loader:

```
recipes/<task_name>/
├── __init__.py
├── config.yaml      # Standalone config for quick testing
├── model.py         # Model class with __init__(hparams, device)
└── data.py          # load_data() returning (train_dataset, val_dataset)
```

Model must follow the signature: `__init__(self, hparams: dict, device: str)`.
Data loader must return `(train_dataset, val_dataset)` as PyTorch Dataset objects.

---

## Validation Checklist

Before running training, verify:
- [ ] `python -m cli validate <run.yaml>` passes
- [ ] `python -m cli preview <run.yaml>` shows correct architecture
- [ ] `python -m cli preflight <run.yaml>` all checks pass
- [ ] `upper_bound >= total_steps` in scheduler_config
- [ ] For SPlus: lr in [1e-3, 1e+0] range in opt.yaml search space
- [ ] `project:` field matches directory naming convention
- [ ] Config directory exists: `configs/<DIR>/`
- [ ] After HPO: `python -m cli hpo-report` shows no boundary warnings
