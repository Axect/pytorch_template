# PyTorch Template

[English](README.md) | [한글](README_KR.md)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.7+](https://img.shields.io/badge/PyTorch-2.7+-ee4c2c.svg)](https://pytorch.org/)
[![Optuna](https://img.shields.io/badge/Optuna-integrated-blue.svg)](https://optuna.org/)
[![W&B](https://img.shields.io/badge/Weights_%26_Biases-supported-FFBE00.svg)](https://wandb.ai/)

A modular, extensible template for PyTorch-based deep learning research. Define your entire experiment — model, optimizer, scheduler, loss function, callbacks — in a single YAML file and run it with one command.

## Key Features

- **YAML-Driven Configuration** — All experiment settings managed in YAML. Frozen, validated configs prevent silent misconfiguration.
- **Callback-Based Training** — Extensible training loop with priority-ordered callbacks. Add behaviors (logging, checkpointing, early stopping) without modifying core code.
- **Configurable Loss & Metrics** — Swap loss functions via YAML (`torch.nn.CrossEntropyLoss`, custom losses). Built-in metric registry (MSE, MAE, R2) with importlib extension.
- **Checkpoint & Resume** — Full state checkpointing (model, optimizer, scheduler, RNG states) with multi-seed resume via `SeedManifest`.
- **Run Provenance** — Automatic capture of Python/PyTorch/CUDA versions, GPU info, git commit, and environment variables per run.
- **Hyperparameter Optimization** — Optuna integration with custom PFL pruner and deep-merge config overrides.
- **Experiment Tracking** — Seamless Weights & Biases logging via callback.
- **CLI** — `typer`-based CLI with `train`, `validate`, `preview`, `doctor`, and `analyze` subcommands.
- **Reproducibility** — Deterministic seed management across Python, NumPy, and PyTorch.

## Callback Architecture

The training loop emits events at defined hook points. Each concern (logging, early stopping, checkpointing) is an independent, priority-ordered callback:

![Callback Execution Flow](assets/callback_flow.png)

| Callback | Priority | Hook | Purpose |
|----------|----------|------|---------|
| `NaNDetectionCallback` | 5 | `on_epoch_end` | Detect NaN loss, signal stop |
| `OptimizerModeCallback` | 10 | `on_train_epoch_begin`, `on_val_begin` | SPlus/ScheduleFree train/eval mode |
| `LossPredictionCallback` | 70 | `on_val_end` | Predict final loss for early pruning |
| `WandbLoggingCallback` | 80 | `on_epoch_end` | Log metrics to W&B |
| `PrunerCallback` | 85 | `on_val_end` | Report to Optuna pruner |
| `EarlyStoppingCallback` | 90 | `on_val_end` | Monitor metric, signal stop |
| `CheckpointCallback` | 95 | `on_epoch_end` | Save periodic/best checkpoints |

Adding custom behavior is as simple as subclassing `TrainingCallback` and adding it to the callback list — zero changes to the training loop.

## Quick Start

1.  **Clone:**
    ```bash
    git clone https://github.com/<your-username>/<your-repo>.git
    cd <your-repo>
    ```

2.  **Install dependencies** ([uv](https://github.com/astral-sh/uv) recommended):
    ```bash
    uv venv && source .venv/bin/activate
    uv pip install -U torch wandb rich beaupy numpy optuna matplotlib scienceplots typer tqdm pyyaml pytorch-optimizer pytorch-scheduler
    ```

3.  **Validate your setup:**
    ```bash
    python -m cli doctor
    ```

4.  **Preview a config** (no training, just inspect):
    ```bash
    python -m cli preview configs/run_template.yaml
    ```

5.  **Train:**
    ```bash
    python -m cli train configs/run_template.yaml
    # Or with device override:
    python -m cli train configs/run_template.yaml --device cpu
    ```

6.  **Hyperparameter optimization:**
    ```bash
    python -m cli train configs/run_template.yaml --optimize-config configs/optimize_template.yaml
    ```

7.  **Analyze results:**
    ```bash
    python -m cli analyze
    # Or non-interactive:
    python -m cli analyze --project MyProject --group MyGroup --seed 42
    ```

> **Legacy CLI**: `python main.py --run_config configs/run_template.yaml` still works for backward compatibility.

## Project Structure

```
pytorch_template/
├── cli.py                 # Typer CLI entrypoint (train, validate, preview, doctor, analyze)
├── main.py                # Legacy argparse entrypoint
├── config.py              # RunConfig (frozen, validated) + OptimizeConfig
├── util.py                # Trainer, run(), data loading, analysis helpers
├── callbacks.py           # Callback system (8 built-in callbacks + CallbackRunner)
├── metrics.py             # Metric registry (MSE, MAE, R2 + importlib extension)
├── checkpoint.py          # CheckpointManager + SeedManifest
├── provenance.py          # Environment capture + config hashing
├── model.py               # Model architectures (MLP)
├── pruner.py              # PFL pruner for Optuna
├── configs/
│   ├── run_template.yaml
│   └── optimize_template.yaml
├── recipes/
│   ├── regression/        # Sine wave regression (MLP + MSELoss)
│   └── classification/    # FashionMNIST classification (CNN + CrossEntropyLoss)
├── tests/                 # 36 unit tests
└── runs/                  # Experiment outputs (auto-created)
```

## Configuration

### Run Configuration (`run_template.yaml`)

```yaml
project: PyTorch_Template
device: cuda:0
net: model.MLP
optimizer: pytorch_optimizer.SPlus
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
criterion: torch.nn.MSELoss          # Any loss function via importlib
criterion_config: {}                  # Arguments for criterion constructor
epochs: 50
batch_size: 256
seeds: [89, 231, 928, 814, 269]
net_config:
  nodes: 64
  layers: 4
optimizer_config:
  lr: 1.e-3
  eps: 1.e-10
scheduler_config:
  total_steps: 50
  upper_bound: 250
  min_lr: 1.e-5
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

**Key fields:**

| Field | Description |
|-------|-------------|
| `net` | Model class path in `module.Class` format |
| `optimizer` | Optimizer class path (supports `torch.optim.*`, `pytorch_optimizer.*`, custom) |
| `scheduler` | Scheduler class path (supports `torch.optim.lr_scheduler.*`, `pytorch_scheduler.*`, custom) |
| `criterion` | Loss function class path (e.g., `torch.nn.MSELoss`, `torch.nn.CrossEntropyLoss`) |
| `criterion_config` | Arguments passed to criterion constructor |
| `seeds` | List of random seeds — each seed is a separate training run |
| `checkpoint_config` | Periodic/best checkpoint saving with configurable policy |

All module paths are resolved via `importlib` at runtime. The config is **frozen** after construction — use `config.with_overrides(field=value)` to create modified copies.

### Optimization Configuration

See [`configs/optimize_template.yaml`](configs/optimize_template.yaml) for the full template. Key sections: `search_space`, `sampler`, `pruner`.

## Customization

### Adding Custom Models

Create a model class in `model.py` or a new file. The constructor must accept `(hparams: dict, device: str)`:

```python
# my_model.py
class MyTransformer(nn.Module):
    def __init__(self, hparams, device="cpu"):
        super().__init__()
        # hparams comes from net_config in YAML
        ...
```

```yaml
net: my_model.MyTransformer
net_config:
  d_model: 256
  nhead: 8
```

### Adding Custom Callbacks

Subclass `TrainingCallback` and override hook methods:

```python
# my_callbacks.py
from callbacks import TrainingCallback

class GradientClipCallback(TrainingCallback):
    priority = 15  # Run early, after OptimizerMode

    def __init__(self, max_norm=1.0):
        self.max_norm = max_norm

    def on_train_step_end(self, trainer, batch_idx, loss, **kwargs):
        torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), self.max_norm)
```

Then add it to the callback list in your training script or extend `run()`.

### Adding Custom Metrics

Register built-in names or importlib paths:

```python
from metrics import MetricRegistry

registry = MetricRegistry(["mse", "mae", "r2", "my_module.MyCustomMetric"])
results = registry.compute(y_pred, y_true)
# {"mse": 0.012, "mae": 0.089, "r2": 0.95, "my_custom_metric": ...}
```

### Switching Loss Functions

Change one line in YAML — no code changes:

```yaml
# Regression
criterion: torch.nn.MSELoss

# Classification
criterion: torch.nn.CrossEntropyLoss

# Custom
criterion: my_losses.FocalLoss
criterion_config:
  gamma: 2.0
  alpha: 0.25
```

### Using Different Schedulers

```yaml
# Built-in PyTorch
scheduler: torch.optim.lr_scheduler.CosineAnnealingLR
scheduler_config:
  T_max: 50
  eta_min: 1.e-5

# ExpHyperbolicLR (via pytorch-scheduler)
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
scheduler_config:
  total_steps: 50
  upper_bound: 250
  min_lr: 1.e-5
```

### Customizing Data Loading

Modify `load_data()` in `util.py` to return your `(train_dataset, val_dataset)`. See [`recipes/`](recipes/) for examples (regression + classification).

## Example Recipes

| Recipe | Task | Model | Loss | Config |
|--------|------|-------|------|--------|
| [`recipes/regression/`](recipes/regression/) | Sine wave fitting | MLP (64 nodes, 4 layers) | MSELoss | [config.yaml](recipes/regression/config.yaml) |
| [`recipes/classification/`](recipes/classification/) | FashionMNIST | SimpleCNN (32 channels) | CrossEntropyLoss | [config.yaml](recipes/classification/config.yaml) |

```bash
python -m cli train recipes/regression/config.yaml --device cpu
```

## CLI Reference

| Command | Description |
|---------|-------------|
| `python -m cli train <config> [--device] [--optimize-config]` | Train model(s) with optional HPO |
| `python -m cli validate <config>` | Validate config without training |
| `python -m cli preview <config>` | Show model architecture and config summary |
| `python -m cli doctor` | Check Python, PyTorch, CUDA, W&B, packages |
| `python -m cli analyze [--project] [--group] [--seed] [--device]` | Analyze trained models |

## Documentation

For a deeper dive into components and customization:

* **[Project Documentation](https://axect.github.io/pytorch_template)** (Generated by [Tutorial-Codebase-Knowledge](https://github.com/The-Pocket/Tutorial-Codebase-Knowledge))

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgments

- [pytorch-optimizer](https://github.com/kozistr/pytorch_optimizer) for optimizers including SPlus.
- [pytorch-scheduler](https://github.com/Axect/pytorch_scheduler) for learning rate schedulers including ExpHyperbolicLR.

## Appendix

<details>
<summary><strong>PFL (Predicted Final Loss) Pruner</strong></summary>

### Overview

The PFL pruner (`pruner.PFLPruner`) predicts the final performance of a training run based on early-stage metrics, pruning unpromising Optuna trials early.

### Configuration

```yaml
pruner:
  name: pruner.PFLPruner
  kwargs:
    n_startup_trials: 10
    n_warmup_epochs: 10
    top_k: 10
    target_epoch: 50
```

### How It Works

1. The first `n_startup_trials` run to completion to establish baseline performance.
2. For subsequent trials, pruning is considered only after `n_warmup_epochs`.
3. The pruner predicts final loss from the current loss history using exponential curve fitting.
4. If the predicted final loss is worse than the top-k completed trials, the trial is pruned.
5. Supports multi-seed runs by averaging metrics across seeds.

</details>
