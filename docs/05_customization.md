---
title: Customization Guide
nav_order: 5
---

# Customization Guide

This template is designed to be extended without touching the training loop. You provide the model, data, loss, and metrics; the framework handles the rest. This chapter explains exactly how each extension point works.

---

## Adding a Custom Model

Every model in this template must satisfy a single contract: its `__init__` takes two arguments — `hparams: dict` and `device: str`.

```python
# recipes/my_task/model.py
import torch.nn as nn

class MyNetwork(nn.Module):
    def __init__(self, hparams: dict, device: str = "cpu"):
        super().__init__()
        self.hparams = hparams
        self.device = device

        # Read architecture parameters from hparams
        hidden = hparams["hidden"]
        n_layers = hparams["layers"]
        input_dim = hparams.get("input_dim", 1)
        output_dim = hparams.get("output_dim", 1)

        layers = [nn.Linear(input_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
```

The `hparams` dict comes directly from `net_config` in the YAML. There is no registration step — the class is loaded by importlib from the dotted path you provide in the config:

```yaml
# recipes/my_task/config.yaml
net: recipes.my_task.model.MyNetwork

net_config:
  hidden: 128
  layers: 5
  input_dim: 3
  output_dim: 1
```

`RunConfig.create_model()` resolves `recipes.my_task.model.MyNetwork`, instantiates it as `MyNetwork(net_config, device=device)`, and returns it. No other changes are needed.

**Practical advice:** put all architecture hyperparameters in `net_config`. Do not hard-code sizes inside `__init__`. This makes HPO possible: you can add `hidden` or `layers` to the search space and the model adapts automatically.

---

## Adding a Custom Data Loader

The `data` field in the config is a dotted path to a function that returns `(train_dataset, val_dataset)` — two `torch.utils.data.Dataset` objects. That is the entire contract.

```python
# recipes/my_task/data.py
import torch
from torch.utils.data import TensorDataset, random_split

def load_data(n: int = 5000, split_ratio: float = 0.8, seed: int = 42):
    """Load or generate data for my task.

    Returns:
        (train_dataset, val_dataset): two PyTorch Dataset objects.
    """
    torch.manual_seed(seed)

    # Generate or load your data here
    x = torch.randn(n, 3)
    y = torch.sin(x[:, 0:1]) + 0.5 * x[:, 1:2] ** 2

    dataset = TensorDataset(x, y)

    train_size = int(n * split_ratio)
    val_size = n - train_size
    generator = torch.Generator().manual_seed(seed)
    return random_split(dataset, [train_size, val_size], generator=generator)
```

Reference it in the config:

```yaml
data: recipes.my_task.data.load_data
```

`RunConfig.load_data()` calls this function with no arguments, so all parameters must have defaults. If you need to pass arguments (e.g., a file path), use a closure or a partial function defined at module level:

```python
# Wrapping a data loader that needs external arguments
import functools

DATA_PATH = "/data/my_dataset.csv"

def _load_from_path(path, split_ratio=0.8, seed=42):
    # ... load from CSV
    pass

load_data = functools.partial(_load_from_path, DATA_PATH)
```

The returned datasets are wrapped in `DataLoader` automatically by the training loop with the `batch_size` and `shuffle` settings from the config.

---

## Switching Loss Functions

For standard PyTorch losses, only the YAML needs to change — no Python code required:

```yaml
# Regression
criterion: torch.nn.MSELoss
criterion_config: {}

# Huber loss (robust to outliers)
criterion: torch.nn.HuberLoss
criterion_config:
  delta: 1.0

# Classification
criterion: torch.nn.CrossEntropyLoss
criterion_config: {}

# Weighted cross-entropy
criterion: torch.nn.CrossEntropyLoss
criterion_config:
  weight: [1.0, 2.0, 1.5]   # class weights
```

`criterion_config` is passed as keyword arguments to the criterion class constructor. `RunConfig.create_criterion()` resolves the path and instantiates the class.

For a custom loss, write a class with a `forward` method and register it via dotted path:

```python
# recipes/my_task/loss.py
import torch
import torch.nn as nn

class RelativeMSELoss(nn.Module):
    """MSE normalized by the target magnitude."""
    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return ((y_pred - y_true) ** 2 / (y_true.abs() + self.eps)).mean()
```

```yaml
criterion: recipes.my_task.loss.RelativeMSELoss
criterion_config:
  eps: 1.e-8
```

---

## Adding Custom Metrics

The `MetricRegistry` in `metrics.py` computes additional metrics alongside the training loss. Built-in options are `"mse"`, `"mae"`, and `"r2"`. To add your own, write a `Metric` subclass:

```python
# recipes/my_task/metrics.py
import torch
from metrics import Metric

class MAPEMetric(Metric):
    """Mean Absolute Percentage Error."""
    name = "mape"

    def __call__(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> float:
        eps = 1e-8
        return ((y_pred - y_true).abs() / (y_true.abs() + eps)).mean().item() * 100.0
```

Then reference it in any code that builds a `MetricRegistry`:

```python
from metrics import MetricRegistry

registry = MetricRegistry([
    "mse",                           # built-in
    "r2",                            # built-in
    "recipes.my_task.metrics.MAPEMetric",  # custom via importlib path
])

results = registry.compute(y_pred, y_true)
# results = {"mse": 0.012, "r2": 0.987, "mape": 3.45}
```

Metrics are computed on the validation set at the end of each epoch. The `name` attribute on the class is what appears as the key in the results dict and in W&B logs.

---

## Adding Custom Callbacks

Callbacks let you inject behavior at specific points in the training loop without modifying `Trainer`. The available hooks are:

| Hook | When it fires | Key arguments |
|---|---|---|
| `on_train_begin` | Before the first epoch | — |
| `on_train_epoch_begin` | Start of each epoch | `epoch` |
| `on_train_step_end` | After each batch | `batch_idx`, `loss` |
| `on_val_begin` | Before validation | `epoch` |
| `on_val_end` | After validation | `epoch`, `val_loss`, `metrics` |
| `on_epoch_end` | End of each epoch | `epoch`, `train_loss`, `val_loss`, `metrics` |
| `on_train_end` | After the last epoch | — |

The `priority` attribute controls execution order: **lower number runs first**. Built-in callbacks use priorities 5–95.

Here is a complete example that logs to a CSV file:

```python
# recipes/my_task/callbacks.py
import csv
from callbacks import TrainingCallback

class CSVLoggerCallback(TrainingCallback):
    """Writes train_loss and val_loss to a CSV file each epoch."""
    priority = 60  # Run after loss prediction (70) but fine here too

    def __init__(self, filepath: str):
        self.filepath = filepath
        self._file = None
        self._writer = None

    def on_train_begin(self, trainer, **kwargs):
        self._file = open(self.filepath, "w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(["epoch", "train_loss", "val_loss"])

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        self._writer.writerow([epoch, train_loss, val_loss])
        self._file.flush()

    def on_train_end(self, trainer, **kwargs):
        if self._file:
            self._file.close()
```

To use it, build the callback runner manually and pass it to `Trainer`:

```python
from callbacks import CallbackRunner, OptimizerModeCallback, WandbLoggingCallback
from recipes.my_task.callbacks import CSVLoggerCallback
from util import Trainer

callback_runner = CallbackRunner([
    OptimizerModeCallback(),
    WandbLoggingCallback(),
    CSVLoggerCallback("training_log.csv"),
])

trainer = Trainer(model, optimizer, scheduler, criterion,
                  callbacks=callback_runner, device=device)
trainer.train(dl_train, dl_val, epochs=config.epochs)
```

Callbacks communicate with the trainer through `trainer` attributes. To pass data out of a callback (for use by another callback or by the caller), set a custom attribute on `trainer`:

```python
def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
    trainer._my_custom_metric = compute_something(train_loss, val_loss)
```

Other callbacks (or post-training code) can then read `trainer._my_custom_metric`. This is the same pattern used by `LossPredictionCallback` (`trainer._loss_prediction`) and `GradientMonitorCallback` (`trainer._max_grad_norm`).

---

## Recipe Pattern

A recipe is a self-contained experiment directory. It bundles a model, a data loader, and a config together so you can clone it and adapt it for a new task without touching anything outside the directory.

```
recipes/
  my_task/
    __init__.py      # empty — makes this a Python package
    config.yaml      # full RunConfig referencing this recipe's own files
    model.py         # MyNetwork class
    data.py          # load_data function
```

The key property: `config.yaml` uses dotted paths that point back into the recipe itself:

```yaml
# recipes/my_task/config.yaml
project: MyTask
device: cuda:0
net: recipes.my_task.model.MyNetwork
data: recipes.my_task.data.load_data
optimizer: pytorch_optimizer.SPlus
scheduler: pytorch_scheduler.ExpHyperbolicLRScheduler
criterion: torch.nn.MSELoss
criterion_config: {}
epochs: 150
batch_size: 256
seeds: [42]
net_config:
  hidden: 128
  layers: 4
optimizer_config:
  lr: 1.e-2
  eps: 1.e-10
scheduler_config:
  total_steps: 150
  upper_bound: 300
  min_lr: 1.e-6
early_stopping_config:
  enabled: false
  patience: 20
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

To run it:

```bash
python cli.py train recipes/my_task/config.yaml
```

To start a new recipe, copy an existing one and change the dotted paths:

```bash
cp -r recipes/regression/ recipes/my_task/
# then edit recipes/my_task/config.yaml, model.py, data.py
```

**Why recipes matter:** when you have multiple experiments (regression, classification, a custom dataset), each lives in its own directory with its own config and code. You do not need to maintain a central model registry or a switch statement in `model.py`. The importlib loading mechanism handles any valid Python dotted path, so adding a new recipe requires no changes to the core framework.

The two built-in recipes demonstrate the pattern concretely:

- `recipes/regression/`: sine wave fitting with a 1D MLP, MSE loss, synthetic data.
- `recipes/classification/`: FashionMNIST with a simple CNN, cross-entropy loss, `torchvision` download.

Use them as starting points and replace the model and data loader for your task.
