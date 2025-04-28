# PyTorch Template Project

[English](README.md) | [한글](README_KR.md)

This project provides a flexible template for PyTorch-based machine learning experiments.
It includes configuration management using YAML files, logging with Weights & Biases (wandb), hyperparameter optimization with Optuna, custom learning rate schedulers, advanced pruning techniques, and a modular structure for easy customization and experimentation.

## Outline

-   [PyTorch Template Project](#pytorch-template-project)
-   [Project Structure](#project-structure)
-   [Prerequisites](#prerequisites)
-   [Setup](#setup)
    -   [1. Create Your Repository from this Template](#1-create-your-repository-from-this-template)
    -   [2. Install dependencies](#2-install-dependencies)
    -   [3. (Optional) Weights & Biases Setup](#3-optional-weights--biases-setup)
-   [Usage](#usage)
    -   [1. Configure Your Run](#1-configure-your-run)
    -   [2. (Optional) Configure Optimization](#2-optional-configure-optimization)
    -   [3. Run the Experiment](#3-run-the-experiment)
    -   [4. Analyze Results](#4-analyze-results)
-   [Configuration Files](#configuration-files)
    -   [Run Configuration (`run_template.yaml`)](#run-configuration-run_templateyaml)
    -   [Optimization Configuration (`optimize_template.yaml`)](#optimization-configuration-optimize_templateyaml)
-   [Customization](#customization)
    -   [1. Customizing Run Configurations](#1-customizing-run-configurations)
    -   [2. Customizing Optimization Search Space](#2-customizing-optimization-search-space)
    -   [3. Using Different Optuna Samplers (e.g., GridSampler)](#3-using-different-optuna-samplers-eg-gridsampler)
    -   [4. Adding Custom Models, Optimizers, Schedulers, Pruners](#4-adding-custom-models-optimizers-schedulers-pruners)
    -   [5. Customizing Data Loading](#5-customizing-data-loading)
    -   [6. Customizing the Training Loop](#6-customizing-the-training-loop)
-   [Analysis Script (`analyze.py`)](#analysis-script-analyzepy)
-   [Contributing](#contributing)
-   [License](#license)
-   [Appendix](#appendix)
    -   [PFL (Predicted Final Loss) Pruner](#pfl-predicted-final-loss-pruner)

## Project Structure

- `config.py`: Defines `RunConfig` and `OptimizeConfig` for managing experiment and optimization settings.
- `main.py`: Entry point, handles arguments and experiment execution.
- `model.py`: Contains model architectures (e.g., MLP).
- `util.py`: Utility functions (data loading, training loop, analysis helpers, etc.).
- `analyze.py`: Script for analyzing completed runs and optimizations.
- `hyperbolic_lr.py`: Implementation of custom hyperbolic learning rate schedulers.
- `pruner.py`: Contains custom pruners like PFLPruner.
- `configs/`: Directory for configuration files.
    - `run_template.yaml`: Template for basic run configuration.
    - `optimize_template.yaml`: Template for optimization configuration.
- `runs/`: Directory where experiment results (models, configs) are saved.
- `requirements.txt`: Lists project dependencies.
- `README.md`: This file.
- `RELEASES.md`: Project release notes.

## Prerequisites

- Python 3.x
- Git

## Setup

1.  **Create Your Repository from this Template:**
    - Navigate to the main page of this repository on GitHub.
    - Click the "Use this template" button (usually located near the top-right).
    - Select "Create a new repository".
    - Choose an owner, provide a repository name for your new project, and configure other options as needed.
    - Click "Create repository from template".
    - You now have a new repository in your account with a copy of this template's files. Clone *your* new repository to your local machine:
      ```sh
      git clone [https://github.com/](https://github.com/)<your-username>/<your-new-repository-name>.git
      cd <your-new-repository-name>
      ```

2.  **Install dependencies:**
    It's recommended to use a virtual environment.
    ```sh
    # Create and activate a virtual environment (example using uv)
    uv venv
    source .venv/bin/activate # On Windows use `.venv\Scripts\activate`

    # Use uv (recommended)
    uv pip sync requirements.txt

    # Or use uv but install manually
    uv pip install -U torch wandb rich beaupy polars numpy optuna matplotlib scienceplots

    # Or use pip
    pip install -r requirements.txt
    ```
    The requirements include PyTorch, wandb, Optuna, beaupy, and other necessary packages.

3.  **(Optional) Weights & Biases Setup:**
    If you want to use wandb for logging:
    - Sign up for a free account at [https://wandb.ai/](https://wandb.ai/).
    - Log in to your account in your terminal:
      ```sh
      wandb login
      ```

## Usage

1.  **Configure Your Run:**
    - Modify `configs/run_template.yaml` or create a copy (e.g., `configs/my_experiment.yaml`) and adjust the parameters. See the [Customization](#customization) section for details.

2.  **(Optional) Configure Optimization:**
    - If you want to perform hyperparameter optimization, modify `configs/optimize_template.yaml` or create a copy (e.g., `configs/my_optimization.yaml`). Define the `search_space`, `sampler`, and `pruner`. See the [Customization](#customization) section.

3.  **Run the Experiment:**

    - **Single Run:**
      ```sh
      python main.py --run_config configs/run_template.yaml
      ```
      (Replace `run_template.yaml` with your specific run configuration file if needed).

    - **Optimization Run:**
      ```sh
      python main.py --run_config configs/run_template.yaml --optimize_config configs/optimize_template.yaml
      ```
      (Replace file names as needed). This will use Optuna to search for the best hyperparameters based on your `optimize_template.yaml`.

4.  **Analyze Results:**
    - Use the interactive analysis script:
      ```sh
      python analyze.py
      ```
    - Follow the prompts to select the project, run group, and seed to load and analyze the model.

## Configuration Files

### Run Configuration (`run_template.yaml`)

-   `project`: Project name (used for wandb and results saving).
-   `device`: Device ('cpu', 'cuda:0', etc.).
-   `net`: Path to the model class (e.g., `model.MLP`).
-   `optimizer`: Path to the optimizer class (e.g., `torch.optim.adamw.AdamW`).
-   `scheduler`: Path to the scheduler class (e.g., `hyperbolic_lr.ExpHyperbolicLR`, `torch.optim.lr_scheduler.CosineAnnealingLR`).
-   `epochs`: Number of training epochs.
-   `batch_size`: Training batch size.
-   `seeds`: List of random seeds for running the experiment multiple times.
-   `net_config`: Dictionary of arguments passed to the model's `__init__` method.
-   `optimizer_config`: Dictionary of arguments for the optimizer.
-   `scheduler_config`: Dictionary of arguments for the scheduler.
-   `early_stopping_config`: Configuration for early stopping.
    -   `enabled`: `true` or `false`.
    -   `patience`: How many epochs to wait after last improvement.
    -   `mode`: 'min' or 'max'.
    -   `min_delta`: Minimum change to qualify as an improvement.

### Optimization Configuration (`optimize_template.yaml`)

-   `study_name`: Name for the Optuna study.
-   `trials`: Number of optimization trials to run.
-   `seed`: Random seed for the optimization sampler.
-   `metric`: Metric to optimize (e.g., `val_loss`).
-   `direction`: 'minimize' or 'maximize'.
-   `sampler`: Optuna sampler configuration.
    -   `name`: Path to the sampler class (e.g., `optuna.samplers.TPESampler`).
    -   `kwargs`: (Optional) Arguments for the sampler.
-   `pruner`: (Optional) Optuna pruner configuration.
    -   `name`: Path to the pruner class (e.g., `pruner.PFLPruner`).
    -   `kwargs`: Arguments for the pruner.
-   `search_space`: Defines hyperparameters to search. Nested under `net_config`, `optimizer_config`, etc.
    -   `type`: 'int', 'float', or 'categorical'.
    -   `min`, `max`: Range for numerical types.
    -   `log`: `true` for logarithmic scale (float).
    -   `step`: Step size (int).
    -   `choices`: List of options (categorical).

## Customization

This template is designed for flexibility. Here’s how to customize different parts:

### 1. Customizing Run Configurations

Modify the parameters in a run configuration YAML file (like `configs/run_template.yaml`) to change experiment settings.

**Example:** Let's create `configs/run_mlp_small_fastlr.yaml` based on `run_template.yaml` but with a smaller network and a different learning rate.

*Original `configs/run_template.yaml` (simplified):*
```yaml
# configs/run_template.yaml
project: PyTorch_Template
device: cuda:0
net: model.MLP
optimizer: torch.optim.adamw.AdamW
scheduler: hyperbolic_lr.ExpHyperbolicLR
epochs: 50
seeds: [89, 231, 928, 814, 269]
net_config:
  nodes: 64 # Original nodes
  layers: 4
optimizer_config:
  lr: 1.e-3 # Original LR
scheduler_config:
  upper_bound: 250
  max_iter: 50
  infimum_lr: 1.e-5
...
````

*New `configs/run_mlp_small_fastlr.yaml`:*

```yaml
# configs/run_mlp_small_fastlr.yaml
project: PyTorch_Template_SmallMLP # Maybe change project name
device: cuda:0
net: model.MLP
optimizer: torch.optim.adamw.AdamW
scheduler: hyperbolic_lr.ExpHyperbolicLR # Or change scheduler
epochs: 50
seeds: [42, 123] # Use different seeds if desired
net_config:
  nodes: 32   # Changed nodes
  layers: 3   # Changed layers
optimizer_config:
  lr: 5.e-3 # Changed learning rate
scheduler_config: # Adjust scheduler params if needed, e.g., related to epochs or LR
  upper_bound: 250
  max_iter: 50
  infimum_lr: 1.e-5
... # Keep or adjust other settings like early_stopping
```

Now you can run this specific configuration:

```sh
python main.py --run_config configs/run_mlp_small_fastlr.yaml
```

### 2. Customizing Optimization Search Space

Modify the `search_space` section in your optimization configuration file (e.g., `configs/optimize_template.yaml`) to change which hyperparameters Optuna searches over and their ranges/choices.

**Example:** Adjusting the search space in `configs/optimize_template.yaml`.

*Original `search_space` (simplified):*

```yaml
# configs/optimize_template.yaml
...
search_space:
  net_config:
    nodes:
      type: categorical
      choices: [32, 64, 128] # Original choices
    layers:
      type: int
      min: 3
      max: 5 # Original max
  optimizer_config:
    lr:
      type: float
      min: 1.e-3 # Original min LR
      max: 1.e-2
      log: true
  scheduler_config:
    infimum_lr: # Only searching infimum_lr
      type: float
      min: 1.e-7
      max: 1.e-4
      log: true
...
```

*Modified `search_space`:*

```yaml
# configs/optimize_template.yaml
...
search_space:
  net_config:
    nodes:
      type: categorical
      choices: [64, 128, 256] # Changed choices for nodes
    layers:
      type: int
      min: 4 # Changed min layers
      max: 6 # Changed max layers
  optimizer_config:
    lr:
      type: float
      min: 5.e-4 # Changed min LR
      max: 5.e-3 # Changed max LR
      log: true
  scheduler_config:
    # Add search for upper_bound
    upper_bound:
        type: int
        min: 100
        max: 300
        step: 50
    infimum_lr:
      type: float
      min: 1.e-6 # Changed range
      max: 1.e-5
      log: true
...
```

This updated configuration will search over different node sizes, layer counts, learning rates, and scheduler parameters.

### 3. Using Different Optuna Samplers (e.g., GridSampler)

You can change the sampler used by Optuna by modifying the `sampler` section in `configs/optimize_template.yaml`.

**Example:** Switching from `TPESampler` to `GridSampler`.

*Original `sampler` section:*

```yaml
# configs/optimize_template.yaml
...
sampler:
  name: optuna.samplers.TPESampler
  #kwargs:
  #  n_startup_trials: 10
...
```

*Using `GridSampler`:*

```yaml
# configs/optimize_template.yaml
...
sampler:
  name: optuna.samplers.GridSampler # Changed sampler name
  # kwargs: {} # GridSampler often doesn't need kwargs here
...
# IMPORTANT CONDITION for GridSampler:
# All parameters defined in the 'search_space' MUST be of type 'categorical'.
# GridSampler explores all combinations of the categorical choices.
# If your search_space contains 'int' or 'float' types, using GridSampler
# will cause an error based on the current implementation in config.py.
# (See _create_sampler and grid_search_space methods)

# Example search_space compatible with GridSampler:
search_space:
  net_config:
    nodes:
      type: categorical
      choices: [64, 128]
    layers:
      type: categorical # Must be categorical
      choices: [3, 4]
  optimizer_config:
    lr:
      type: categorical # Must be categorical
      choices: [1.e-3, 5.e-3]
  scheduler_config:
    infimum_lr:
      type: categorical # Must be categorical
      choices: [1.e-5, 1.e-6]
...
```

**Condition:** To use `GridSampler`, ensure *all* parameters listed under `search_space` have `type: categorical`. The code automatically constructs the required format for `GridSampler` but only if this condition is met.

### 4. Adding Custom Models, Optimizers, Schedulers, Pruners

  - **Models:** Create your model class (inheriting from `torch.nn.Module`) in `model.py` or a new Python file. Ensure its `__init__` method accepts a config dictionary (e.g., `net_config` from the YAML) as the first argument. Update the `net:` path in your run config YAML.
  - **Optimizers/Schedulers:** Implement your custom classes or use existing ones from `torch.optim` or elsewhere (like `hyperbolic_lr.py`). Update the `optimizer:` or `scheduler:` path and `*_config` dictionaries in the YAML. The template uses `importlib` to load classes dynamically based on the paths provided.
  - **Pruners:** Create your pruner class (inheriting from `pruner.BasePruner` or implementing the Optuna pruner interface) in `pruner.py` or a new file. Update the `pruner:` section in the optimization YAML.

### 5. Customizing Data Loading

  - Modify the `load_data` function in `util.py` to load your specific dataset. It should return PyTorch `Dataset` objects for training and validation.

### 6. Customizing the Training Loop

  - Modify the `Trainer` class in `util.py`. Adjust the `train_epoch`, `val_epoch`, and `train` methods for your specific task, loss functions, or metrics. Ensure the `train` method returns the value specified as the `metric` in your optimization config if applicable.

## Analysis Script (`analyze.py`)

The `analyze.py` script provides an interactive command-line interface to load and inspect results from completed runs.

  - It uses helper functions from `util.py` (like `select_project`, `select_group`, `select_seed`, `load_model`, `load_study`, `load_best_model`) to navigate the saved runs in the `runs/` directory.
  - You can easily extend the `main` function in `analyze.py` to perform more detailed analysis, plotting, or evaluation specific to your project needs.

## Contributing

Contributions are welcome\! Please feel free to submit a Pull Request.

## License

This project is provided as a template and is intended to be freely used, modified, and distributed. Users of this template are encouraged to choose a license that best suits their specific project needs.

For the template itself:

  - You are free to use, modify, and distribute this template.
  - No attribution is required, although it is appreciated.
  - The template is provided "as is", without warranty of any kind.

When using this template for your own project, please remember to:

1.  Remove this license section or replace it with your chosen license.
2.  Ensure all dependencies and libraries used in your project comply with their respective licenses.

For more information on choosing a license, visit [https://choosealicense.com/](https://choosealicense.com/).

## Appendix

<details>
<summary><strong>PFL (Predicted Final Loss) Pruner</strong></summary>

### Overview

The PFL pruner (`pruner.PFLPruner`) is a custom pruner inspired by techniques to predict the final performance of a training run based on early-stage metrics. It helps optimize hyperparameter search by early stopping unpromising trials based on their predicted final loss (`pfl`).

### Key Features

  - Maintains a list of the `top_k` best-performing completed trials based on their final validation loss.
  - For ongoing trials (after a warmup period), it predicts the final loss based on the current loss history.
  - It compares the current trial's predicted final loss (`pfl`) with the minimum `pfl` observed among the `top_k` completed trials.
  - Prunes the current trial if its predicted final loss is worse (lower, since `pfl` is -log10(loss)) than the worst `pfl` in the top-k list.
  - Supports multi-seed runs by averaging metrics across seeds for decision making.
  - Integrates with Optuna's study mechanism.

### Configuration

In your `optimize_template.yaml`, configure the pruner under the `pruner` section:

```yaml
pruner:
  name: pruner.PFLPruner # Path to the pruner class
  kwargs:
    n_startup_trials: 10    # Number of trials to complete before pruning starts
    n_warmup_epochs: 10     # Number of epochs within a trial before pruning is considered
    top_k: 10               # Number of best completed trials to keep track of
    target_epoch: 50        # The target epoch used for predicting final loss
```

### How It Works

1.  The first `n_startup_trials` run to completion without being pruned to establish baseline performance.
2.  For subsequent trials, pruning is considered only after `n_warmup_epochs`.
3.  The pruner calculates the average predicted final loss (`pfl`) for the current trial based on the loss history across its seeds.
4.  It compares this `pfl` to the `pfl` values of the `top_k` trials that have already completed.
5.  If the current trial's `pfl` is lower than the minimum `pfl` recorded among the top completed trials, the trial is pruned (as lower `pfl` indicates worse predicted performance).
6.  When a trial completes, its final validation loss and `pfl` are considered for inclusion in the `top_k` list.

</details>

