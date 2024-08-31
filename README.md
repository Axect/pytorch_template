# PyTorch Template Project

This project provides a flexible template for PyTorch-based machine learning experiments.
It includes configuration management, logging with Weights & Biases (wandb), and a modular structure for easy customization and hyperparameter optimization.

## Project Structure

- `config.py`: Defines the `RunConfig` and `OptimizeConfig` classes for managing experiment configurations and optimization settings.
- `main.py`: The entry point of the project, handling command-line arguments and experiment execution.
- `model.py`: Contains the model architecture (currently an MLP).
- `util.py`: Utility functions for data loading, device selection, training, and analysis.
- `run_template.yaml`: Template for run configuration.
- `optimize_template.yaml`: Template for optimization configuration.

## Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/pytorch_template.git
   cd pytorch_template
   ```

2. Install the required packages:
   ```
   pip install torch wandb survey polars numpy optuna
   ```

3. (Optional) Set up a Weights & Biases account for experiment tracking.

## Usage

1. Configure your experiment by modifying `run_template.yaml` or creating a new YAML file based on it.

2. (Optional) Configure hyperparameter optimization by modifying `optimize_template.yaml` or creating a new YAML file based on it.

3. Run the experiment:
   ```
   python main.py --run_config path/to/run_config.yaml [--optimize_config path/to/optimize_config.yaml]
   ```

   If `--optimize_config` is provided, the script will perform hyperparameter optimization using Optuna.

## Configuration

### Run Configuration (`run_template.yaml`)

- `project`: Project name for wandb logging
- `device`: Device to run on (e.g., 'cpu', 'cuda:0')
- `net`: Model class to use
- `optimizer`: Optimizer class
- `scheduler`: Learning rate scheduler class
- `epochs`: Number of training epochs
- `batch_size`: Batch size for training
- `seeds`: List of random seeds for multiple runs
- `net_config`: Model-specific configuration
- `optimizer_config`: Optimizer-specific configuration
- `scheduler_config`: Scheduler-specific configuration

### Optimization Configuration (`optimize_template.yaml`)

- `study_name`: Name of the optimization study
- `trials`: Number of optimization trials
- `seed`: Random seed for optimization
- `metric`: Metric to optimize
- `direction`: Direction of optimization ('minimize' or 'maximize')
- `sampler`: Optuna sampler configuration
- `pruner`: (Optional) Optuna pruner configuration
- `search_space`: Definition of the hyperparameter search space

## Customization

- Modify the `MLP` class in `model.py` or add new model architectures.
- Adjust the `load_data` function in `util.py` to work with your dataset.
- Extend the `RunConfig` and `OptimizeConfig` classes in `config.py` to include additional parameters.

## Features

- Configurable experiments using YAML files
- Integration with Weights & Biases for experiment tracking
- Support for multiple random seeds
- Flexible model architecture (currently MLP)
- Device selection (CPU/CUDA)
- Learning rate scheduling
- Hyperparameter optimization using Optuna

## Analysis

The `util.py` file includes functions for analyzing completed runs:

- `select_group`: Select a run group for analysis
- `select_seed`: Select a specific seed from a run group
- `load_model`: Load a trained model and its configuration

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
