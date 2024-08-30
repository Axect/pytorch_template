# PyTorch Template Project

This project provides a flexible template for PyTorch-based machine learning experiments. It includes configuration management, logging with Weights & Biases, and a modular structure for easy customization.

## Project Structure

- `config.py`: Defines the `RunConfig` class for managing experiment configurations.
- `main.py`: The entry point of the project, handling command-line arguments and experiment execution.
- `model.py`: Contains the model architecture (currently an MLP).
- `util.py`: Utility functions for data loading, device selection, and training.

## Setup

You can set up this project in two ways:

1. Use this repository as a template:
   - Click the "Use this template" button at the top of the repository page on GitHub.
   - Choose a name for your new repository and create it.
   - Clone your new repository:
     ```
     git clone https://github.com/yourusername/your-new-repo-name.git
     cd your-new-repo-name
     ```

2. Clone the repository directly:
   ```
   git clone https://github.com/Axect/pytorch_template
   cd pytorch_template
   ```

After setting up the repository, install the required packages:
```
pip install torch wandb survey polars numpy optuna
```

(Optional) Set up a Weights & Biases account for experiment tracking.

## Usage

1. Configure your experiment in `main.py` or by modifying the `RunConfig` in `config.py`.

2. Run the experiment:
   ```
   python main.py --project YourProjectName --seed 42
   ```

   You can specify a project name and a random seed. If no seed is provided, the script will use multiple predefined seeds.

3. Follow the prompts to select:
   - Run mode (Run or Optimize)
   - Device (CPU or available CUDA devices)
   - Batch size
   - Number of epochs
   - Model configuration (number of nodes and layers)
   - Learning rate
   - Scheduler configuration

## Customization

- Modify the `MLP` class in `model.py` or add new model architectures.
- Adjust the `load_data` function in `util.py` to work with your dataset.
- Extend the `RunConfig` class in `config.py` to include additional parameters.

## Features

- Configurable experiments using the `RunConfig` class
- Integration with Weights & Biases for experiment tracking
- Support for multiple random seeds
- Flexible model architecture (currently MLP)
- Device selection (CPU/CUDA)
- Learning rate scheduling

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the [MIT License](LICENSE).
