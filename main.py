import torch
from torch import nn
import torch.nn.functional as F
import wandb
import survey
import optuna

from util import load_data, set_seed, select_device, Trainer
from config import RunConfig, OptimizeConfig

import random
import numpy as np
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_config", type=str, required=True, help="Path to the YAML config file")
    parser.add_argument("--optimize_config", type=str, help="Path to the optimization YAML config file")
    args = parser.parse_args()

    wandb.require("core") # pyright: ignore

    device = select_device()
    print(f"device: {device}")

    # Load data
    dl_train, dl_val = load_data() # pyright: ignore

    # Run Config
    base_config = RunConfig.from_yaml(args.run_config)

    # Run
    if args.optimize:
        def objective(trial, base_config, optimize_config, dl_train, dl_val):
            params = optimize_config.suggest_params(trial)
            
            config = base_config.copy()
            for category, category_params in params.items():
                config[category].update(category_params)
            
            run_config = RunConfig(**config)
            return run(run_config, run_config.seeds, dl_train, dl_val)

        optimize_config = OptimizeConfig.from_yaml(args.optimize_config)
        study = optimize_config.create_study()
        study.optimize(lambda trial: objective(trial, base_config, optimize_config, dl_train, dl_val), n_trials=optimize_config.trials)

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
        
    else:
        run(base_config, dl_train, dl_val)


if __name__ == "__main__":
    main()
