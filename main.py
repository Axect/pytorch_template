import torch
from torch import nn
import torch.nn.functional as F
import wandb
import survey

from model import MLP
from util import load_data, set_seed, select_device, Trainer
from config import RunConfig, model_setup, optimizer_setup, scheduler_setup

import random
import numpy as np
import argparse
import os


def run(run_config: RunConfig, seeds, dl_train, dl_val):
    project = run_config.project
    device = run_config.device

    group_name = run_config.gen_group_name()
    tags = run_config.gen_tags()

    total_loss = 0
    for seed in seeds:
        set_seed(seed)

        model = run_config.create_model().to(device)
        optimizer = run_config.create_optimizer(model)
        scheduler = run_config.create_scheduler(optimizer)

        run_name = group_name + f"[{seed}]"
        wandb.init(
            project=project,
            name=run_name,
            group=group_name,
            tags=tags,
            config=run_config.gen_config(),
        )

        trainer = Trainer(model, optimizer, scheduler, criterion=F.mse_loss, device=device)
        val_loss = trainer.train(dl_train, dl_val, epochs=run_config.epochs)
        total_loss += val_loss

        # Save model & configs
        if not os.path.exists(f"checkpoints/{run_name}"):
            os.makedirs(f"checkpoints/{run_name}")
        torch.save(model.state_dict(), f"checkpoints/{run_name}/model.pt")
        run_config.to_json(f"checkpoints/{run_name}/config.json")

        wandb.finish() # pyright: ignore
    return total_loss / len(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="PyTorch_Template")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--change_betas", action=argparse.BooleanOptionalAction)
    parser.add_argument("--change_weight_decay", action=argparse.BooleanOptionalAction)
    parser.set_defaults(change_betas=False, change_weight_decay=False)
    args = parser.parse_args()

    wandb.require("core") # pyright: ignore

    project = args.project
    seeds = [args.seed] if args.seed != 0 else [89, 231, 928, 814, 269]

    device = select_device()
    print(f"device: {device}")

    # Run mode
    run_modes = ['Run', 'Optimize']
    run_mode = survey.routines.select("Run mode", options=run_modes)
    run_mode = run_modes[run_mode] # pyright: ignore

    # Load data
    dl_train, dl_val = load_data() # pyright: ignore

    # Batch size & Epoch
    batch_size = survey.routines.numeric(
        "Input Batch size",
        decimal=False
    )
    epochs = survey.routines.numeric(
        "Input Epochs",
        decimal=False
    )

    # Model config
    model, model_config = model_setup()

    # Optimizer config
    optimizer, optimizer_config = optimizer_setup(change_betas=args.change_betas, change_weight_decay=args.change_weight_decay)

    # Scheduler config
    scheduler, scheduler_config = scheduler_setup(lr, epochs) # pyright: ignore

    # Run Config
    run_config = RunConfig(
        project=project,
        device=device,
        net=model,                          # pyright: ignore
        optimizer=optimizer,                # pyright: ignore
        scheduler=scheduler,
        net_config=model_config,            # pyright: ignore
        optimizer_config=optimizer_config,  # pyright: ignore
        scheduler_config=scheduler_config,
        epochs=epochs,                      # pyright: ignore
        batch_size=batch_size,              # pyright: ignore
    )

    # Run
    if run_mode == 'Run':
        run(run_config, seeds, dl_train, dl_val)
    elif run_mode == 'Optimize':
        optimizable_config = run_config.optimizable_config()

if __name__ == "__main__":
    main()
