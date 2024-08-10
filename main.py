import torch
from torch import nn
import torch.nn.functional as F
import wandb
import survey

from model import MLP
from util import load_data, set_seed, select_device, Trainer
from config import RunConfig

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

        wandb.finish()
    return total_loss / len(seeds)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=str, default="PyTorch_Template")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    wandb.require("core")

    project = args.project
    seeds = [args.seed] if args.seed != 0 else [89, 231, 928, 814, 269]

    device = select_device()
    print(f"device: {device}")

    # Run mode
    run_modes = ['Run', 'Optimize']
    run_mode = survey.routines.select("Run mode", options=run_modes)
    run_mode = run_modes[run_mode]

    # Load data
    dl_train, dl_val = load_sho()

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
    nodes = survey.routines.numeric(
        "Input Nodes",
        decimal=False
    )
    layers = survey.routines.numeric(
        "Input Layers",
        decimal=False
    )
    model_config = {
        "nodes": nodes,
        "layers": layers,
    }

    # Optimizer config
    lr = survey.routines.numeric(
        "Input Learning Rate",
        decimal=True
    )
    optimizer_config = {
        "lr": lr,
    }

    # Scheduler config
    T_max = epochs
    eta_min = survey.routines.numeric(
        "Input eta_min",
        decimal=True
    )
    scheduler_config = {
        "T_max": T_max,
        "eta_min": eta_min,
    }

    # Run Config
    run_config = RunConfig(
        project=project,
        device=device,
        net=MLP,
        optimizer=torch.optim.AdamW,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        net_config=model_config,
        optimizer_config=optimizer_config,
        scheduler_config=scheduler_config,
        epochs=epochs,
        batch_size=batch_size,
    )

    # Run
    if run_mode == 'Run':
        run(run_config, seeds, dl_train, dl_val)
    elif run_mode == 'Optimize':
        pass

if __name__ == "__main__":
    main()
