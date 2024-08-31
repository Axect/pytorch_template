import torch
from torch import nn
from torch.utils.data import TensorDataset
import torch.nn.functional as F
import polars as pl
import numpy as np
import survey
import wandb

from config import RunConfig

import random
import os
from math import pi


def load_data(n=10000, split_ratio=0.8, seed=42):
    # Fix Seed
    torch.manual_seed(seed)

    x = torch.linspace(0, 1, n) + torch.rand(n) * 0.01
    y = torch.cos(x * (2 * pi)) + torch.rand(n) * 0.01

    ics = torch.randperm(n)
    ics_train = ics[:int(n * split_ratio)]
    ics_val   = ics[int(n * split_ratio):]

    x_train = x[ics_train].view(-1, 1)
    y_train = y[ics_train].view(-1, 1)
    x_val   = x[ics_val].view(-1, 1)
    y_val   = y[ics_val].view(-1, 1)

    train_ds    = TensorDataset(x_train, y_train)
    val_ds      = TensorDataset(x_val, y_val)

    return train_ds, val_ds


def set_seed(seed: int):
    # random
    random.seed(seed)
    # numpy
    np.random.seed(seed)
    # pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device():
    device_count = torch.cuda.device_count()
    if device_count >= 1:
        options = [f"cuda:{i}" for i in range(device_count)] + ["cpu"]
        device_index = survey.routines.select(
            "Select device", options=options
        )
        device = options[device_index]
    else:
        device = "cpu"
    return device


# List all checkpoints and select one
def select_checkpoint(path="checkpoints/"):
    # run_name + [seed]
    folders = os.listdir(path)

    run_names = set()
    for folder in folders:
        if folder.endswith("]"):
            run_names.add(folder.split("[")[0])
    run_names = list(run_names)
    run_idx = survey.routines.select("Select run", options=run_names)
    run = run_names[run_idx]

    seeds = set()
    for folder in folders:
        if folder.endswith("]"):
            if folder.split("[")[0] == run:
                seeds.add(folder.split("[")[1].split("]")[0])
    seeds = list(seeds)

    seed_idx = survey.routines.select("Select seed", options=seeds)

    seed = seeds[seed_idx]
    
    return os.path.join(path, run + f"[{seed}]")


class Trainer:
    def __init__(self, model, optimizer, scheduler, criterion, device="cpu"):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device

    def step(self, x):
        return self.model(x)

    def train_epoch(self, dl_train):
        self.model.train()
        train_loss = 0
        for x, y in dl_train:
            x = x.to(self.device).requires_grad_(True)
            y = y.to(self.device)
            y_pred = self.step(x)
            loss = self.criterion(y_pred, y)
            train_loss += loss.item()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        train_loss /= len(dl_train)
        return train_loss

    def val_epoch(self, dl_val):
        self.model.eval()
        val_loss = 0
        for x, y in dl_val:
            x = x.to(self.device).requires_grad_(True)
            y = y.to(self.device)
            y_pred = self.step(x)
            loss = self.criterion(y_pred, y)
            val_loss += loss.item()
        val_loss /= len(dl_val)
        return val_loss

    def train(self, dl_train, dl_val, epochs):
        val_loss = 0
        for epoch in range(epochs):
            train_loss = self.train_epoch(dl_train)
            val_loss = self.val_epoch(dl_val)
            self.scheduler.step()
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": self.optimizer.param_groups[0]["lr"],
            })
            if epoch % 10 == 0:
                print(f"epoch: {epoch}, train_loss: {train_loss}, val_loss: {val_loss}, lr: {self.optimizer.param_groups[0]['lr']}")
        return val_loss


def run(run_config: RunConfig, dl_train, dl_val):
    project = run_config.project
    device = run_config.device
    seeds = run_config.seeds
    group_name = run_config.gen_group_name()
    tags = run_config.gen_tags()

    group_path = f"runs/{run_config.project}/{group_name}"
    if not os.path.exists(group_path):
        os.makedirs(group_path)
    run_config.to_yaml(f"{group_path}/config.yaml")

    total_loss = 0
    for seed in seeds:
        set_seed(seed)

        model = run_config.create_model().to(device)
        optimizer = run_config.create_optimizer(model)
        scheduler = run_config.create_scheduler(optimizer)

        run_name = f"{seed}"
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
        run_path = f"{group_path}/{run_name}"
        if not os.path.exists(run_path):
            os.makedirs(run_path)
        torch.save(model.state_dict(), f"{run_path}/model.pt")

        wandb.finish() # pyright: ignore

    return total_loss / len(seeds)
