import torch
from torch import nn
import polars as pl
import numpy as np
import survey
import wandb

import random
import os


def load_data(ratio=0.8, seed=42):
    pass


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
