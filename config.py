import torch
from torch import nn

from model import MLP

from dataclasses import dataclass
import json

@dataclass
class RunConfig:
    project: str
    device: str
    net: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: torch.optim.lr_scheduler._LRScheduler
    epochs: int
    batch_size: int
    net_config: dict[str, int]
    optimizer_config: dict[str, int | float]
    scheduler_config: dict[str, int | float]

    def create_model(self):
        return self.net(self.net_config, device=self.device)

    def create_optimizer(self, model):
        return self.optimizer(model.parameters(), **self.optimizer_config)

    def create_scheduler(self, optimizer):
        return self.scheduler(optimizer, **self.scheduler_config)

    def gen_group_name(self):
        name = f"{self.net.__name__}"

        for k, v in self.net_config.items():
            name += f"_{k[0]}_{v}"

        for k, v in self.optimizer_config.items():
            if isinstance(v, float):
                name += f"_{k[0]}_{v:.4e}"
            else:
                name += f"_{k[0]}_{v}"

        for k, v in self.scheduler_config.items():
            if isinstance(v, float):
                name += f"_{k[0]}_{v:.4e}"
            else:
                name += f"_{k[0]}_{v}"

        return name

    def gen_tags(self):
        tags = [self.net.__name__]

        for k, v in self.net_config.items():
            tags.append(f"{k[0]}={v}")

        return tags

    def gen_config(self):
        configs = {
            "project": self.project,
            "device": self.device,
            "net": self.net.__name__,
            "optimizer": self.optimizer.__name__,
            "scheduler": self.scheduler.__name__,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
        }
        for k, v in self.net_config.items():
            configs[k] = v
        for k, v in self.optimizer_config.items():
            configs[k] = v
        for k, v in self.scheduler_config.items():
            configs[k] = v
        return configs

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.gen_config(), f)



def default_run_config():
    return RunConfig(
        project="PyTorch_Template",
        device="cpu",
        net=MLP,
        optimizer=torch.optim.AdamW,
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,
        epochs=50,
        batch_size=256,
        net_config={
            "nodes": 128,
            "layers": 4,
        },
        optimizer_config={
            "lr": 1e-3,
        },
        scheduler_config={
            "T_max": 50,
            "eta_min": 1e-5,
        },
    )
