import torch
from torch import nn
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from hyperbolic_lr import HyperbolicLR, ExpHyperbolicLR
import survey

from model import MLP

from dataclasses import dataclass
import json
import yaml

@dataclass
class RunConfig:
    project: str
    device: str
    net: nn.Module
    optimizer: torch.optim.Optimizer                    # pyright: ignore
    scheduler: torch.optim.lr_scheduler._LRScheduler
    epochs: int
    batch_size: int
    net_config: dict[str, int]
    optimizer_config: dict[str, int | float]
    scheduler_config: dict[str, int | float]

    def create_model(self):
        return self.net(self.net_config, device=self.device)

    def create_optimizer(self, model):
        return self.optimizer(model.parameters(), **self.optimizer_config) # pyright: ignore

    def create_scheduler(self, optimizer):
        return self.scheduler(optimizer, **self.scheduler_config) # pyright: ignore

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

        tags.append(self.optimizer.__name__) # pyright: ignore
        tags.append(self.scheduler.__name__) # pyright: ignore

        return tags

    def gen_config(self):
        configs = {
            "project": self.project,
            "device": self.device,
            "net": self.net.__name__,
            "optimizer": self.optimizer.__name__, # pyright: ignore
            "scheduler": self.scheduler.__name__, # pyright: ignore
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

    def optimizable_config(self):
        optimizable = {}
        for k, v in self.net_config.items():
            optimizable[k] = v
        for k, v in self.optimizer_config.items():
            optimizable[k] = v
        for k, v in self.scheduler_config.items():
            optimizable[k] = v
        return optimizable

    def to_json(self, path):
        with open(path, "w") as f:
            json.dump(self.gen_config(), f)

    def from_json(self, path):
        with open(path, "r") as f:
            config = json.load(f)

        self.project = config["project"]
        self.device = config["device"]
        self.epochs = config["epochs"]
        self.batch_size = config["batch_size"]

        self.net = eval(config["net"])
        self.optimizer = eval(config["optimizer"])
        self.scheduler = eval(config["scheduler"])

        for k in self.net_config:
            if k in config:
                self.net_config[k] = config[k]
        for k in self.optimizer_config:
            if k in config:
                self.optimizer_config[k] = config[k]
        for k in self.scheduler_config:
            if k in config:
                self.scheduler_config[k] = config[k]


def default_run_config():
    return RunConfig(
        project="PyTorch_Template",
        device="cpu",
        net=MLP,                                                # pyright: ignore
        optimizer=torch.optim.AdamW,                            # pyright: ignore
        scheduler=torch.optim.lr_scheduler.CosineAnnealingLR,   # pyright: ignore
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


@classmethod
def from_yaml(cls, path):
    available_models = {
        "MLP": MLP
    }
    available_optimizers = {
        "Adam": torch.optim.adam.Adam,
        "AdamW": torch.optim.adamw.AdamW
    }
    available_schedulers = {
        "PolynomialLR": torch.optim.lr_scheduler.PolynomialLR,
        "CosineAnnealingLR": torch.optim.lr_scheduler.CosineAnnealingLR,
        "ExponentialLR": torch.optim.lr_scheduler.ExponentialLR,
        "HyperbolicLR": HyperbolicLR,
        "ExpHyperbolicLR": ExpHyperbolicLR
    }

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    
    return cls(
        project=config['project'],
        device=config['device'],
        net=available_models[config['net']],
        optimizer=available_optimizers[config['optimizer']],
        scheduler=available_schedulers[config['scheduler']],
        epochs=config['epochs'],
        batch_size=config['batch_size'],
        net_config=config['net_config'],
        optimizer_config=config['optimizer_config'],
        scheduler_config=config['scheduler_config']
    )


def model_setup():
    models = {
        "MLP": MLP,
    }
    model_name = list(models.keys())
    model_idx = survey.routines.select("Model", options=model_name)
    model_name = model_name[model_idx] # pyright: ignore
    model = models[model_name]
    model_config = {}
    if model_name == "MLP":
        nodes = survey.routines.numeric(
            "Input nodes",
            decimal=False
        )
        layers = survey.routines.numeric(
            "Input layers",
            decimal=False
        )
        model_config = {
            "nodes": nodes,
            "layers": layers,
        }
    else:
        raise ValueError("Model name not found.")
    return model, model_config


def optimizer_setup(change_betas: bool, change_weight_decay: bool):
    optimizers = {
        "Adam": Adam,
        "AdamW": AdamW,
    }
    optimizer_name = list(optimizers.keys())
    optimizer_idx = survey.routines.select("Optimizer", options=optimizer_name)
    optimizer_name = optimizer_name[optimizer_idx] # pyright: ignore
    optimizer = optimizers[optimizer_name]
    optimizer_config = {}
    lr = survey.routines.numeric(
        "Input Learning Rate",
        decimal=True
    )
    optimizer_config["lr"] = lr
    if change_betas:
        betas = [
            survey.routines.numeric(
                "Input beta1",
                decimal=True
            ),
            survey.routines.numeric(
                "Input beta2",
                decimal=True
            ),
        ]
        optimizer_config["betas"] = betas
    if change_weight_decay:
        weight_decay = survey.routines.numeric(
            "Input weight_decay",
            decimal=True
        )
        optimizer_config["weight_decay"] = weight_decay
    return optimizer, optimizer_config


def scheduler_setup(lr: float, epochs: int):
    schedulers = {
        "CosineAnnealingLR": CosineAnnealingLR,
        "HyperbolicLR": HyperbolicLR,
        "ExpHyperbolicLR": ExpHyperbolicLR,
    }
    schedulers_name = list(schedulers.keys())
    scheduler_idx = survey.routines.select("Scheduler", options=schedulers_name)
    scheduler_name = schedulers_name[scheduler_idx] # pyright: ignore
    scheduler = schedulers[scheduler_name]
    scheduler_config = {}
    if scheduler_name == "CosineAnnealingLR":
        T_max = epochs
        eta_min = survey.routines.numeric(
            "Input eta_min",
            decimal=True
        )
        scheduler_config = {
            "T_max": T_max,
            "eta_min": eta_min,
        }
    elif scheduler_name in ["HyperbolicLR", "ExpHyperbolicLR"]:
        upper_bound = survey.routines.numeric(
            "Input upper_bound",
            decimal=False
        )
        max_iter = epochs
        init_lr = lr
        infimum_lr = survey.routines.numeric(
            "Input infimum_lr",
            decimal=True
        )
        scheduler_config = {
            "upper_bound": upper_bound,
            "max_iter": max_iter,
            "init_lr": init_lr,
            "infimum_lr": infimum_lr,
        }
    else:
        raise ValueError("Invalid scheduler")
    return scheduler, scheduler_config
