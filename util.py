import torch
from torch.utils.data import TensorDataset, random_split
import numpy as np
import beaupy
import wandb
import optuna
from tqdm import tqdm

from config import RunConfig
from callbacks import (
    CallbackRunner, OptimizerModeCallback, EarlyStoppingCallback,
    WandbLoggingCallback, PrunerCallback, LossPredictionCallback,
    NaNDetectionCallback, CheckpointCallback,
    GradientMonitorCallback, OverfitDetectionCallback,
    CSVLoggingCallback, TUILoggingCallback, LatestModelCallback,
)
from checkpoint import CheckpointManager, SeedManifest
from provenance import save_provenance, compute_config_hash

import random
import os
import math
import time
from math import pi


def load_data(n=10000, split_ratio=0.8, seed=42):
    # Fix random seed for reproducibility
    torch.manual_seed(seed)

    x_noise = torch.rand(n) * 0.02
    x = torch.linspace(0, 1, n) + x_noise
    x = x.clamp(0, 1) # Fix x to be in [0, 1]

    noise_level = 0.05
    y = (
        1.0 * torch.sin(4 * pi * x)
        + 0.5 * torch.sin(10 * pi * x)
        + 1.5 * (x**2)
        + torch.randn(n) * noise_level
    )

    x = x.view(-1, 1)
    y = y.view(-1, 1)

    full_dataset = TensorDataset(x, y)

    train_size = int(n * split_ratio)
    val_size = n - train_size

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=generator
    )

    return train_dataset, val_dataset


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



def predict_final_loss(losses, max_epochs):
    """Predict the final validation loss using shifted exponential decay.

    Fits L(t) = a * exp(-b * t) + c to EMA-smoothed losses.
    Returns the predicted raw loss value at max_epochs.
    Works with positive and negative losses.
    """
    n = len(losses)
    if n < 10:
        return float(losses[-1])

    y = np.array(losses, dtype=np.float64)

    # EMA smoothing — adaptive span
    span = min(n // 3, 20)
    alpha = 2.0 / (span + 1)
    ema = np.empty(n)
    ema[0] = y[0]
    for i in range(1, n):
        ema[i] = alpha * y[i] + (1 - alpha) * ema[i - 1]

    # Three equally-spaced anchor points from smoothed curve
    i1, i2, i3 = n // 3, 2 * n // 3, n - 1
    y1, y2, y3 = ema[i1], ema[i2], ema[i3]

    d12 = y1 - y2
    d23 = y2 - y3

    # Need both differences nonzero and same sign (monotonic decay or increase)
    if abs(d12) < 1e-15 or abs(d23) < 1e-15:
        return float(ema[-1])

    r = d23 / d12

    if r <= 0 or r >= 1:
        # Non-convergent: loss increasing, oscillating, or accelerating
        # Use damped linear extrapolation from recent trend
        window = min(10, n - 1)
        recent_rate = (ema[-1] - ema[-1 - window]) / window
        remaining = max(max_epochs - n, 0)
        predicted = ema[-1] + recent_rate * remaining * 0.5
        return float(predicted) if np.isfinite(predicted) else float(ema[-1])

    # Convergent decay: fit L(t) = a * exp(-b * t) + c
    d = float(i2 - i1)
    b = -np.log(r) / d
    t1 = float(i1)
    t2 = float(i2)

    denom = np.exp(-b * t1) - np.exp(-b * t2)
    if abs(denom) < 1e-30:
        return float(ema[-1])

    a = d12 / denom
    c = y1 - a * np.exp(-b * t1)

    predicted = a * np.exp(-b * max_epochs) + c

    if np.isfinite(predicted):
        return float(predicted)

    return float(ema[-1])


class Trainer:
    def __init__(
        self,
        model,
        optimizer,
        scheduler,
        criterion,
        callbacks=None,
        device="cpu",
    ):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.criterion = criterion
        self.device = device
        self.callbacks = callbacks if callbacks is not None else CallbackRunner()
        self._total_epochs = 0
        self._loss_prediction = None
        self._max_grad_norm: float | None = None
        self._overfit_gap_ratio: float | None = None

    def step(self, x):
        return self.model(x)

    def train_epoch(self, dl_train, epoch=None, total_epochs=None):
        self.model.train()
        self.callbacks.fire("on_train_epoch_begin", trainer=self, epoch=epoch)
        train_loss = 0
        total_size = 0

        # Create progress bar description
        desc = f"Epoch {epoch+1}/{total_epochs}" if epoch is not None and total_epochs is not None else "Training"

        for batch_idx, (x, y) in enumerate(tqdm(dl_train, desc=desc, leave=False)):
            x = x.to(self.device)
            y = y.to(self.device)
            y_pred = self.step(x)
            loss = self.criterion(y_pred, y)
            train_loss += loss.item() * x.shape[0]
            total_size += x.shape[0]
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.callbacks.fire("on_train_step_end", trainer=self, batch_idx=batch_idx, loss=loss.item())
        train_loss /= total_size
        return train_loss

    def val_epoch(self, dl_val, epoch=None):
        self.model.eval()
        self.callbacks.fire("on_val_begin", trainer=self, epoch=epoch)
        val_loss = 0
        total_size = 0
        with torch.inference_mode():
            for x, y in tqdm(dl_val, desc="Validation", leave=False):
                x = x.to(self.device)
                y = y.to(self.device)
                y_pred = self.step(x)
                loss = self.criterion(y_pred, y)
                val_loss += loss.item() * x.shape[0]
                total_size += x.shape[0]
        val_loss /= total_size
        self.callbacks.fire("on_val_end", trainer=self, epoch=epoch, val_loss=val_loss, metrics={})
        return val_loss

    def train(self, dl_train, dl_val, epochs):
        self._total_epochs = epochs
        self.callbacks.fire("on_train_begin", trainer=self, epochs=epochs)
        val_loss = 0

        for epoch in tqdm(range(epochs), desc="Overall Progress"):
            train_loss = self.train_epoch(dl_train, epoch=epoch, total_epochs=epochs)
            val_loss = self.val_epoch(dl_val, epoch=epoch)

            self.callbacks.fire(
                "on_epoch_end", trainer=self, epoch=epoch,
                train_loss=train_loss, val_loss=val_loss, metrics={},
            )

            # Check callback signals
            break_flag = False
            for cb in self.callbacks.callbacks:
                if isinstance(cb, NaNDetectionCallback) and cb.nan_detected:
                    val_loss = math.inf
                    break_flag = True
                    break
                if isinstance(cb, EarlyStoppingCallback) and cb.should_stop:
                    tqdm.write(f"Early stopping triggered at epoch {epoch}")
                    break_flag = True
                    break
            if break_flag:
                break

            self.scheduler.step()

        self.callbacks.fire("on_train_end", trainer=self)
        return val_loss


def run(
    run_config: RunConfig, dl_train, dl_val, group_name=None, trial=None, pruner=None
):
    project = run_config.project
    device = run_config.device
    seeds = run_config.seeds
    if not group_name:
        group_name = run_config.gen_group_name()
    tags = run_config.gen_tags()

    group_path = f"runs/{run_config.project}/{group_name}"
    if not os.path.exists(group_path):
        os.makedirs(group_path)
    run_config.to_yaml(f"{group_path}/config.yaml")

    # Register trial at the beginning if pruner exists
    if pruner is not None and trial is not None and hasattr(pruner, "register_trial"):
        pruner.register_trial(trial.number)

    # Create seed manifest for multi-seed resume support
    manifest = SeedManifest(group_path)

    # Create criterion from config
    criterion = run_config.create_criterion()
    use_wandb = run_config.logging == "wandb"

    try:
        for seed in seeds:
            # Skip already-completed seeds
            if manifest.is_complete(seed):
                tqdm.write(f"Seed {seed} already complete, skipping")
                continue

            set_seed(seed)

            model = run_config.create_model().to(device)
            optimizer = run_config.create_optimizer(model)
            scheduler = run_config.create_scheduler(optimizer)

            run_name = f"{seed}"

            if use_wandb:
                wandb.init(
                    project=project,
                    name=run_name,
                    group=group_name,
                    tags=tags,
                    config=run_config.gen_config(),
                )

            # Build callbacks list
            callbacks_list = [
                OptimizerModeCallback(),
                NaNDetectionCallback(),
                GradientMonitorCallback(),
                LossPredictionCallback(run_config.epochs),
                OverfitDetectionCallback(),
            ]
            if use_wandb:
                callbacks_list.append(WandbLoggingCallback())
            else:
                callbacks_list.append(TUILoggingCallback())
            # Always-on callbacks: CSV logging + latest model save
            run_path = f"{group_path}/{run_name}"
            if not os.path.exists(run_path):
                os.makedirs(run_path)
            callbacks_list.append(CSVLoggingCallback(f"{run_path}/metrics.csv"))
            callbacks_list.append(LatestModelCallback(f"{run_path}/latest_model.pt"))

            if run_config.early_stopping_config and run_config.early_stopping_config.enabled:
                callbacks_list.append(
                    EarlyStoppingCallback(
                        patience=run_config.early_stopping_config.patience,
                        mode=run_config.early_stopping_config.mode,
                        min_delta=run_config.early_stopping_config.min_delta,
                    )
                )
            if pruner is not None and trial is not None:
                callbacks_list.append(PrunerCallback(pruner, trial, seed))

            # Create CheckpointManager if enabled
            if run_config.checkpoint_config.enabled:
                config_hash = compute_config_hash(run_config)
                ckpt_manager = CheckpointManager(
                    run_dir=run_path,
                    save_every_n=run_config.checkpoint_config.save_every_n_epochs,
                    keep_last_k=run_config.checkpoint_config.keep_last_k,
                    save_best=run_config.checkpoint_config.save_best,
                    monitor=run_config.checkpoint_config.monitor,
                    mode=run_config.checkpoint_config.mode,
                )
                callbacks_list.append(CheckpointCallback(ckpt_manager, config_hash))

            callback_runner = CallbackRunner(callbacks_list)

            trainer = Trainer(
                model,
                optimizer,
                scheduler,
                criterion=criterion,
                callbacks=callback_runner,
                device=device,
            )

            start_time = time.time()
            val_loss = trainer.train(dl_train, dl_val, epochs=run_config.epochs)
            end_time = time.time()

            # Save model & configs
            torch.save(model.state_dict(), f"{run_path}/model.pt")

            # Save provenance
            save_provenance(run_path, run_config, model, device, start_time, end_time)

            # Mark seed as complete
            manifest.mark_complete(seed, val_loss)

            if use_wandb:
                wandb.finish()

            # Early stopping if loss becomes inf
            if math.isinf(val_loss):
                break

    except optuna.TrialPruned:
        if use_wandb:
            wandb.finish()
        raise
    except Exception as e:
        tqdm.write(f"Runtime error during training: {e}")
        if use_wandb:
            wandb.finish()
        raise optuna.TrialPruned()
    finally:
        # Call trial_finished only once after all seeds are done
        if (
            pruner is not None
            and trial is not None
            and hasattr(pruner, "complete_trial")
        ):
            pruner.complete_trial(trial.number)

    complete_count = manifest.get_complete_count()
    return manifest.get_total_loss() / (complete_count if complete_count > 0 else 1)


# ┌──────────────────────────────────────────────────────────┐
#  For Analyze
# └──────────────────────────────────────────────────────────┘
def select_project():
    runs_path = "runs/"
    projects = [
        d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))
    ]
    projects.sort()
    if not projects:
        raise ValueError(f"No projects found in {runs_path}")

    selected_project = beaupy.select(projects)
    return selected_project


def select_group(project):
    runs_path = f"runs/{project}"
    groups = [
        d for d in os.listdir(runs_path) if os.path.isdir(os.path.join(runs_path, d))
    ]
    groups.sort()
    if not groups:
        raise ValueError(f"No run groups found in {runs_path}")

    selected_group = beaupy.select(groups)
    return selected_group


def select_seed(project, group_name):
    group_path = f"runs/{project}/{group_name}"
    seeds = [
        d for d in os.listdir(group_path) if os.path.isdir(os.path.join(group_path, d))
    ]
    seeds.sort()
    if not seeds:
        raise ValueError(f"No seeds found in {group_path}")

    selected_seed = beaupy.select(seeds)
    return selected_seed


def select_device():
    devices = ["cpu"] + [f"cuda:{i}" for i in range(torch.cuda.device_count())]
    selected_device = beaupy.select(devices)
    return selected_device


def load_model(project, group_name, seed, weights_only=True):
    """
    Load a trained model and its configuration.

    Args:
        project (str): The name of the project.
        group_name (str): The name of the run group.
        seed (str): The seed of the specific run.
        weights_only (bool, optional): If True, only load the model weights without loading the entire pickle file.
                                       This can be faster and use less memory. Defaults to True.

    Returns:
        tuple: A tuple containing the loaded model and its configuration.

    Raises:
        FileNotFoundError: If the config or model file is not found.

    Example usage:
        # Load full model
        model, config = load_model("MyProject", "experiment1", "seed42")

        # Load only weights (faster and uses less memory)
        model, config = load_model("MyProject", "experiment1", "seed42", weights_only=True)
    """
    config_path = f"runs/{project}/{group_name}/config.yaml"
    model_path = f"runs/{project}/{group_name}/{seed}/model.pt"

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found for {project}/{group_name}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found for {project}/{group_name}/{seed}"
        )

    config = RunConfig.from_yaml(config_path)
    model = config.create_model()

    # Use weights_only option in torch.load
    state_dict = torch.load(model_path, map_location="cpu", weights_only=weights_only)
    model.load_state_dict(state_dict)

    return model, config


def load_study(project, study_name):
    """
    Load the best study from an optimization run.

    Args:
        project (str): The name of the project.
        study_name (str): The name of the study.

    Returns:
        optuna.Study: The loaded study object.
    """
    study = optuna.load_study(study_name=study_name, storage=f"sqlite:///{project}.db")
    return study


def load_best_model(project, study_name, weights_only=True):
    """
    Load the best model and its configuration from an optimization study.

    Args:
        project (str): The name of the project.
        study_name (str): The name of the study.

    Returns:
        tuple: A tuple containing the loaded model, its configuration, and the best trial number.
    """
    study = load_study(project, study_name)
    best_trial = study.best_trial
    project_name = project
    group_name = best_trial.user_attrs["group_name"]

    # Select Seed
    seed = select_seed(project_name, group_name)
    best_model, best_config = load_model(
        project_name, group_name, seed, weights_only=weights_only
    )

    return best_model, best_config
