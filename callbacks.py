from __future__ import annotations
from typing import TYPE_CHECKING, Any
import math

if TYPE_CHECKING:
    from util import Trainer

class TrainingCallback:
    """Base class for training callbacks. Override methods to add behavior."""
    priority: int = 100  # lower number = earlier execution

    def on_train_begin(self, trainer: Trainer, **kwargs: Any) -> None: pass
    def on_train_epoch_begin(self, trainer: Trainer, epoch: int, **kwargs: Any) -> None: pass
    def on_train_step_end(self, trainer: Trainer, batch_idx: int, loss: float, **kwargs: Any) -> None: pass
    def on_val_begin(self, trainer: Trainer, epoch: int, **kwargs: Any) -> None: pass
    def on_val_end(self, trainer: Trainer, epoch: int, val_loss: float, metrics: dict, **kwargs: Any) -> None: pass
    def on_epoch_end(self, trainer: Trainer, epoch: int, train_loss: float, val_loss: float, metrics: dict, **kwargs: Any) -> None: pass
    def on_train_end(self, trainer: Trainer, **kwargs: Any) -> None: pass


class CallbackRunner:
    """Manages and dispatches events to callbacks in priority order."""
    def __init__(self, callbacks: list[TrainingCallback] | None = None):
        self.callbacks = sorted(callbacks or [], key=lambda c: c.priority)

    def add(self, callback: TrainingCallback) -> None:
        self.callbacks.append(callback)
        self.callbacks.sort(key=lambda c: c.priority)

    def fire(self, event: str, **kwargs: Any) -> None:
        for cb in self.callbacks:
            method = getattr(cb, event, None)
            if method:
                method(**kwargs)


class OptimizerModeCallback(TrainingCallback):
    """Switches optimizer train/eval mode for SPlus and ScheduleFree optimizers."""
    priority = 10  # Run early

    def on_train_epoch_begin(self, trainer, epoch, **kwargs):
        if hasattr(trainer.optimizer, 'train') and callable(getattr(trainer.optimizer, 'train')):
            # Only call if it's an optimizer method, not the Module.train()
            if not isinstance(trainer.optimizer, type(trainer.model)):
                trainer.optimizer.train()

    def on_val_begin(self, trainer, epoch, **kwargs):
        if hasattr(trainer.optimizer, 'eval') and callable(getattr(trainer.optimizer, 'eval')):
            if not isinstance(trainer.optimizer, type(trainer.model)):
                trainer.optimizer.eval()


class EarlyStoppingCallback(TrainingCallback):
    """Early stopping based on monitored metric."""
    priority = 90

    def __init__(self, patience: int = 10, mode: str = "min", min_delta: float = 0.0001):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_value: float | None = None
        self.should_stop = False

    def on_val_end(self, trainer, epoch, val_loss, metrics, **kwargs):
        value = val_loss  # Use primary loss
        if self.best_value is None:
            self.best_value = value
            return

        if self.mode == "min":
            improved = value <= self.best_value * (1 - self.min_delta)
        else:
            improved = value >= self.best_value * (1 + self.min_delta)

        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1

        if self.counter >= self.patience:
            self.should_stop = True

    def state_dict(self) -> dict:
        return {
            "counter": self.counter,
            "best_value": self.best_value,
            "should_stop": self.should_stop,
        }

    def load_state_dict(self, state: dict) -> None:
        self.counter = state["counter"]
        self.best_value = state["best_value"]
        self.should_stop = state["should_stop"]


class WandbLoggingCallback(TrainingCallback):
    """Logs metrics to Weights & Biases."""
    priority = 80

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        import wandb
        log_dict = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": trainer.optimizer.param_groups[0]["lr"],
        }
        log_dict.update(metrics)

        if hasattr(trainer, '_max_grad_norm') and trainer._max_grad_norm is not None:
            log_dict["max_grad_norm"] = trainer._max_grad_norm

        if hasattr(trainer, '_overfit_gap_ratio') and trainer._overfit_gap_ratio is not None:
            log_dict["overfit_gap_ratio"] = trainer._overfit_gap_ratio

        if hasattr(trainer, '_loss_prediction') and trainer._loss_prediction is not None:
            log_dict["predicted_final_loss"] = trainer._loss_prediction

        wandb.log(log_dict)

        if epoch % 10 == 0 or epoch == trainer._total_epochs - 1:
            from tqdm import tqdm
            print_str = f"epoch: {epoch}"
            for key, value in log_dict.items():
                if isinstance(value, float):
                    print_str += f", {key}: {value:.4e}"
            tqdm.write(print_str)


class PrunerCallback(TrainingCallback):
    """Reports metrics to Optuna pruner and raises TrialPruned if needed."""
    priority = 85

    def __init__(self, pruner, trial, seed):
        self.pruner = pruner
        self.trial = trial
        self.seed = seed

    def on_val_end(self, trainer, epoch, val_loss, metrics, **kwargs):
        import optuna
        if self.pruner is not None and self.trial is not None:
            self.pruner.report(
                trial_id=self.trial.number,
                seed=self.seed,
                epoch=epoch,
                value=val_loss,
            )
            if self.pruner.should_prune():
                raise optuna.TrialPruned()


class LossPredictionCallback(TrainingCallback):
    """Tracks validation losses and computes predicted final loss."""
    priority = 70

    def __init__(self, max_epochs: int):
        self.val_losses: list[float] = []
        self.max_epochs = max_epochs

    def on_val_end(self, trainer, epoch, val_loss, metrics, **kwargs):
        self.val_losses.append(val_loss)
        if epoch >= 10:
            from util import predict_final_loss
            trainer._loss_prediction = predict_final_loss(self.val_losses, self.max_epochs)
        else:
            trainer._loss_prediction = None


class NaNDetectionCallback(TrainingCallback):
    """Detects NaN losses and signals training to stop."""
    priority = 5  # Run very early

    def __init__(self):
        self.nan_detected = False

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        if math.isnan(train_loss) or math.isnan(val_loss):
            from tqdm import tqdm
            tqdm.write("Early stopping due to NaN loss")
            self.nan_detected = True


class GradientMonitorCallback(TrainingCallback):
    """Monitors gradient norms to detect exploding gradients."""
    priority = 12  # After OptimizerModeCallback (10)

    def __init__(self, warn_threshold: float = 1e4):
        self.warn_threshold = warn_threshold
        self._step_grad_norms: list[float] = []
        self.epoch_max_grad_norms: list[float] = []
        self._current_epoch_max = 0.0

    def on_train_epoch_begin(self, trainer, epoch, **kwargs):
        self._step_grad_norms = []
        self._current_epoch_max = 0.0

    def on_train_step_end(self, trainer, batch_idx, loss, **kwargs):
        total_norm = 0.0
        for param in trainer.model.parameters():
            if param.grad is not None:
                total_norm += param.grad.norm().item() ** 2
        total_norm = total_norm ** 0.5

        self._step_grad_norms.append(total_norm)
        self._current_epoch_max = max(self._current_epoch_max, total_norm)

        if total_norm > self.warn_threshold:
            from tqdm import tqdm
            tqdm.write(
                f"[GradientMonitor] grad norm {total_norm:.2e} "
                f"exceeds threshold {self.warn_threshold:.2e} at batch {batch_idx}"
            )

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        self.epoch_max_grad_norms.append(self._current_epoch_max)
        trainer._max_grad_norm = self._current_epoch_max


class OverfitDetectionCallback(TrainingCallback):
    """Detects overfitting by monitoring train/val loss divergence."""
    priority = 75  # After LossPredictionCallback (70)

    def __init__(self, warmup_epochs: int = 5, window_size: int = 5):
        self.warmup_epochs = warmup_epochs
        self.window_size = window_size
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.overfit_detected = False

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)

        if epoch < self.warmup_epochs or len(self.train_losses) < self.window_size:
            trainer._overfit_gap_ratio = None
            return

        recent_train = self.train_losses[-self.window_size:]
        recent_val = self.val_losses[-self.window_size:]

        # Detect sustained divergence: train decreasing AND val increasing
        train_decreasing = all(
            recent_train[i] >= recent_train[i + 1]
            for i in range(len(recent_train) - 1)
        )
        val_increasing = all(
            recent_val[i] <= recent_val[i + 1]
            for i in range(len(recent_val) - 1)
        )

        if train_decreasing and val_increasing:
            self.overfit_detected = True
            gap_ratio = recent_val[-1] / recent_train[-1] if recent_train[-1] > 0 else float("inf")
            from tqdm import tqdm
            tqdm.write(
                f"[OverfitDetection] epoch {epoch}: "
                f"train_loss decreasing, val_loss increasing "
                f"(gap ratio: {gap_ratio:.2f})"
            )
            trainer._overfit_gap_ratio = gap_ratio
        else:
            trainer._overfit_gap_ratio = None


class CheckpointCallback(TrainingCallback):
    """Saves periodic and best-model checkpoints during training."""
    priority = 95  # After early stopping check

    def __init__(self, checkpoint_manager, config_hash: str = ""):
        self.checkpoint_manager = checkpoint_manager
        self.config_hash = config_hash

    def on_epoch_end(self, trainer, epoch, train_loss, val_loss, metrics, **kwargs):
        early_stopping_state = None
        for cb in trainer.callbacks.callbacks:
            if isinstance(cb, EarlyStoppingCallback):
                early_stopping_state = cb.state_dict()
                break
        self.checkpoint_manager.maybe_save(
            epoch, trainer.model, trainer.optimizer, trainer.scheduler,
            val_loss, metrics, early_stopping_state, self.config_hash,
        )
