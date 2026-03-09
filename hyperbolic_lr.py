import math


class HyperbolicLR:
    """
    HyperbolicLR

    Args:
        optimizer: Optimizer
        upper_bound: Upper bound on various max_iters
        max_iter: Maximum number of iterations
        init_lr: Initial learning rate
        infimum_lr: The infimum of the hyperbolic learning rate
    """

    def __init__(self, optimizer, upper_bound=1000, max_iter=100, infimum_lr=1e-6):
        init_lr = optimizer.param_groups[0]["lr"]
        if upper_bound < max_iter:
            raise ValueError("upper_bound must be greater than max_iter")
        elif infimum_lr >= init_lr:
            raise ValueError("infimum_lr must be less than init_lr")
        self._optimizer = optimizer
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr
        self.delta_lr = init_lr - infimum_lr
        self.iter = 0

    def state_dict(self):
        return {
            "iter": self.iter,
            "init_lr": self.init_lr,
            "infimum_lr": self.infimum_lr,
            "upper_bound": self.upper_bound,
            "max_iter": self.max_iter,
            "delta_lr": self.delta_lr,
        }

    def load_state_dict(self, state_dict):
        self.iter = state_dict["iter"]
        self.init_lr = state_dict["init_lr"]
        self.infimum_lr = state_dict.get("infimum_lr", self.init_lr - state_dict["delta_lr"])
        self.upper_bound = state_dict["upper_bound"]
        self.max_iter = state_dict["max_iter"]
        self.delta_lr = state_dict["delta_lr"]

    def step(self):
        """
        Update the learning rate
        """
        self._update_learning_rate()

    def zero_grad(self):
        """
        Zero out the gradients with the inner optimizer
        """
        self._optimizer.zero_grad()

    def get_last_lr(self):
        """
        Get the last learning rates from the inner optimizer
        """
        return [param_group["lr"] for param_group in self._optimizer.param_groups]

    def _get_lr(self):
        """
        Get the learning rate
        """
        x = self.iter
        N = self.max_iter
        U = self.upper_bound
        return self.init_lr + self.delta_lr * (
            math.sqrt((N - x) / U * (2 - (N + x) / U)) - math.sqrt(N / U * (2 - N / U))
        )

    def _update_learning_rate(self):
        """
        Update the learning rate
        """
        self.iter += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr


class ExpHyperbolicLR:
    """
    ExpHyperbolicLR

    Args:
        optimizer: Optimizer
        upper_bound: Upper bound on various max_iters
        max_iter: Maximum number of iterations
        init_lr: Initial learning rate
        infimum_lr: The infimum of the hyperbolic learning rate
    """

    def __init__(self, optimizer, upper_bound=1000, max_iter=100, infimum_lr=1e-6):
        init_lr = optimizer.param_groups[0]["lr"]
        if upper_bound < max_iter:
            raise ValueError("upper_bound must be greater than max_iter")
        elif infimum_lr >= init_lr:
            raise ValueError("infimum_lr must be less than init_lr")
        self._optimizer = optimizer
        self.upper_bound = upper_bound
        self.max_iter = max_iter
        self.init_lr = init_lr
        self.infimum_lr = infimum_lr
        self.lr_ratio = init_lr / infimum_lr
        self.iter = 0

    def state_dict(self):
        return {
            "iter": self.iter,
            "init_lr": self.init_lr,
            "infimum_lr": self.infimum_lr,
            "upper_bound": self.upper_bound,
            "max_iter": self.max_iter,
            "lr_ratio": self.lr_ratio,
        }

    def load_state_dict(self, state_dict):
        self.iter = state_dict["iter"]
        self.init_lr = state_dict["init_lr"]
        self.infimum_lr = state_dict.get("infimum_lr", self.init_lr / state_dict["lr_ratio"])
        self.upper_bound = state_dict["upper_bound"]
        self.max_iter = state_dict["max_iter"]
        self.lr_ratio = state_dict["lr_ratio"]

    def step(self):
        """
        Update the learning rate
        """
        self._update_learning_rate()

    def zero_grad(self):
        """
        Zero out the gradients with the inner optimizer
        """
        self._optimizer.zero_grad()

    def get_last_lr(self):
        """
        Get the last learning rates from the inner optimizer
        """
        return [param_group["lr"] for param_group in self._optimizer.param_groups]

    def _get_lr(self):
        """
        Get the learning rate
        """
        x = self.iter
        N = self.max_iter
        U = self.upper_bound
        return self.init_lr * self.lr_ratio ** (
            math.sqrt((N - x) / U * (2 - (N + x) / U)) - math.sqrt(N / U * (2 - N / U))
        )

    def _update_learning_rate(self):
        """
        Update the learning rate
        """
        self.iter += 1
        lr = self._get_lr()
        for param_group in self._optimizer.param_groups:
            param_group["lr"] = lr
