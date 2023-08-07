import math

import torch
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LambdaLR


class FlatAnnealLR(torch.optim.lr_scheduler.LambdaLR):
    """
    Schedule LR linear anneal from 1.0 to `eta_min` after `T_max * flat_ratio` steps.
    """

    def __init__(
        self,
        optimizer,
        T_max: int,
        flat_ratio: float = 0.75,
        eta_min: float = 1e-7,
        **kwargs
    ):
        self.T_max = T_max
        self.flat_ratio = flat_ratio
        self.eta_min = eta_min

        super().__init__(optimizer, self.flat_anneal, **kwargs)

    def flat_anneal(self, step):
        if step < self.T_max * self.flat_ratio:
            return 1.0

        return (
            self.eta_min
            + (1.0 - self.eta_min)
            * (
                1.0
                + math.cos(
                    math.pi
                    * (step - self.T_max * self.flat_ratio)
                    / (self.T_max * (1.0 - self.flat_ratio))
                )
            )
            / 2.0
        )


class CosineAnnealAlpha:
    """
    Schedule cosine anneal to alpha for first `T_max * (1-flat_ratio)` steps.
    """

    def __init__(
        self,
        T_max: int,
        alpha: float = 0.5,
        flat_ratio: float = 0.75,
    ):
        self.T_max = T_max
        self.alpha = alpha
        self.flat_ratio = flat_ratio

        self.current_step = -1

    def step(self):
        self.current_step += 1
        if self.current_step < self.T_max * (1.0 - self.flat_ratio):
            return (
                self.alpha
                + (1.0 - self.alpha)
                * (
                    1.0
                    + math.cos(
                        math.pi
                        * self.current_step
                        / (self.T_max * (1.0 - self.flat_ratio))
                    )
                )
                / 2.0
            )

        return self.alpha


class TimmCosineLRScheduler(LambdaLR):
    def __init__(self, optimizer, **kwargs):
        self.init_lr = optimizer.param_groups[0]["lr"]
        self.scheduler = CosineLRScheduler(optimizer, **kwargs)
        super().__init__(optimizer, self)

    def __call__(self, epoch):
        desired_lr = self.scheduler._get_lr(epoch)[0]
        mult = desired_lr / self.init_lr
        return mult
