import math

import einops
import torch
from torch import linalg as LA
from torch import nn

from src.constants import EPSILON_FP16  # noqa: I900


class SparsityLoss(nn.Module):
    """
    Compute the sparsity loss of a mask.
    """

    def __init__(self, beta: float, num_masks: int):
        super().__init__()
        self.beta = beta
        self.num_masks = num_masks

    def forward(self, masks: torch.Tensor):
        """
        Args:
            masks: masks of shape (batch_size, num_masks, height, width)
        """
        masks_probs = masks.sum([2, 3]) / (masks.shape[-1] * masks.shape[-2])

        loss = (1 / (torch.sin(masks_probs * math.pi) + EPSILON_FP16)) - 1

        norm_term = math.sin(self.num_masks ** (-self.beta) * math.pi)
        loss *= norm_term

        return loss.mean(dim=1).mean()


class DiversityLoss(nn.Module):
    """
    Compute the diversity loss of masks.
    """

    def __init__(self):
        super().__init__()

    def forward(self, masks: torch.Tensor):
        """
        Args:
            masks: masks of shape (batch_size, num_masks, height, width)
        """

        masks = einops.rearrange(masks, "b n h w -> b n (h w)")
        mask_norms = LA.norm(masks, dim=2)

        identity = torch.eye(masks.shape[1], device=masks.device)
        tops = einops.einsum(masks, masks, "b n hw, b m hw-> b n m")
        bottoms = (
            einops.einsum(mask_norms, mask_norms, "b n, b m-> b n m") + EPSILON_FP16
        )

        loss = ((identity - tops / bottoms) ** 2).mean(dim=(1, 2))
        return loss.mean()


class SmoothBCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, eps: float = 0.1, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps

    def forward(self, input: torch.Tensor, target: torch.Tensor):
        target = target * (1 - 2 * self.eps) + self.eps
        return super().forward(input, target)
