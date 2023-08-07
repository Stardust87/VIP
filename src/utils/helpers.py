import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2
from lion_pytorch import Lion

from configs import types as t  # noqa: I900
from configs.base import OptimizerConfig  # noqa: I900
from src.constants import MEANS, STDS  # noqa: I900
from src.utils.scheduler import FlatAnnealLR, TimmCosineLRScheduler  # noqa: I900


def get_optimizer(params, T_max: int, config: OptimizerConfig):
    """
    Build optimizer and scheduler based on the given config.

    Args:
        params: Model parameters.
        T_max: Number of total training steps.
        config: Optimizer config.

    Returns:
        Optimizer and scheduler.
    """

    if config.name == t.Optimizer.ADAMW:
        optimizer = torch.optim.AdamW(
            params=params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.name == t.Optimizer.ADAM:
        optimizer = torch.optim.Adam(
            params=params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.name == t.Optimizer.LION:
        optimizer = Lion(
            params=params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    elif config.name == t.Optimizer.SGD:
        optimizer = torch.optim.SGD(
            params=params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
    else:
        raise NotImplementedError(f"Optimizer {config.name} is not implemented.")

    if config.scheduler is None or config.scheduler == t.Scheduler.CONSTANT:
        return optimizer, None

    if config.scheduler == t.Scheduler.ONE_CYCLE:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer,
            max_lr=config.lr,
            total_steps=T_max,
            pct_start=config.warmup_ratio,
        )
    elif config.scheduler == t.Scheduler.COSINE:
        scheduler = TimmCosineLRScheduler(
            optimizer,
            t_initial=T_max,
            lr_min=1e-7,
            warmup_lr_init=1e-7,
            warmup_t=int(T_max * config.warmup_ratio),
            cycle_limit=1,
            t_in_epochs=False,
            k_decay=1.0,
        )
    elif config.scheduler == t.Scheduler.FLAT_ANNEAL:
        scheduler = FlatAnnealLR(
            optimizer=optimizer,
            T_max=T_max,
        )
    else:
        raise NotImplementedError(f"Scheduler {config.scheduler} is not implemented.")

    return optimizer, scheduler


def get_transforms(transform_size: int = 256, target_size: int = 224):
    """
    Build train and test transforms based on the given sizes.

    Args:
        transform_size: size needed to prepare image for transformations.
        target_size: Size of the image after cropping.

    Returns:
        Train and test transforms.
    """

    train_transform = A.Compose(
        [
            A.LongestMaxSize(transform_size),
            A.PadIfNeeded(transform_size, transform_size),
            A.RandomCrop(target_size, target_size),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.CoarseDropout(max_holes=16, p=0.5, mask_fill_value=0),
            A.Normalize(mean=MEANS, std=STDS),
            ToTensorV2(),
        ]
    )

    test_transform = A.Compose(
        [
            A.LongestMaxSize(target_size),
            A.PadIfNeeded(target_size, target_size),
            A.Normalize(mean=MEANS, std=STDS),
            ToTensorV2(),
        ],
    )

    return train_transform, test_transform


def get_mask_transforms(transform_size: int = 256, target_size: int = 224):
    mask_transform = A.Compose(
        [
            A.LongestMaxSize(transform_size),
            A.PadIfNeeded(transform_size, transform_size),
            A.RandomCrop(target_size, target_size),
            A.HorizontalFlip(p=0.5),
            A.HueSaturationValue(
                hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.5
            ),
            A.CoarseDropout(
                min_holes=12,
                max_holes=18,
                min_height=24,
                max_height=36,
                min_width=24,
                max_width=36,
                p=1.0,
                mask_fill_value=0,
            ),
            A.Normalize(mean=MEANS, std=STDS),
            ToTensorV2(),
        ]
    )

    return mask_transform, None
