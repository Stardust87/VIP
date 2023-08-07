from dataclasses import dataclass

from configs import types as t  # noqa: I900
from configs.base import (  # noqa: I900
    DataConfig,
    ExperimentConfig,
    FAMEModelConfig,
    I2PAModelConfig,
    ModelConfig,
    OptimizerConfig,
    TrainConfig,
)


@dataclass
class I2PAConfig(ExperimentConfig):
    model: ModelConfig = I2PAModelConfig(
        backbone="resnet18",
    )
    data: DataConfig = DataConfig(
        # batch_size=96,
        batch_size=64,
        sample_ratio=1.0,
        pin_memory=False,
        num_workers=12,
        return_masked=True,
    )
    optimizer: OptimizerConfig = OptimizerConfig(
        name=t.Optimizer.ADAMW,
        # lr=4e-5,
        lr=1e-4,
        weight_decay=3e-4,
        scheduler=t.Scheduler.COSINE,
        warmup_ratio=0.1,
    )
    train: TrainConfig = TrainConfig(
        max_epochs=4,
        max_batches_to_log=0,
        log_config=True,
        precision="bf16-mixed",
    )


@dataclass
class FAMEConfig(ExperimentConfig):
    model: ModelConfig = FAMEModelConfig(
        name="fame",
        # backbone="convnext_nano",
        backbone="resnet18",
        pop_alpha=0.7,
        diversity_alpha=0.05,
        sparsity_alpha=0.2,
        n_masks_pos=3,
        n_masks_neu=0,
        opt_all_pos=False,
        opt_neu=False,
        ibot_checkpoint="vit_small_block_1k",
        num_attentions=6,
        mask_temperature=1.0,
    )
    data: DataConfig = DataConfig(
        # batch_size=96,
        batch_size=64,
        sample_ratio=1.0,
        train_ratio=1.0,
        pin_memory=False,
        num_workers=12,
    )
    optimizer: OptimizerConfig = OptimizerConfig(
        name=t.Optimizer.ADAMW,
        lr=1e-4,
        # lr=4e-5,
        weight_decay=3e-4,
        scheduler=t.Scheduler.COSINE,
        warmup_ratio=0.1,
    )
    optimizer_mask: OptimizerConfig = OptimizerConfig(
        name=t.Optimizer.ADAMW,
        lr=1e-5,
        weight_decay=1e-6,
        scheduler=t.Scheduler.COSINE,
        warmup_ratio=0.1,
    )
    train: TrainConfig = TrainConfig(
        max_epochs=4,
        max_batches_to_log=1,
        log_config=True,
        precision="bf16-mixed",
    )
