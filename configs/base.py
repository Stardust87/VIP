from dataclasses import dataclass
from pathlib import Path
from typing import Union

from omegaconf import MISSING

from configs.types import Optimizer, Scheduler  # noqa: I900
from src.constants import DATA_PATH  # noqa: I900


@dataclass
class ModelConfig:
    name: str = MISSING
    backbone: str = "resnet50"


@dataclass
class I2PAModelConfig(ModelConfig):
    name: str = "i2pa"
    backbone: str = "convnext_nano"
    pop_alpha: float = 0.7


@dataclass
class FAMEModelConfig(ModelConfig):
    name: str = "fame"
    pop_alpha: float = 0.6
    pop_flat_ratio: float = 0.7
    diversity_alpha: float = 0.1
    sparsity_alpha: float = 0.4
    sparsity_beta: float = 1.0
    smoothing: float = 0.05
    n_masks_pos: int = 3
    n_masks_neu: int = 1
    opt_all_pos: bool = True
    opt_neu: bool = False
    ibot_checkpoint: str = "vit_small_block_1k"
    num_attentions: int = 6
    mask_temperature: float = 1.0


@dataclass
class DataConfig:
    path: Path = DATA_PATH
    batch_size: int = 32
    num_workers: int = 12
    pin_memory: bool = False
    train_size: float = 0.8
    test_size: float = 0.1
    sample_ratio: float = 1.0
    train_ratio: float = 1.0
    image_transform_size: int = 256
    image_target_size: int = 224
    return_masked: bool = False


@dataclass
class OptimizerConfig:
    name: Optimizer = MISSING
    lr: float = 3e-4
    weight_decay: float = 1e-2
    scheduler: Scheduler = Scheduler.ONE_CYCLE
    warmup_ratio: float = 0.1


@dataclass
class TrainConfig:
    max_epochs: int = 5
    max_batches_to_log: int = 5
    log_config: bool = True
    precision: Union[int, str] = "bf16-mixed"


@dataclass
class ExperimentConfig:
    model: ModelConfig = MISSING
    data: DataConfig = MISSING
    optimizer: OptimizerConfig = MISSING
    train: TrainConfig = MISSING
