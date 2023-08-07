import argparse

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
)
from pytorch_lightning.loggers import WandbLogger

from configs import FAMEConfig as cfg  # noqa: I900
from src.constants import (  # noqa: I900
    DATA_COLAB_PATH,
    DATA_MAC_PATH,
    LOG_DIR,
    PROJECT_NAME,
)
from src.data import InfluencerPairDataModule  # noqa: I900
from src.models import LitFAME  # noqa: I900
from src.utils.progress import LitProgressBar  # noqa: I900

parser = argparse.ArgumentParser()
parser.add_argument("-m", "--mac", action="store_true", default=False)
parser.add_argument("-c", "--colab", action="store_true", default=False)
parser.add_argument("-o", "--offline", action="store_true", default=False)
args = parser.parse_args()

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

cfg = OmegaConf.structured(cfg)

torch.set_float32_matmul_precision("high")

if args.mac:
    DEVICE = "mps"
    cfg.data.batch_size = 8
    cfg.data.num_workers = 0
    cfg.data.path = DATA_MAC_PATH
    cfg.data.sample_ratio = 0.05
    cfg.train.precision = 32
    cfg.model.ibot_checkpoint = "dev_vit_small_block_1k"


if args.colab:
    cfg.data.batch_size = 96 if cfg.model.opt_all_pos else 128
    cfg.data.num_workers = 12
    cfg.data.path = DATA_COLAB_PATH
    cfg.data.sample_ratio = 1.0
    cfg.model.ibot_checkpoint = "colab_vit_small_block_1k"
    cfg.train.precision = "16-mixed"
    if DEVICE == "cpu":
        cfg.train.precision = 32
        cfg.data.num_workers = 0


print(OmegaConf.to_yaml(cfg))

influencer_dm = InfluencerPairDataModule(cfg.data)

model = LitFAME(cfg)


class CustomModelCheckpoint(ModelCheckpoint):
    def __init__(self, model_name: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.filename = model_name + "-e-{epoch:02d}-gs-{global_step}-acc-{val/acc:.4f}"

    def on_validation_end(self, trainer, pl_module):
        trainer.callback_metrics.update(
            {"global_step": int(trainer.global_step / 2 - 1)}
        )
        super().on_validation_end(trainer, pl_module)

    def format_checkpoint_name(self, *args, **kwargs):
        return (
            super().format_checkpoint_name(*args, **kwargs).replace("global_step=", "")
        )


callbacks = [
    CustomModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=8,
        save_last=True,
        auto_insert_metric_name=False,
        model_name=model.name,
    ),
    LearningRateMonitor(logging_interval="step"),
]

if not args.colab:
    callbacks.append(RichProgressBar())
else:
    callbacks.append(LitProgressBar())


logger = WandbLogger(
    project=PROJECT_NAME,
    name=model.name,
    save_dir=str(LOG_DIR),
    offline=args.offline,
)

trainer = pl.Trainer(
    accelerator=DEVICE,
    devices=1,
    callbacks=callbacks,
    logger=logger,
    max_epochs=cfg.train.max_epochs,
    precision=cfg.train.precision,
    val_check_interval=0.25,
)

trainer.fit(model, influencer_dm)
