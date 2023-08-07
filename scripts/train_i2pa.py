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

from configs import I2PAConfig as cfg  # noqa: I900
from src.constants import (  # noqa: I900
    DATA_COLAB_PATH,
    DATA_MAC_PATH,
    LOG_DIR,
    PROJECT_NAME,
)
from src.data import InfluencerPairDataModule  # noqa: I900
from src.models import LitI2PA  # noqa: I900
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

if args.colab:
    torch.set_float32_matmul_precision("high")
    cfg.data.batch_size = 96 if cfg.model.opt_all_pos else 128
    cfg.data.num_workers = 12
    cfg.data.path = DATA_COLAB_PATH
    cfg.data.sample_ratio = 1.0
    cfg.train.precision = "16-mixed"
    if DEVICE == "cpu":
        cfg.train.precision = 32
        cfg.data.num_workers = 0


print(OmegaConf.to_yaml(cfg))

influencer_dm = InfluencerPairDataModule(cfg.data)

model = LitI2PA(cfg)

callbacks = [
    ModelCheckpoint(
        monitor="val/acc",
        mode="max",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
        filename=model.name + "-{epoch:02d}-acc-{val/acc:.4f}",
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
