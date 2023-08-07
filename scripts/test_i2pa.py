import argparse

import pytorch_lightning as pl
import torch
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger

from src.constants import (  # noqa: I900
    DATA_COLAB_PATH,
    DATA_MAC_PATH,
    LOG_DIR,
    PROJECT_NAME,
)
from src.data import InfluencerPairDataModule  # noqa: I900
from src.utils import load_checkpoint  # noqa: I900
from src.utils.progress import LitProgressBar  # noqa: I900

parser = argparse.ArgumentParser()
parser.add_argument("--run-id", type=str, required=True)
parser.add_argument("-m", "--mac", action="store_true", default=False)
parser.add_argument("-c", "--colab", action="store_true", default=False)
parser.add_argument("-o", "--offline", action="store_true", default=False)
args = parser.parse_args()

DEVICE = "gpu" if torch.cuda.is_available() else "cpu"


model, cfg = load_checkpoint(args.run_id)


if args.mac:
    DEVICE = "mps"
    cfg.data.batch_size = 8
    cfg.data.num_workers = 0
    cfg.data.path = DATA_MAC_PATH
    cfg.data.sample_ratio = 1.0
    cfg.train.precision = 32
    cfg.train.max_batches_to_log = 6


if args.colab:
    torch.set_float32_matmul_precision("high")
    cfg.data.batch_size = 128
    cfg.data.num_workers = 12
    cfg.data.path = DATA_COLAB_PATH
    cfg.data.sample_ratio = 1.0
    cfg.train.precision = "16-mixed"
    if DEVICE == "cpu":
        cfg.train.precision = 32
        cfg.data.num_workers = 0


print(OmegaConf.to_yaml(cfg))

influencer_dm = InfluencerPairDataModule(cfg.data)


if not args.colab:
    callbacks = [RichProgressBar()]
else:
    callbacks = [LitProgressBar()]


logger = WandbLogger(
    project=PROJECT_NAME,
    name=model.name,
    save_dir=str(LOG_DIR),
    offline=args.offline,
    id=args.run_id,
    resume="must",
)

trainer = pl.Trainer(
    accelerator=DEVICE,
    devices=1,
    callbacks=callbacks,
    logger=logger,
    precision=cfg.train.precision,
)

trainer.test(model, influencer_dm)
