import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from einops import rearrange
from omegaconf import OmegaConf
from PIL import Image

from configs.base import ExperimentConfig  # noqa: I900
from src.constants import LOG_DIR  # noqa: I900


class LitBaseModel(pl.LightningModule):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.cfg = cfg
        self.table_data = None

    @property
    def name(self):
        backbone = self.cfg.model.backbone.split(".")[0]
        return f"{self.cfg.model.name}-{backbone}"

    def shuffle_pairs(self, img_high: torch.Tensor, img_low: torch.Tensor):
        batch_size = img_high.shape[0]

        with torch.no_grad():
            labels = (torch.rand(batch_size) > 0.5).to(img_high.device)
            labels = rearrange(labels, "b -> b 1 1 1")

            img_a = torch.where(labels, img_high, img_low)
            img_b = torch.where(labels, img_low, img_high)

            labels = labels.squeeze()

        return img_a, img_b, labels.float()

    def log_predictions(
        self,
        img_high: np.ndarray,
        img_low: np.ndarray,
        dist_high: list[Image.Image],
        dist_low: list[Image.Image],
        probability: float,
        meta: dict,
    ) -> None:
        batch_size = img_high.shape[0]

        for ind in range(batch_size):
            self.table_data.append(
                [
                    meta["category"][ind],
                    wandb.Image(img_high[ind]),
                    wandb.Image(img_low[ind]),
                    wandb.Image(dist_high[ind], mode="RGB"),
                    wandb.Image(dist_low[ind], mode="RGB"),
                    probability[ind].item(),
                    meta["shortcode_high"][ind],
                    meta["shortcode_low"][ind],
                ]
            )

    def on_validation_start(self):
        if self.cfg.train.max_batches_to_log > 0:
            self.table_data = []

    def on_test_start(self):
        self.on_validation_start()

    def on_validation_end(self, table_name: str = "predictions"):
        if self.cfg.train.max_batches_to_log > 0:
            self.logger.log_table(
                f"{table_name}_{self.logger.experiment.id}",
                data=self.table_data,
                columns=[
                    "category",
                    "image_high",
                    "image_low",
                    "distribution_high",
                    "distribution_low",
                    "probability",
                    "shortcode_high",
                    "shortcode_low",
                ],
            )

    def on_test_end(self):
        self.on_validation_end(table_name="test_predictions")

    def on_train_start(self):
        if self.cfg.train.log_config:
            self.logger.log_hyperparams(self.cfg)

            config_path = (
                LOG_DIR
                / self.logger.experiment.project
                / self.logger.experiment.id
                / "config.yaml"
            )
            config_path.parent.mkdir(parents=True, exist_ok=True)
            OmegaConf.save(self.cfg, config_path)
