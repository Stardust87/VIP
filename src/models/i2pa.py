import torch
from einops import rearrange
from torch import nn
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from configs import I2PAConfig  # noqa: I900
from src.models.encoders import TimmEncoder  # noqa: I900
from src.utils.helpers import get_optimizer  # noqa: I900
from src.utils.image import inverse_normalize  # noqa: I900

from .base import LitBaseModel


class LitI2PA(LitBaseModel):
    def __init__(self, cfg: I2PAConfig):
        super().__init__(cfg=cfg)

        self.model = TimmEncoder(self.cfg.model.backbone)
        self.loss_fn = nn.BCEWithLogitsLoss()

        self.train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

    def forward(self, img_a, img_b):
        score_a = self.model(img_a)
        score_b = self.model(img_b)

        return score_a, score_b

    def mask_step(self, img_a, img_b, mask_a, mask_b):
        img_a, img_b, labels = self.shuffle_pairs(img_a, img_b)
        labels = rearrange(labels, "b -> b 1 1 1").bool()
        mask_a_ = torch.where(labels, mask_a, mask_b)
        labels = labels.squeeze().float()

        score_a, score_b = self(img_a, img_b)
        score_diff_prob = (score_a - score_b).squeeze()

        loss_img = self.loss_fn(score_diff_prob, labels)

        score_mask_a, score_mask_b = self(mask_a_, img_b)
        score_diff_prob_mask = (score_mask_a - score_mask_b).squeeze()

        loss_mask = self.loss_fn(score_diff_prob_mask, labels)

        loss = (
            self.cfg.model.pop_alpha * loss_img
            + (1 - self.cfg.model.pop_alpha) * loss_mask
        )

        return loss, score_diff_prob, labels

    def shared_step(self, img_a, img_b, train=False):
        if train:
            img_a, img_b, labels = self.shuffle_pairs(img_a, img_b)
        else:
            labels = torch.ones(img_a.shape[0], device=img_a.device)

        score_a, score_b = self(img_a, img_b)
        score_diff_prob = (score_a - score_b).squeeze()

        return (self.loss_fn(score_diff_prob, labels), score_diff_prob, labels)

    def training_step(self, batch, batch_idx):
        img_high, img_low, mask_high, mask_low = (
            batch["image_high"],
            batch["image_low"],
            batch["mask_high"],
            batch["mask_low"],
        )
        loss, score_diff_prob, labels = self.mask_step(
            img_high, img_low, mask_high, mask_low
        )

        self.train_acc(score_diff_prob, labels)
        self.log("train/loss", loss, on_step=True, on_epoch=False, prog_bar=False)
        self.log(
            "train/acc", self.train_acc, on_step=True, on_epoch=False, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        img_high, img_low = batch["image_high"], batch["image_low"]
        loss, score_diff_prob, labels = self.shared_step(img_high, img_low)

        pop_probs = torch.sigmoid(score_diff_prob)
        pop_labels = torch.cat([labels, 1 - labels], dim=0)
        pop_probs = torch.cat([pop_probs, 1 - pop_probs], dim=0)

        self.val_acc(score_diff_prob, labels)
        self.val_auroc(pop_probs, pop_labels)
        self.log_dict(
            {
                "val/loss": loss,
                "val/acc": self.val_acc,
                "val/auroc": self.val_auroc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        if batch_idx < self.cfg.train.max_batches_to_log:
            self.log_predictions(
                inverse_normalize(img_high).cpu(),
                inverse_normalize(img_low).cpu(),
                score_diff_prob.cpu(),
                batch,
            )

        return loss

    def test_step(self, batch, batch_idx):
        img_high, img_low = batch["image_high"], batch["image_low"]
        loss, score_diff_prob, labels, *_ = self.shared_step(img_high, img_low)

        pop_probs = torch.sigmoid(score_diff_prob)
        pop_labels = torch.cat([labels, 1 - labels], dim=0)
        pop_probs = torch.cat([pop_probs, 1 - pop_probs], dim=0)

        self.test_acc(score_diff_prob, labels)
        self.test_auroc(pop_probs, pop_labels)
        self.log_dict(
            {
                "test/loss": loss,
                "test/acc": self.test_acc,
                "test/auroc": self.test_auroc,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def predict_step(self, batch, batch_idx):
        return self.model(batch)

    def configure_optimizers(self):
        optimizer, scheduler = get_optimizer(
            self.parameters(),
            self.trainer.estimated_stepping_batches,
            self.cfg.optimizer,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
