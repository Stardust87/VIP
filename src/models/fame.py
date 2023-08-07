import matplotlib
import numpy as np
import torch
from einops import rearrange
from torchmetrics.classification import BinaryAccuracy, BinaryAUROC

from configs import FAMEConfig  # noqa: I900
from src.models.base import LitBaseModel  # noqa: I900
from src.models.encoders import TimmEncoder  # noqa: I900
from src.models.ibot import iBOT  # noqa: I900
from src.models.unet import Unet  # noqa: I900
from src.utils.helpers import get_optimizer  # noqa: I900
from src.utils.image import inverse_normalize  # noqa: I900
from src.utils.losses import (  # noqa: I900
    DiversityLoss,
    SmoothBCEWithLogitsLoss,
    SparsityLoss,
)
from src.utils.scheduler import CosineAnnealAlpha  # noqa: I900
from src.utils.visualize import plot_masks, plot_masks_dist  # noqa: I900

matplotlib.use("Agg")


class LitFAME(LitBaseModel):
    def __init__(self, cfg: FAMEConfig):
        super().__init__(cfg=cfg)
        self.pop_alpha = cfg.model.pop_alpha
        self.diversity_alpha = cfg.model.diversity_alpha
        self.sparsity_alpha = cfg.model.sparsity_alpha

        self.n_masks_pos = cfg.model.n_masks_pos
        self.n_masks_neu = cfg.model.n_masks_neu
        self.n_masks = self.n_masks_pos + self.n_masks_neu
        self.opt_all_pos = cfg.model.opt_all_pos
        self.opt_neu = cfg.model.opt_neu
        self.image_size = cfg.data.image_target_size

        self.popularity_loss = SmoothBCEWithLogitsLoss(eps=cfg.model.smoothing)
        self.diversity_loss = DiversityLoss()
        self.sparsity_loss = SparsityLoss(
            beta=cfg.model.sparsity_beta, num_masks=self.n_masks
        )

        self.train_acc = BinaryAccuracy()
        self.mask_train_acc = BinaryAccuracy()
        self.val_acc = BinaryAccuracy()
        self.test_acc = BinaryAccuracy()

        self.val_auroc = BinaryAUROC()
        self.test_auroc = BinaryAUROC()

        self.encoder = TimmEncoder(cfg.model.backbone)
        self.ibot = iBOT(interpolate=False, checkpoint=cfg.model.ibot_checkpoint)
        self.num_attentions = cfg.model.num_attentions
        self.mask_temperature = cfg.model.mask_temperature

        self.mask_encoder = Unet(
            num_blocks=int(np.log2(self.image_size) - 1),
            img_size=self.image_size,
            filter_start=40,
            in_chnls=3,
            out_chnls=self.n_masks,
            norm="gn",
            num_attentions=self.num_attentions,
            mask_temperature=self.mask_temperature,
        )

        self.automatic_optimization = False

    def configure_optimizers(self):
        self.pop_alpha_sch = CosineAnnealAlpha(
            self.trainer.estimated_stepping_batches, alpha=self.pop_alpha
        )
        self.pop_alpha = self.pop_alpha_sch.step()

        pop_opt, pop_sch = get_optimizer(
            self.encoder.parameters(),
            self.trainer.estimated_stepping_batches,
            self.cfg.optimizer,
        )

        mask_opt, mask_sch = get_optimizer(
            self.mask_encoder.parameters(),
            self.trainer.estimated_stepping_batches,
            self.cfg.optimizer_mask,
        )

        return (
            {
                "optimizer": pop_opt,
                "lr_scheduler": {"scheduler": pop_sch, "name": "lr-encoder"},
            },
            {
                "optimizer": mask_opt,
                "lr_scheduler": {"scheduler": mask_sch, "name": "lr-mask"},
            },
        )

    def forward(self, x):
        return self.encoder(x)

    def flip_gradient(self, state: bool):
        for param in self.encoder.parameters():
            param.requires_grad = state

    def popularity_forward(self, img_a, img_b):
        score_a = self(img_a)
        score_b = self(img_b)

        return (score_a - score_b).squeeze(-1)

    def mask_forward(self, img, attentions):
        masks = self.mask_encoder(img, attentions)
        return masks

    def sample_masks(self, masks):
        if self.n_masks_neu == 1:
            masks_pos, mask_neutral = (
                masks[:, :-1],
                masks[:, -1:],
            )
        else:
            masks_pos, mask_neutral = masks, None

        if self.opt_all_pos:
            return masks_pos, mask_neutral
        else:
            chosen_mask = torch.randint(
                low=0, high=self.n_masks_pos, size=(masks.shape[0],)
            )
            mask_pos = masks_pos[torch.arange(masks.shape[0]), chosen_mask]
            return mask_pos.unsqueeze(1), mask_neutral

    def popularity_step(self, img_a, img_b, att_a, att_b, labels, pop_opt):
        img_pop_score = self.popularity_forward(img_a, img_b)

        masks_a = self.mask_forward(img_a, att_a)
        mask_pos_a, _ = self.sample_masks(masks_a)

        mask_pop_loss = 0
        for mask_pos in torch.chunk(mask_pos_a, chunks=mask_pos_a.shape[1], dim=1):
            pop_score_ = self.popularity_forward(img_a * (1 - mask_pos), img_b)
            pop_loss_ = self.popularity_loss(pop_score_, labels)
            mask_pop_loss = mask_pop_loss + pop_loss_

        mask_pop_loss = mask_pop_loss / mask_pos_a.shape[1]

        pop_loss = (
            self.pop_alpha * self.popularity_loss(img_pop_score, labels)
            + (1 - self.pop_alpha) * mask_pop_loss
        )

        self.train_acc(img_pop_score, labels)

        pop_opt.zero_grad()
        self.manual_backward(pop_loss)
        pop_opt.step()

        return pop_loss, img_pop_score

    def get_adversarial_images(self, img_a, img_b, att_a, att_b, outputs_mask):
        outputs_mask = rearrange(outputs_mask, "b -> b 1 1 1")

        img_high = torch.where(outputs_mask, img_a, img_b)
        img_low = torch.where(outputs_mask, img_b, img_a)

        att_high = torch.where(outputs_mask, att_a, att_b)
        att_low = torch.where(outputs_mask, att_b, att_a)

        return img_high, img_low, att_high, att_low

    def mask_step(self, img_a, img_b, att_a, att_b, outputs_mask, mask_opt):
        fake_labels = torch.ones_like(
            outputs_mask, dtype=torch.long, device=outputs_mask.device
        )

        img_high, img_low, att_high, att_low = self.get_adversarial_images(
            img_a, img_b, att_a, att_b, outputs_mask
        )

        masks_high = self.mask_forward(img_high, att_high)
        masks_low = self.mask_forward(img_low, att_low)

        masks_high_pos, mask_high_neutral = self.sample_masks(masks_high)

        mask_pop_loss = 0
        for mask_high_pos in torch.chunk(
            masks_high_pos, chunks=masks_high_pos.shape[1], dim=1
        ):
            adv_pop_score_ = self.popularity_forward(
                img_high * (1 - mask_high_pos), img_low
            )
            mask_pop_loss = mask_pop_loss + self.popularity_loss(
                adv_pop_score_, 1 - fake_labels
            )
            self.mask_train_acc(adv_pop_score_, 1 - fake_labels)

        if mask_high_neutral is not None and self.opt_neu:
            chosen_mask = torch.randint(
                low=0, high=masks_high_pos.shape[1], size=(masks_high.shape[0],)
            )
            chosen_pos_mask = masks_high_pos[
                torch.arange(masks_high.shape[0]), chosen_mask
            ]

            adv_pop_score_ = self.popularity_forward(
                img_high * (1 - mask_high_neutral),
                img_high * (1 - chosen_pos_mask.unsqueeze(1)),
            )
            mask_pop_loss = mask_pop_loss + self.popularity_loss(
                adv_pop_score_, fake_labels
            )

        mask_pop_loss = mask_pop_loss / (masks_high_pos.shape[1] + self.n_masks_neu)

        diversity = self.diversity_loss(torch.cat([masks_high, masks_low], dim=0))
        sparsity = self.sparsity_loss(torch.cat([masks_high, masks_low], dim=0))

        mask_loss = (
            mask_pop_loss
            + self.diversity_alpha * diversity
            + self.sparsity_alpha * sparsity
        )

        mask_opt.zero_grad()
        self.manual_backward(mask_loss)
        mask_opt.step()

        return mask_loss, mask_pop_loss, diversity, sparsity

    def training_step(self, batch, batch_idx):
        (
            img_high,
            img_low,
        ) = (batch["image_high"], batch["image_low"])

        with torch.no_grad():
            img_a, img_b, labels = self.shuffle_pairs(img_high, img_low)
            att_a = self.ibot(img_a)
            att_b = self.ibot(img_b)

        # optimizers and lr schedulers
        pop_opt, mask_opt = self.optimizers()
        pop_sch, mask_sch = self.lr_schedulers()

        # popularity
        self.flip_gradient(True)
        pop_loss, pop_score = self.popularity_step(
            img_a, img_b, att_a, att_b, labels, pop_opt
        )

        # masking
        self.flip_gradient(False)

        outputs_mask = torch.sigmoid(pop_score) > 0.5
        mask_loss, mask_pop_loss, diversity, sparsity = self.mask_step(
            img_a, img_b, att_a, att_b, outputs_mask, mask_opt
        )

        self.log_dict(
            {
                "train/loss": pop_loss,
                "train/acc": self.train_acc,
                "pop_alpha": self.pop_alpha,
                "mask/loss": mask_loss,
                "mask/pop_loss": mask_pop_loss,
                "mask/diversity": diversity,
                "mask/sparsity": sparsity,
                "mask/acc": self.mask_train_acc,
            },
            on_step=True,
            on_epoch=False,
            prog_bar=False,
        )

        pop_sch.step()
        mask_sch.step()
        self.pop_alpha = self.pop_alpha_sch.step()

    def validation_step(self, batch, batch_idx):
        img_high, img_low, meta = batch["image_high"], batch["image_low"], batch
        labels = torch.ones(img_high.shape[0], device=img_low.device)

        pop_score = self.popularity_forward(img_high, img_low)
        loss = self.popularity_loss(pop_score, labels)

        pop_probs = torch.sigmoid(pop_score)
        pop_labels = torch.cat([labels, 1 - labels], dim=0)
        pop_probs = torch.cat([pop_probs, 1 - pop_probs], dim=0)

        self.val_acc(pop_score, labels)
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
            img_to_process = min(img_high.shape[0], 24)
            img_high = img_high[:img_to_process]
            img_low = img_low[:img_to_process]

            att_high = self.ibot(img_high)
            att_low = self.ibot(img_low)

            masks_high = self.mask_forward(img_high, att_high)
            masks_low = self.mask_forward(img_low, att_low)

            img_high = inverse_normalize(img_high)
            img_low = inverse_normalize(img_low)

            plot_neutral = self.n_masks_neu == 1

            with np.errstate(divide="ignore", invalid="ignore"):
                self.log_predictions(
                    self.visualize_masks(img_high, masks_high),
                    self.visualize_masks(img_low, masks_low),
                    [
                        plot_masks_dist(mask, plot_neutral=plot_neutral)
                        for mask in masks_high
                    ],
                    [
                        plot_masks_dist(mask, plot_neutral=plot_neutral)
                        for mask in masks_low
                    ],
                    torch.sigmoid(pop_score).cpu(),
                    meta,
                )

    def test_step(self, batch, batch_idx):
        img_high, img_low, meta = batch["image_high"], batch["image_low"], batch
        labels = torch.ones(img_high.shape[0], device=img_low.device)

        pop_score = self.popularity_forward(img_high, img_low)
        loss = self.popularity_loss(pop_score, labels)

        pop_probs = torch.sigmoid(pop_score)
        pop_labels = torch.cat([labels, 1 - labels], dim=0)
        pop_probs = torch.cat([pop_probs, 1 - pop_probs], dim=0)

        self.test_acc(pop_score, labels)
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

        if batch_idx < self.cfg.train.max_batches_to_log:
            img_to_process = min(img_high.shape[0], 48)
            img_high = img_high[:img_to_process]
            img_low = img_low[:img_to_process]

            att_high = self.ibot(img_high)
            att_low = self.ibot(img_low)

            masks_high = self.mask_forward(img_high, att_high)
            masks_low = self.mask_forward(img_low, att_low)

            img_high = inverse_normalize(img_high)
            img_low = inverse_normalize(img_low)

            plot_neutral = self.n_masks_neu == 1

            with np.errstate(divide="ignore", invalid="ignore"):
                self.log_predictions(
                    self.visualize_masks(img_high, masks_high),
                    self.visualize_masks(img_low, masks_low),
                    [
                        plot_masks_dist(mask, plot_neutral=plot_neutral)
                        for mask in masks_high
                    ],
                    [
                        plot_masks_dist(mask, plot_neutral=plot_neutral)
                        for mask in masks_low
                    ],
                    torch.sigmoid(pop_score).cpu(),
                    meta,
                )

    def visualize_masks(self, images, masks):
        # images: [B, C, H, W]
        # masks: [B, N, H, W]
        images = images.cpu().numpy()
        masks = masks.cpu().numpy()

        processed_images = []

        for idx in range(images.shape[0]):
            image = images[idx]
            image_masks = masks[idx]

            processed_images.append(plot_masks(image, image_masks, self.n_masks_neu))

        return np.stack(processed_images, axis=0)
