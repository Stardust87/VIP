import random
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset, random_split

from configs.base import DataConfig  # noqa: I900
from src.utils import read_image  # noqa: I900
from src.utils.helpers import get_mask_transforms, get_transforms  # noqa: I900


class InfluencerPairDataset(Dataset):
    def __init__(
        self,
        pairs_data: pd.DataFrame,
        path: Path,
        transform=None,
        mask_transform=None,
        is_train=False,
    ):
        self.path = path
        self.pairs = pairs_data
        self.images_path = self.path / "Images" / "image"
        self.transform = transform
        self.mask_transform = mask_transform
        self.is_train = is_train

        if self.transform is None:
            raise ValueError("Transforms should be defined")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, ind):
        img_high_path, img_low_path = self.pairs.loc[
            ind,
            ["image_filename_high", "image_filename_low"],
        ].values

        meta = {
            "shortcode_high": self.pairs.loc[ind, "shortcode_high"],
            "shortcode_low": self.pairs.loc[ind, "shortcode_low"],
            "category": self.pairs.loc[ind, "category"],
            "image_filename_high": img_high_path,
            "image_filename_low": img_low_path,
        }

        img_high = read_image(self.images_path / img_high_path)
        img_low = read_image(self.images_path / img_low_path)

        transformed_high = self.transform(image=img_high)
        transformed_low = self.transform(image=img_low)

        transformed_high, transformed_low = (
            transformed_high["image"],
            transformed_low["image"],
        )

        meta |= {
            "image_high": transformed_high,
            "image_low": transformed_low,
        }

        if self.mask_transform is not None:
            mask_high = self.mask_transform(image=img_high)["image"]
            mask_low = self.mask_transform(image=img_low)["image"]

            meta |= {
                "mask_high": mask_high,
                "mask_low": mask_low,
            }

        return meta


class InfluencerPairDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.train_transform, self.test_transform = get_transforms(
            transform_size=cfg.image_transform_size, target_size=cfg.image_target_size
        )
        self.mask_transform = None
        if cfg.return_masked:
            self.mask_transform, _ = get_mask_transforms(
                transform_size=cfg.image_transform_size,
                target_size=cfg.image_target_size,
            )

        self.cfg = cfg
        self.path = cfg.path
        self.pairs_data = pd.read_csv(self.path / "popularity_pairs.csv")
        if cfg.sample_ratio < 1.0:
            self.pairs_data = self.pairs_data.sample(
                frac=cfg.sample_ratio, random_state=42
            )

    def setup(self, stage=None):
        val_size = 1 - self.cfg.train_size - self.cfg.test_size
        train, val, test = random_split(
            self.pairs_data,
            [self.cfg.train_size, val_size, self.cfg.test_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_indices = train.indices
        if self.cfg.train_ratio < 1.0:
            new_train_size = int(len(train.indices) * self.cfg.train_ratio)
            train_indices = random.sample(train_indices, new_train_size)

        self.train_ds = InfluencerPairDataset(
            self.pairs_data.iloc[train_indices].reset_index(drop=True),
            transform=self.train_transform,
            mask_transform=self.mask_transform,
            path=self.path,
            is_train=True,
        )

        self.val_ds = InfluencerPairDataset(
            self.pairs_data.iloc[val.indices].reset_index(drop=True),
            transform=self.test_transform,
            path=self.path,
        )

        self.test_ds = InfluencerPairDataset(
            self.pairs_data.iloc[test.indices].reset_index(drop=True),
            transform=self.test_transform,
            path=self.path,
        )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.cfg.batch_size,
            pin_memory=self.cfg.pin_memory,
            num_workers=self.cfg.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.cfg.batch_size,
            pin_memory=self.cfg.pin_memory,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.cfg.batch_size,
            pin_memory=self.cfg.pin_memory,
            num_workers=self.cfg.num_workers,
            shuffle=False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()
