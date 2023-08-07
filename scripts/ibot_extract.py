from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.constants import DATA_PATH  # noqa: I900
from src.data import InfluencerPairDataset  # noqa: I900
from src.models.ibot import iBOT  # noqa: I900
from src.utils.helpers import get_ibot_transforms  # noqa: I900

OUTPUT_PATH = DATA_PATH / "Attentions" / "attention"


def save_attentions(attentions: torch.Tensor, filenames: list[str]) -> None:
    for attention, filename in zip(attentions, filenames):
        attention = attention.cpu().detach().clone()

        filename = Path(filename).stem + ".pt"
        torch.save(attention, OUTPUT_PATH / filename)


influencer_ds = InfluencerPairDataset(
    pd.read_csv(DATA_PATH / "popularity_pairs.csv"),
    DATA_PATH,
    transform=get_ibot_transforms(),
)

influencer_dl = DataLoader(
    influencer_ds,
    batch_size=64,
    shuffle=False,
    num_workers=16,
    pin_memory=True,
)

ibot = iBOT(interpolate=False).cuda()

for batch in tqdm(influencer_dl):
    images_high, images_low, meta = batch
    images_high = images_high.cuda()
    images_low = images_low.cuda()

    with torch.no_grad():
        attentions_high = ibot(images_high)
        attentions_low = ibot(images_low)

    image_high_filenames = meta["image_filename_high"]
    image_low_filenames = meta["image_filename_low"]

    save_attentions(attentions_high, image_high_filenames)
    save_attentions(attentions_low, image_low_filenames)
