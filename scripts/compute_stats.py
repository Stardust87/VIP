import torch
from tqdm import tqdm

from src.data.influencer import InfluencerPairDataModule  # noqa: I900


def compute_means_and_stds(
    loader: InfluencerPairDataModule, n_channels: int = 3
) -> tuple[list, list]:
    """
    Compute means and standard deviations of images.

    Args:
        loader: InfluencerPairDataModule.
        n_channels: Number of channels in images.

    Returns:
        Means and standard deviations of images.

    """
    psum = torch.zeros(n_channels)
    psum_sq = torch.zeros(n_channels)
    count = 0

    for batch in tqdm(loader):
        images_high, images_low, meta = batch
        images = torch.cat((images_high, images_low), dim=0)

        if images.max() > 1:
            raise ValueError("Images should be normalized to [0, 1]")

        for image in images:
            psum += image.sum(dim=(1, 2))
            psum_sq += (image**2).sum(dim=(1, 2))

            count += image.shape[1] * image.shape[2]

    total_mean = psum / count
    total_std = torch.sqrt((psum_sq / count) - (total_mean**2))

    return total_mean.numpy().tolist(), total_std.numpy().tolist()


if __name__ == "__main__":
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2

    from configs import FAMEConfig as cfg  # noqa: I900
    from src.constants import DATA_MAC_PATH  # noqa: I900

    MAC = True

    if MAC:
        cfg.data.path = DATA_MAC_PATH
        cfg.data.num_workers = 0

    cfg.data.sample_ratio = 1 / 8
    datamodule = InfluencerPairDataModule(cfg.data)
    datamodule.train_transform = A.Compose(
        [
            A.LongestMaxSize(cfg.data.image_transform_size),
            A.PadIfNeeded(cfg.data.image_transform_size, cfg.data.image_transform_size),
            A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
            ToTensorV2(),
        ]
    )
    datamodule.setup()

    means, stds = compute_means_and_stds(datamodule.train_dataloader(), n_channels=3)

    print(f"{means=}")
    print(f"{stds=}")
