import shutil
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.constants import DATA_PATH, DATA_SAMPLE_PATH  # noqa: I900


def copy_dataset_files(output_path: Path, pairs_data: pd.DataFrame) -> None:
    """
    Copy dataset files to output directory.

    Args:
        output_path: Path to output directory.
        pairs_data: DataFrame with pairs data.

    Returns:
        None.
    """

    images_path = DATA_PATH / "Images" / "image"
    output_path.mkdir(parents=True, exist_ok=True)

    pairs_data.to_csv(output_path / "popularity_pairs.csv", index=False)

    output_images_path = output_path / "Images" / "image"
    output_images_path.mkdir(parents=True, exist_ok=True)

    image_filenames = list(pairs_data["image_filename_high"]) + list(
        pairs_data["image_filename_low"]
    )
    for image_filename in tqdm(image_filenames):
        shutil.copy(
            images_path / image_filename,
            output_images_path / image_filename,
        )


def make_dataset_sample(
    output_path: Path = DATA_SAMPLE_PATH,
    sample_ratio: float = 0.2,
) -> None:
    """
    Make sample of dataset with given ratio.

    Args:
        output_path: Path to output directory.
        sample_ratio: Ratio of dataset to sample.

    Returns:
        None.
    """

    pairs_data = pd.read_csv(DATA_PATH / "popularity_pairs.csv")
    if sample_ratio < 1.0:
        pairs_data = pairs_data.sample(frac=sample_ratio, random_state=42)

    copy_dataset_files(output_path, pairs_data)


if __name__ == "__main__":
    make_dataset_sample()
