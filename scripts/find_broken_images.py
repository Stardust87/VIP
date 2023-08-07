import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import DATA_PATH  # noqa: I900
from src.utils.image import read_image  # noqa: I900

pairs_df = pd.read_csv(DATA_PATH / "popularity_pairs.csv")
images_path = DATA_PATH / "Images" / "image"


def is_incorrect(image: np.ndarray) -> bool:
    """
    Check if given image has incorrect shape.

    Args:
        image: Image to check.

    Returns:
        True if image has incorrect shape, False otherwise.
    """

    return image.shape[2] != 3


incorrect_indices = []

pbar = tqdm(pairs_df.itertuples(), total=len(pairs_df))
for row in pbar:
    idx = row.Index

    try:
        img_high = read_image(images_path / row.image_filename_high)
        img_low = read_image(images_path / row.image_filename_low)
    except KeyboardInterrupt:
        break
    except:  # noqa: E722
        incorrect_indices.append(idx)
        continue

    if is_incorrect(img_high) or is_incorrect(img_low):
        incorrect_indices.append(idx)

    pbar.set_postfix({"incorrect": len(incorrect_indices)})

fixed_pairs_df = pairs_df.drop(incorrect_indices).reset_index(drop=True)
fixed_pairs_df.to_csv(DATA_PATH / "popularity_pairs_fixed.csv", index=False)
