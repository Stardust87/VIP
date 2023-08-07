import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from src.constants import DATA_PATH, DATA_SAMPLE_PATH  # noqa: F401, I900
from src.utils import load_all_categories  # noqa: I900
from src.utils.pairs import (  # noqa: I900
    are_hashtags_comparable,
    are_mentions_comparable,
    get_images_combinations,
    get_user_data,
    is_high_difference_popularity_pair,
    is_proper_time_diff,
    remove_popularity_outliers,
)

OUTPUT_PATH = DATA_PATH

categories_df = load_all_categories(OUTPUT_PATH, exclude=["other", "beauty"])
categories_df["loglikes"] = categories_df["#likes"].apply(np.log)
categories_df = categories_df.rename(
    columns={"#hashtags": "hashtags", "#mentions": "mentions"},
)


def check_constraints(
    user_high: pd.DataFrame, user_low: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Check constraints for pairs of images.

    Args:
        user_high: DataFrame with potential high popularity images.
        user_low: DataFrame with potential low popularity images.

    Returns:
        Tuple of DataFrames with images that satisfy constraints.
    """

    is_within_constraints_mask = np.stack(
        [
            is_proper_time_diff(
                user_high.timestamp,
                user_low.timestamp,
            ),
            are_hashtags_comparable(
                user_high.hashtags,
                user_low.hashtags,
            ),
            are_mentions_comparable(
                user_high.mentions,
                user_low.mentions,
            ),
            is_high_difference_popularity_pair(
                user_high.loglikes,
                user_low.loglikes,
            ),
        ],
        axis=0,
    )

    is_within_constraints_mask = np.all(is_within_constraints_mask, axis=0)

    user_high = user_high[is_within_constraints_mask]
    user_low = user_low[is_within_constraints_mask]

    return user_high.reset_index(drop=True), user_low.reset_index(drop=True)


pairs_dfs = []
pbar = tqdm(categories_df.username.unique())
for username in pbar:
    user_data = get_user_data(categories_df, username)
    user_data = remove_popularity_outliers(user_data)

    user_pairs_records = []
    user_high, user_low = get_images_combinations(user_data)

    if (user_high.loglikes < user_low.loglikes).any():
        raise ValueError("Posts are not sorted by popularity")

    user_high, user_low = check_constraints(user_high, user_low)
    user_high = user_high.rename(
        columns={col: f"{col}_high" for col in user_high.columns},
    )
    user_low = user_low.rename(
        columns={col: f"{col}_low" for col in user_low.columns},
    )

    user_pairs_df = pd.concat([user_high, user_low], axis=1)

    user_pairs_df = user_pairs_df.sample(frac=1).reset_index(drop=True)
    user_pairs_df = user_pairs_df.drop_duplicates(subset=["shortcode_high"])
    user_pairs_df = user_pairs_df.drop_duplicates(subset=["shortcode_low"])

    user_pairs_df = user_pairs_df.drop(
        columns=[
            "index_high",
            "index_low",
            "username_high",
            "username_low",
            "timestamp_high",
            "timestamp_low",
            "hashtags_high",
            "hashtags_low",
            "mentions_high",
            "mentions_low",
            "loglikes_high",
            "loglikes_low",
            "#likes_high",
            "#likes_low",
            "category_low",
        ],
    )

    user_pairs_df = user_pairs_df.rename(columns={"category_high": "category"})

    pairs_dfs.append(user_pairs_df)
    pbar.set_postfix({"collected pairs": sum(len(df) for df in pairs_dfs)})


pairs_df = pd.concat(pairs_dfs, ignore_index=True)
pairs_df.to_csv(OUTPUT_PATH / "popularity_pairs.csv", index=False)
