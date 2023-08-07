import ast

import pandas as pd
from tqdm import tqdm

from src.constants import (  # noqa: I900
    DATA_PATH,
    MIN_LIKES,
    MIN_POSTS,
    InfluencerCategory,
)
from src.utils import build_filename  # noqa: I900


def filter_min_likes(metadata: pd.DataFrame, min_likes: int = 50) -> pd.DataFrame:
    """
    Filter metadata to contain only images with at least `min_likes` likes.

    Args:
        metadata: DataFrame with metadata.
        min_likes: Minimum number of likes.

    Returns:
        Filtered DataFrame.
    """

    return metadata[metadata["#likes"] >= min_likes]


def filter_min_posts(metadata: pd.DataFrame, min_posts: int = 100) -> pd.DataFrame:
    """
    Filter metadata to contain only images with at least `min_posts` posts.

    Args:
        metadata: DataFrame with metadata.
        min_posts: Minimum number of posts.

    Returns:
        Filtered DataFrame.
    """

    return metadata.groupby("username").filter(lambda posts: len(posts) >= min_posts)


def count_values_in_str_list(text_list: str) -> int:
    """
    Count number of values in string list.

    Args:
        text_list: String representation of list.

    Returns:
        Number of values in list.
    """

    return len(ast.literal_eval(text_list))


def count_caption_values(metadata: pd.DataFrame, col: str) -> pd.DataFrame:
    """
    Count number of captions.

    Args:
        metadata: DataFrame with metadata.
        col: Column name.

    Returns:
        DataFrame with column containg calculated caption count.
    """

    metadata[f"#{col}"] = metadata[col].apply(count_values_in_str_list)
    return metadata.drop(columns=[col])


categories = [category.value for category in InfluencerCategory]
included_users = []


for category in tqdm(categories):
    metadata_path = DATA_PATH / "MetadataCategories" / f"{category}.csv"
    metadata_df = pd.read_csv(metadata_path)

    metadata_df = filter_min_likes(metadata_df, MIN_LIKES)
    metadata_df = filter_min_posts(metadata_df, MIN_POSTS)

    metadata_df = count_caption_values(metadata_df, "hashtags")
    metadata_df = count_caption_values(metadata_df, "mentions")

    metadata_df = metadata_df.reset_index(drop=True)

    metadata_df.to_csv(
        DATA_PATH / "SlimMetadataCategories" / build_filename(category),
        index=False,
    )

    included_users.extend(list(metadata_df["username"].unique()))

influencers_df = pd.read_csv(DATA_PATH / "influencers.csv")
influencers_df = influencers_df[influencers_df["username"].isin(included_users)]
influencers_df.to_csv(
    DATA_PATH / build_filename("influencers"),
    index=False,
)
