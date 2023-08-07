from pathlib import Path

import pandas as pd

from src.constants import (  # noqa: I900
    DATA_PATH,
    MIN_LIKES,
    MIN_POSTS,
    InfluencerCategory,
)


def build_filename(base: str) -> str:
    """
    Build filename.

    Args:
        base: Base part of builded filename.

    Returns:
        Filename including minimal number of likes and posts in file.

    """
    return f"{base}_L{MIN_LIKES}_P{MIN_POSTS}.csv"


def load_category(root_path: Path, category: str) -> pd.DataFrame:
    """
    Load data given category information.

    Args:
        root_path: Path to data directory.
        category: Category name.

    Returns:
        DataFrame with data for given category.
    """

    path = root_path / "SlimMetadataCategories" / build_filename(category)
    df = pd.read_csv(path)
    df["category"] = category
    return df


def load_all_categories(
    root_path: Path = DATA_PATH,
    exclude: list[str] = None,
) -> pd.DataFrame:
    """
    Load data for all categories.

    Args:
        root_path: Path to data directory.
        exclude: List of categories to exclude.

    Returns:
        DataFrame with data for all not excluded categories.
    """

    if exclude is None:
        exclude = []

    categories_dfs = [
        load_category(root_path, category.value)
        for category in InfluencerCategory
        if category.value not in exclude
    ]

    categories_df = pd.concat(categories_dfs).reset_index(drop=True)
    return categories_df.drop_duplicates(subset=["shortcode"])
