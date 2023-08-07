from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm


def get_user_data(categories_df: pd.DataFrame, username: str):
    """
    Extract all user data.

    Args:
        categories_df: DataFrame with all categories.
        username: Username.

    Returns:
        DataFrame with all user data.
    """

    return categories_df[categories_df["username"] == username]


def calculate_higher_popularity_probability(
    loglikes_high: pd.Series,
    loglikes_low: pd.Series,
    sigma: int = 0.3,
):
    """
    Calculate probability of image having higher popularity.

    Args:
        loglikes_high: Logarithm of likes for high popularity image.
        loglikes_low: Logarithm of likes for low popularity image.
        sigma: Constant.

    Returns:
        Probability of image high having higher popularity than image low.
    """

    normal_dist = norm()
    return normal_dist.cdf((loglikes_high - loglikes_low) / (sigma * np.sqrt(2)))


def is_high_difference_popularity_pair(
    loglikes_high: pd.Series,
    loglikes_low: pd.Series,
    threshold: int = 0.95,
    sigma: int = 0.3,
):
    """
    Check if image pair obeys popularity difference rule.

    Args:
        loglikes_high: Logarithm of likes for high popularity image.
        loglikes_low: Logarithm of likes for low popularity image.
        threshold: Threshold for considering high difference pair.
        sigma: Constant.

    Returns:
        True if pair obeys popularity difference rule, False otherwise.
    """

    prob = calculate_higher_popularity_probability(loglikes_high, loglikes_low, sigma)
    return prob > threshold


def get_image_filename_from_shortcode(shortcode: str, user_data: pd.DataFrame):
    """
    Extract filename of an image given its shortcode.

    Args:
        shortcode: Shortcode of an image.
        user_data: Dataframe with user data.

    Returns:
        Image filename.
    """

    return user_data[user_data["shortcode"] == shortcode][
        "image_filename"
    ].values.item()


def get_images_combinations(user_data: pd.DataFrame):
    """
    Return combinations of images pairs for given user obeys high-low popularity rule.

    Args:
        user_data: Dataframe with user data.

    Returns:
        Paired dataframes with high and low popularity user data.
    """

    images_combinations = list(
        combinations(
            np.arange(len(user_data)),
            2,
        ),
    )

    user_high = user_data.iloc[[comb[0] for comb in images_combinations]].reset_index()
    user_low = user_data.iloc[[comb[1] for comb in images_combinations]].reset_index()

    incorrect_order_mask = user_high["loglikes"] < user_low["loglikes"]

    user_high[incorrect_order_mask], user_low[incorrect_order_mask] = (
        user_low[incorrect_order_mask],
        user_high[incorrect_order_mask],
    )

    return user_high, user_low


def remove_popularity_outliers(user_data: pd.DataFrame, k_remove: int = 10):
    """
    Remove posts with highest and lowest popularity.

    Args:
        user_data: Dataframe with user data.
        k_remove: Number of posts to remove either for lowest and highest popularity.

    Returns:
        Dataframe with user data with removed posts.
    """

    if len(user_data) < 3 * k_remove:
        return user_data

    user_data = user_data.sort_values(by="loglikes")
    return user_data.iloc[k_remove:-k_remove]


def is_proper_time_diff(time_high: pd.Series, time_low: pd.Series, days: int = 10):
    """
    Check if considered posts have proper time difference.

    Args:
        time_high: Time of uploading for high popularity images.
        time_low: Time of uploading for low popularity images.
        days: Maximum difference of days between uploading.

    Returns:
        Boolean array with information if considered posts have proper time difference.
    """

    threshold = days * 24 * 60 * 60
    return np.abs(time_high - time_low) <= threshold


def are_hashtags_comparable(
    hashtags_high: pd.Series,
    hashtags_low: pd.Series,
    max_diff: int = 3,
):
    """
    Check if considered posts have comparable number of hashtags.

    Args:
        hashtags_high: Numbers of hashtags for high popularity images.
        hashtags_low: Numbers of hashtags for low popularity images.
        max_diff: Maximum difference between hashtags numbers.

    Returns:
        Boolean mask with comparable number of hashtags.
    """

    return np.abs(hashtags_high - hashtags_low) <= max_diff


def are_mentions_comparable(
    mentions_high: pd.Series,
    mentions_low: pd.Series,
    max_diff: int = 3,
):
    """
    Check if considered posts have comparable number of mentions.

    Args:
        mentions_high: Numbers of mentions for high popularity images.
        mentions_low: Numbers of mentions for low popularity images.
        max_diff: Maximum difference between mentions numbers.

    Returns:
        Boolean mask with comparable number of mentions.
    """

    return np.abs(mentions_high - mentions_low) <= max_diff
