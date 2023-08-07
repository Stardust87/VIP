import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.constants import DATA_PATH  # noqa: I900


def get_hashtags(caption):
    if isinstance(caption, str):
        return re.findall("#[a-z0-9_]+", caption)

    return []


def get_mentions(caption):
    if isinstance(caption, str):
        return [
            f"@{match[1]}" for match in re.findall(r"(^|[^\w])@([\w\_\.]+)", caption)
        ]

    return []


def get_interesting_keys(metadata: dict) -> dict:
    """
    Filter metadata to get only important keys.

    Args:
        metadata: Dictionary with metadata.

    Returns:
        Dictionary with filtered metadata.
    """

    likes_count = metadata.get("edge_media_preview_like", {}).get("count", np.nan)

    hashtags, mentions = [], []
    caption_edges = metadata.get("edge_media_to_caption", {}).get("edges", [])
    if caption_edges:
        caption = caption_edges[0]["node"]["text"]
        hashtags.extend(get_hashtags(caption))
        mentions.extend(get_mentions(caption))

    return {
        "#likes": likes_count,
        "hashtags": hashtags,
        "mentions": mentions,
        "timestamp": metadata.get("taken_at_timestamp", np.nan),
        "shortcode": metadata.get("shortcode", np.nan),
    }


mapping_file = DATA_PATH / "JSON-Image_files_mapping.txt"
influencers_file = DATA_PATH / "influencers.csv"

mapping_df = pd.read_csv(
    mapping_file,
    sep="\t",
    header=0,
    names=["influencer_name", "JSON_PostMetadata_file_name", "Image_file_name"],
)
influencers_df = pd.read_csv(influencers_file)

mapping_df = mapping_df.join(
    influencers_df.set_index("username"),
    on="influencer_name",
    how="left",
)

for category, category_df in mapping_df.groupby("category"):
    exctracted_records = []

    for row in tqdm(category_df.itertuples(), total=len(category_df), desc=category):
        # metadata
        metadata_filename = row.JSON_PostMetadata_file_name
        meta_path = (
            DATA_PATH
            / "Metadata"
            / "info"
            / f"{row.influencer_name}-{metadata_filename}"
        )

        if not meta_path.exists():
            continue

        try:
            with meta_path.open() as json_file:
                metadata = json.load(json_file)
        except json.decoder.JSONDecodeError:
            continue

        # image
        images_filenames = ast.literal_eval(row.Image_file_name)
        images_count = len(images_filenames)

        image_filename = f"{row.influencer_name}-{images_filenames[0]}"

        # record
        record = {
            "username": row.influencer_name,
            "image_filename": image_filename,
            **get_interesting_keys(metadata),
        }

        image_path = Path(DATA_PATH / "Images" / "image" / record["image_filename"])

        no_likes = np.isnan(record["#likes"])
        multiple_images = images_count > 1
        image_not_exists = not image_path.exists()

        if any([no_likes, multiple_images, image_not_exists]):
            continue
        else:
            exctracted_records.append(record)

    df = pd.DataFrame(exctracted_records)
    df.to_csv(DATA_PATH / "MetadataCategories" / f"{category}.csv", index=False)
