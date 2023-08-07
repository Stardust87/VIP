import os
from enum import Enum
from pathlib import Path

DATA_PATH = Path("/media/szacho/Extreme/data/influencer")
CHECKPOINTS_PATH = Path("/media/szacho/Extreme/checkpoints")

LOG_DIR = Path(os.getenv("LOG_DIR", "lightning_logs"))
PROJECT_NAME = "vip"

MIN_LIKES = 50
MIN_POSTS = 100

DATA_MAC_PATH = Path("influencer_sample")
DATA_COLAB_PATH = Path("/content/influencer_sample")


MEANS = (0.485, 0.456, 0.406)
STDS = (0.229, 0.224, 0.225)

EPSILON_FP16 = 1e-5


class InfluencerCategory(str, Enum):
    """Influencer category."""

    BEAUTY = "beauty"
    FAMILY = "family"
    FASHION = "fashion"
    FITNESS = "fitness"
    FOOD = "food"
    INTERIOR = "interior"
    OTHER = "other"
    PET = "pet"
    TRAVEL = "travel"
