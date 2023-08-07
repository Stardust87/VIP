from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch
from einops import rearrange

from src.constants import MEANS, STDS  # noqa: I900


def read_image(path: Union[Path, str]) -> np.ndarray:
    """
    Read image from path and convert it to RGB format.

    Args:
        path: Path to image.

    Returns:
        Image in RGB format.
    """

    if isinstance(path, Path):
        path = str(path)

    image = cv2.imread(path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def read_video(path: Union[Path, str]) -> np.ndarray:
    """
    Read video from path and convert it to RGB format.

    Args:
        path: Path to video.

    Returns:
        Video in RGB format.
    """

    if isinstance(path, Path):
        path = str(path)

    video = cv2.VideoCapture(path)
    frames = []

    while video.isOpened():
        success, frame = video.read()
        if not success:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)

    return np.stack(frames)


def inverse_normalize(images: torch.Tensor, mean=MEANS, std=STDS):
    """
    Inverse normalize image.

    Args:
        image: Image to inverse normalize.
        mean: Mean to inverse normalize image.
        std: Standard deviation to inverse normalize image.

    Returns:
        Inverse normalized image.
    """

    mean = torch.tensor(mean, device=images.device)
    std = torch.tensor(std, device=images.device)

    mean = rearrange(mean, "c -> 1 c 1 1")
    std = rearrange(std, "c -> 1 c 1 1")

    return images * std + mean


def resize_image(image: np.ndarray, size: int, keep_aspect_ratio: bool = True):
    """
    Resize image to given size.

    Args:
        image: Image to resize.
        size: Size to resize image.
        keep_aspect_ratio: If True, keep aspect ratio of image.

    Returns:
        Resized image.
    """

    if keep_aspect_ratio:
        h, w = image.shape[:2]
        if h > w:
            new_h, new_w = size, int(size * w / h)
        else:
            new_h, new_w = int(size * h / w), size

        return cv2.resize(image, (new_w, new_h))

    return cv2.resize(image, (size, size))
