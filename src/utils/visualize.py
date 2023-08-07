import io

import cv2
import numpy as np
import plotly.express as px
import pydensecrf.densecrf as dcrf
import seaborn as sns
import torch
from matplotlib import pyplot as plt
from PIL import Image, ImageColor
from pydensecrf.utils import create_pairwise_bilateral, unary_from_softmax

from src.utils.pairs import calculate_higher_popularity_probability  # noqa: I900

COLORS = px.colors.qualitative.Plotly


def plot_high_difference_popularity_pair(
    img_high: np.ndarray,
    img_low: np.ndarray,
    loglikes_high: float,
    loglikes_low: float,
) -> None:
    """
    Plot a pair of images which obey popularity difference rule.

     Args:
         img_high: Image with higher popularity.
         img_low: Image with lower popularity.
         loglikes_high: Log-likelihoods of the higher popularity image.
         loglikes_low: Log-likelihoods of the lower popularity image.

     Returns:
         None.
    """

    title_high = f"{loglikes_high=:.0f}"
    title_low = f"{loglikes_low=:.0f}"

    fig, ax = plt.subplots(1, 2)
    ax[0].imshow(img_high)
    ax[0].set_title(title_high)
    ax[1].imshow(img_low)
    ax[1].set_title(title_low)

    prob = calculate_higher_popularity_probability(loglikes_high, loglikes_low)
    plt.suptitle(f"Probability of higher popularity: {prob:.3f}")
    plt.show()


def draw_masks(mask_labels: np.ndarray, n_masks_neutral: int = 1) -> np.ndarray:
    """
    Draw masks on a black background.

     Args:
        mask_labels: Argmax of predicted masks for every pixel.
        n_masks_neutral: Number of neutral masks.

     Returns:
        Image of dominating masks.
    """

    mask_labels = mask_labels.squeeze()
    mask_image = np.zeros((*mask_labels.shape[:2], 3), dtype=np.uint8)

    for label in range(mask_labels.max() + 1 - n_masks_neutral):
        color = ImageColor.getrgb(COLORS[(label + 1) % len(COLORS)])
        mask_image[mask_labels == label, :] = color

    return mask_image


def overlay_masks(
    image: np.ndarray,
    mask_labels: np.ndarray,
    alpha: float = 0.5,
    n_masks_neutral: int = 1,
) -> np.ndarray:
    """
    Overlay masks on an image.

     Args:
        image: Image to overlay masks on.
        mask_labels: Argmax of predicted masks for every pixel.
        alpha: Alpha value for overlay.
        n_masks_neutral: Number of neutral masks.

     Returns:
        Image with masks overlayed.
    """

    mask_image = draw_masks(mask_labels, n_masks_neutral=n_masks_neutral)
    return cv2.addWeighted(image, alpha, mask_image, 1 - alpha, 0)


def run_crf(image: np.ndarray, masks: np.ndarray) -> np.ndarray:
    """
    Run CRF on given image and masks.

     Args:
        image: Base image.
        masks: Predicted masks.

     Returns:
        Mask labels after CRF.
    """

    H, W = image.shape[1:]

    unary = unary_from_softmax(masks)
    unary = np.ascontiguousarray(unary)

    pairwise_energy = create_pairwise_bilateral(
        sdims=(10, 10), schan=(0.01,), img=image, chdim=0
    )

    d = dcrf.DenseCRF2D(H, W, len(masks))  # width, height, nlabels
    d.setUnaryEnergy(unary)
    d.addPairwiseEnergy(pairwise_energy, compat=10)

    Q_unary = d.inference(10)
    map_soln = np.argmax(Q_unary, axis=0).reshape((H, W))

    return map_soln


def plot_masks(image: np.ndarray, masks: np.ndarray, n_masks_neutral: int = 1):
    """
    Create a plot of image: original, overlayed with masks, CRF masks, predicted masks.

     Args:
        image: Base image.
        masks: Predicted masks.
        n_masks_neutral: Number of neutral masks.

     Returns:
        Plot of image: original, overlayed with masks, CRF masks, predicted masks.
    """

    image_masks_labels = np.argmax(masks, axis=0)
    crf_masks_labels = run_crf(image, masks)

    image = (image * 255).astype(np.uint8).transpose(1, 2, 0)

    overlayed_mask = overlay_masks(
        image, crf_masks_labels, n_masks_neutral=n_masks_neutral
    )
    crf_drawn_mask = draw_masks(crf_masks_labels, n_masks_neutral=n_masks_neutral)
    drawn_mask = draw_masks(image_masks_labels, n_masks_neutral=n_masks_neutral)

    return np.concatenate([image, overlayed_mask, crf_drawn_mask, drawn_mask], axis=1)


def plot_masks_dist(masks: torch.Tensor, plot_neutral: bool = False):
    """
    Create a plot of masks distribution.

     Args:
        masks: Predicted masks.

     Returns:
        Plot of masks distribution.
    """
    masks = masks.detach().cpu().numpy()

    fig, ax = plt.subplots(figsize=(5, 2))
    sns.set_theme(style="dark")
    for mask_dim in range(masks.shape[0]):
        if (mask_dim == masks.shape[0] - 1) and plot_neutral:
            color = "gray"
        else:
            color = COLORS[mask_dim + 1]

        try:
            sns.histplot(
                masks[mask_dim].flatten(),
                color=color,
                kde=True,
                stat="density",
                element="step",
                alpha=0.4,
                linewidth=0,
                ax=ax,
            )
        except:  # noqa: E722
            print("Error in plot_masks_dist")

    plt.ylabel("")
    plt.xticks(np.arange(0, 1.1, step=0.1), fontsize=7, color="gray")
    plt.yticks(fontsize=7, color="gray")
    ax.spines[["bottom", "left", "top", "right"]].set_visible(False)
    plt.tight_layout()

    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png", bbox_inches="tight", transparent=True, dpi=300)
    plt.close()

    return Image.open(img_buf)
