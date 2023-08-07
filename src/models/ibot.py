import torch
from torch import nn

from src.constants import DATA_COLAB_PATH, DATA_MAC_PATH, DATA_PATH  # noqa: I900
from src.models.vit import vit_small  # noqa: I900

IBOT_CHECKPOINTS_ROOT = DATA_PATH.parent.parent / "checkpoints" / "iBOT"
MAC_IBOT_CHECKPOINTS_ROOT = DATA_MAC_PATH / "checkpoints" / "iBOT"
COLAB_IBOT_CHECKPOINTS_ROOT = DATA_COLAB_PATH / "checkpoints" / "iBOT"

IBOT_CHECKPOINTS = {
    "vit_base_block_22k": IBOT_CHECKPOINTS_ROOT / "vit_base_block_22k.pth",
    "vit_base_rand_1k": IBOT_CHECKPOINTS_ROOT / "vit_base_rand_1k.pth",
    "vit_small_block_1k": IBOT_CHECKPOINTS_ROOT / "vit_small_block_1k.pth",
    "dev_vit_small_block_1k": MAC_IBOT_CHECKPOINTS_ROOT / "vit_small_block_1k.pth",
    "colab_vit_small_block_1k": COLAB_IBOT_CHECKPOINTS_ROOT / "vit_small_block_1k.pth",
}


class iBOT(nn.Module):
    def __init__(
        self,
        checkpoint: str = "vit_small_block_1k",
        interpolate: bool = False,
    ):
        super().__init__()
        self.checkpoint = IBOT_CHECKPOINTS[checkpoint]
        self.interpolate = interpolate

        self.vit = self.load_vit()

    def load_vit(self):
        state_dict = torch.load(self.checkpoint)["state_dict"]
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        vit = vit_small(patch_size=16, return_all_tokens=True)
        vit.load_state_dict(state_dict, strict=False)

        vit.eval()
        for p in vit.parameters():
            p.requires_grad = False

        return vit

    def forward(self, x):
        """
        Return attention maps for last layer.
        """
        batch_size = x.shape[0]
        w_featmap = x.shape[-2] // 16
        h_featmap = x.shape[-1] // 16

        attentions = self.vit.get_last_selfattention(x)
        n_heads = attentions.shape[1]  # number of heads

        attentions = attentions[:, :, 0, 1:].reshape(batch_size, n_heads, -1)
        attentions = attentions.reshape(batch_size, n_heads, w_featmap, h_featmap)

        if self.interpolate:
            attentions = nn.functional.interpolate(
                attentions, scale_factor=16, mode="nearest"
            )

        return attentions
