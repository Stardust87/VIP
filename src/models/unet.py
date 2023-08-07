import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def conv1d(
    ni: int, no: int, ks: int = 1, stride: int = 1, padding: int = 0, bias: bool = False
):
    "Create and initialize a `nn.Conv1d` layer with spectral normalization."
    conv = nn.Conv1d(ni, no, ks, stride=stride, padding=padding, bias=bias)
    nn.init.kaiming_normal_(conv.weight)
    if bias:
        conv.bias.data.zero_()  # type: ignore
    return nn.utils.spectral_norm(conv)


class SimpleSelfAttention(nn.Module):
    def __init__(self, n_in: int, ks=1, sym=False, use_bias=False):
        super().__init__()
        self.conv = conv1d(n_in, n_in, ks, padding=ks // 2, bias=use_bias)
        self.gamma = torch.nn.Parameter(torch.tensor([0.0]))  # type: ignore
        self.sym = sym
        self.n_in = n_in

    def forward(self, x):
        if self.sym:  # check ks=3
            # symmetry hack by https://github.com/mgrankin
            c = self.conv.weight.view(self.n_in, self.n_in)
            c = (c + c.t()) / 2
            self.conv.weight = c.view(self.n_in, self.n_in, 1)
        size = x.size()
        x = x.view(*size[:2], -1)  # (C,N)
        # changed the order of multiplication to avoid O(N^2) complexity
        # (x*xT)*(W*x) instead of (x*(xT*(W*x)))
        convx = self.conv(x)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        xxT = torch.bmm(
            x, x.permute(0, 2, 1).contiguous()
        )  # (C,N) * (N,C) = (C,C)   => O(NC^2)
        o = torch.bmm(xxT, convx)  # (C,C) * (C,N) = (C,N)   => O(NC^2)
        o = self.gamma * o + x
        return o.view(*size).contiguous()


class ConvGELU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, attention=False):
        super().__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding),
            nn.Identity() if not attention else SimpleSelfAttention(nout),
            nn.GELU(),
        )


class ConvINGELU(nn.Sequential):
    def __init__(self, nin, nout, kernel, stride=1, padding=0, attention=False):
        super().__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.InstanceNorm2d(nout, affine=True),
            nn.Identity() if not attention else SimpleSelfAttention(nout),
            nn.GELU(),
        )


class ConvGNGELU(nn.Sequential):
    def __init__(
        self,
        nin,
        nout,
        kernel,
        stride=1,
        padding=0,
        groups=8,
        attention=False,
    ):
        super().__init__(
            nn.Conv2d(nin, nout, kernel, stride, padding, bias=False),
            nn.GroupNorm(groups, nout),
            nn.Identity() if not attention else SimpleSelfAttention(nout),
            nn.GELU(),
        )


class SCSE(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.cSE = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.GELU(),
            nn.Conv2d(in_channels // reduction, in_channels, 1),
            nn.Sigmoid(),
        )
        self.sSE = nn.Sequential(nn.Conv2d(in_channels, 1, 1), nn.Sigmoid())

    def forward(self, x):
        return x * self.cSE(x) + x * self.sSE(x)


class Unet(nn.Module):
    def __init__(
        self,
        num_blocks,
        img_size=64,
        filter_start=32,
        in_chnls=4,
        out_chnls=1,
        norm="in",
        num_attentions=6,
        mask_temperature=1.0,
    ):
        super().__init__()

        self.num_attentions = num_attentions
        self.mask_temperature = mask_temperature

        c = filter_start

        if norm == "in":
            conv_block = ConvINGELU
        elif norm == "gn":
            conv_block = ConvGNGELU
        else:
            conv_block = ConvGELU

        if num_blocks == 5:
            enc_in = [in_chnls, c, c, 2 * c, 2 * c + self.num_attentions]
            enc_out = [c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c]
        elif num_blocks == 6:
            enc_in = [in_chnls, c, c, c, 2 * c + self.num_attentions, 2 * c]
            enc_out = [c, c, c, 2 * c, 2 * c, 2 * c]
            dec_in = [4 * c, 4 * c, 4 * c, 2 * c, 2 * c, 2 * c]
            dec_out = [2 * c, 2 * c, c, c, c, c]

        self.down = []
        self.up = []
        # 3x3 kernels, stride 1, padding 1
        for block_idx, (in_channel, out_channel) in enumerate(zip(enc_in, enc_out)):
            apply_attention = block_idx == len(enc_in) - 2
            self.down.append(
                conv_block(
                    in_channel,
                    out_channel,
                    kernel=3,
                    stride=1,
                    padding=1,
                    attention=apply_attention,
                ),
            )
        for in_channel, out_channel in zip(dec_in, dec_out):
            self.up.append(
                nn.Sequential(
                    conv_block(in_channel, out_channel, 3, 1, 1), SCSE(out_channel)
                )
            )

        self.down = nn.ModuleList(self.down)
        self.up = nn.ModuleList(self.up)
        self.attn_proj = nn.Conv2d(
            self.num_attentions,
            self.num_attentions,
            kernel_size=1,
            bias=False,
        )
        self.featuremap_size = img_size // 2 ** (num_blocks - 1)

        self.mlp = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(128),
            nn.GELU(),
            nn.LazyLinear(128),
            nn.GELU(),
            nn.LazyLinear(2 * c * self.featuremap_size**2),
            nn.GELU(),
        )

        self.classifier = nn.Conv2d(c, out_chnls, 1)
        self.activation = nn.Softmax(dim=1)

    def forward(self, x, attentions=None):
        x_down = [x]
        skip = []

        # Down
        for block_idx, block in enumerate(self.down):
            act = block(x_down[-1])
            skip.append(act)

            if block_idx < len(self.down) - 1:
                act = F.interpolate(
                    act, scale_factor=0.5, mode="nearest", recompute_scale_factor=True
                )

            concat_with_attentions = (
                attentions is not None
                and self.num_attentions > 0
                and attentions.shape[-2:] == act.shape[-2:]
            )
            if concat_with_attentions:
                # attentions = self.attn_proj(attentions)
                act = torch.cat([act, attentions], dim=1)
            x_down.append(act)

        # FC
        x_up = self.mlp(x_down[-1])
        x_up = rearrange(
            x_up, "b (c h w) -> b c h w", h=self.featuremap_size, w=self.featuremap_size
        )

        # Up
        for block_idx, block in enumerate(self.up):
            features = torch.cat([x_up, skip[-1 - block_idx]], dim=1)
            x_up = block(features)

            if block_idx < len(self.up) - 1:
                x_up = F.interpolate(
                    x_up, scale_factor=2.0, mode="nearest", recompute_scale_factor=True
                )

        logits = self.classifier(x_up) / self.mask_temperature
        return self.activation(logits)
