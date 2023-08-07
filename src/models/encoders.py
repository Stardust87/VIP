import timm
import torch
from torch import nn as nn


class TimmEncoder(nn.Module):
    def __init__(self, name: str, num_outputs: int = 1):
        super().__init__()
        self.model = timm.create_model(name, pretrained=True)

        if hasattr(self.model, "fc"):
            self.model.fc = nn.LazyLinear(num_outputs)
        else:
            self.model.head.fc = nn.LazyLinear(num_outputs)

    def forward(self, x: torch.Tensor):
        return self.model(x)
