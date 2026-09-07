# Copyright (C) 2026-present Naver Corporation. All rights reserved.
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Adapted for BLASt3R from DINOv2 (https://github.com/facebookresearch/dinov2),
# dinov2/layers/layer_scale.py, which in turn derives from timm
# (https://github.com/huggingface/pytorch-image-models),
# timm/models/vision_transformer.py. Both are licensed under the
# Apache License 2.0.

from typing import Union

import torch
from torch import Tensor
from torch import nn


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: Union[float, Tensor] = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: Tensor) -> Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma