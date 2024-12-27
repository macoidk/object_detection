from collections import OrderedDict
import torch
import torch.nn as nn
import numpy as np


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x


class ConvNeXtLargeBackbone(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.alpha = alpha

        # ConvNeXt-Large specific dimensions
        self.filters = np.array([
            192 * self.alpha,
            384 * self.alpha,
            768 * self.alpha,
            1536 * self.alpha,
            1536 * self.alpha
        ]).astype('int')

        s = self.filters

        # Stem stage
        self.stem = nn.Sequential(
            nn.Conv2d(3, s[0], kernel_size=4, stride=4),
            nn.LayerNorm(s[0], eps=1e-6, data_format="channels_first")
        )

        # Stage 1
        self.stage1 = self._make_stage(s[0], s[0], 3)
        self.downsample1 = self._make_downsample(s[0], s[1])

        # Stage 2
        self.stage2 = self._make_stage(s[1], s[1], 3)
        self.downsample2 = self._make_downsample(s[1], s[2])

        # Stage 3
        self.stage3 = self._make_stage(s[2], s[2], 9)
        self.downsample3 = self._make_downsample(s[2], s[3])

        # Stage 4
        self.stage4 = self._make_stage(s[3], s[3], 3)
        self.downsample4 = self._make_downsample(s[3], s[4])

    def _make_stage(self, dim_in, dim_out, num_blocks):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(ConvNeXtBlock(dim_out))
        return nn.Sequential(*blocks)

    def _make_downsample(self, dim_in, dim_out):
        return nn.Sequential(
            nn.LayerNorm(dim_in, eps=1e-6, data_format="channels_first"),
            nn.Conv2d(dim_in, dim_out, kernel_size=2, stride=2),
        )

    def forward(self, x):
        # Initial feature extraction
        x = self.stem(x)  # stride 4
        out_stride_4 = x

        # Stage 1
        x = self.stage1(x)
        out_stride_8 = self.downsample1(x)

        # Stage 2
        x = self.stage2(out_stride_8)
        out_stride_16 = self.downsample2(x)

        # Stage 3
        x = self.stage3(out_stride_16)
        out_stride_32 = self.downsample3(x)

        # Stage 4
        x = self.stage4(out_stride_32)
        out_stride_64 = self.downsample4(x)

        return out_stride_4, out_stride_8, out_stride_16, out_stride_32, out_stride_64


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob # Calculate the probability of saving

        # Create a mask of the same size as the batch
        # shape[0] - the size of the batch, other dimensions = 1

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize 0 1
        output = x.div(keep_prob) * random_tensor
        return output