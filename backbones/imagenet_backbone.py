from collections import OrderedDict

import numpy as np
import torch.nn as nn


class ImageNetBackbone(nn.Module):
    def __init__(self, alpha=1.0):
        super().__init__()
        self.block_num = 1
        self.alpha = alpha

        # Define base filters with alpha scaling
        self.filters = np.array(
            [
                96 * self.alpha,  # Increased initial filters
                192 * self.alpha,
                384 * self.alpha,
                768 * self.alpha,
                1024 * self.alpha,  # Deeper feature representation
            ]
        ).astype("int")

        s = self.filters
        # Initial conv with larger kernel
        self.layer1 = self.conv_bn_relu(3, s[0], False, kernel_size=7, padding=3)
        self.layer2 = self.conv_bn_relu(s[0], s[0], True)  # stride 2

        # Mid-level features
        self.layer3 = self.conv_bn_relu(s[0], s[1], False)
        self.layer4 = self.conv_bn_relu(s[1], s[1], True)  # stride 4

        # Higher-level features with residual connections
        self.layer5 = self.conv_bn_relu(s[1], s[2], False)
        self.layer6 = self.conv_bn_relu(s[2], s[2], False)
        self.layer7 = self.conv_bn_relu(s[2], s[2], True)  # stride 8

        # Deep features
        self.layer8 = self.conv_bn_relu(s[2], s[3], False)
        self.layer9 = self.conv_bn_relu(s[3], s[3], False)
        self.layer10 = self.conv_bn_relu(s[3], s[3], True)  # stride 16

        # Final layers
        self.layer11 = self.conv_bn_relu(s[3], s[4], False)
        self.layer12 = self.conv_bn_relu(s[4], s[4], False)
        self.layer13 = self.conv_bn_relu(s[4], s[4], True)  # stride 32

    def conv_bn_relu(
        self, input_num, output_num, max_pool=False, kernel_size=3, padding=1
    ):
        block = OrderedDict()
        block[f"conv_{self.block_num}"] = nn.Conv2d(
            input_num, output_num, kernel_size=kernel_size, stride=1, padding=padding
        )
        block[f"bn_{self.block_num}"] = nn.BatchNorm2d(
            output_num, eps=1e-3, momentum=0.01
        )
        block[f"relu_{self.block_num}"] = nn.ReLU()

        if max_pool:
            block[f"pool_{self.block_num}"] = nn.MaxPool2d(kernel_size=2, stride=2)

        self.block_num += 1
        return nn.Sequential(block)

    def forward(self, x):
        # Extract features at multiple scales
        out = self.layer1(x)
        out_stride_2 = self.layer2(out)
        out = self.layer3(out_stride_2)
        out_stride_4 = self.layer4(out)
        out = self.layer5(out_stride_4)
        out = self.layer6(out)
        out_stride_8 = self.layer7(out)
        out = self.layer8(out_stride_8)
        out = self.layer9(out)
        out_stride_16 = self.layer10(out)
        out = self.layer11(out_stride_16)
        out = self.layer12(out)
        out_stride_32 = self.layer13(out)

        return out_stride_2, out_stride_4, out_stride_8, out_stride_16, out_stride_32
