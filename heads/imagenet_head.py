from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ImageNetHead(nn.Module):
    def __init__(self, backbone_output_filters, class_number=20):
        super().__init__()
        self.connection_num = 4  # Increased connections for better feature utilization
        self.class_number = class_number
        self.backbone_output_filters = backbone_output_filters

        # Gradually reducing filters for smoother feature transition
        self.filters = [256, 128, 64, 32]
        head_filters = [self.backbone_output_filters[-1]] + self.filters

        # Feature processing blocks
        for i, filter_num in enumerate(self.filters):
            # Main processing path
            name = f"head_{i + 1}"
            setattr(
                self,
                name,
                self.conv_bn_relu(name, head_filters[i], head_filters[i + 1])
            )

            # Backbone connections with attention mechanism
            if i < self.connection_num:
                # Connection block
                name = f"after_{-2 - i}"
                setattr(
                    self,
                    name,
                    self.conv_bn_relu(
                        name,
                        self.backbone_output_filters[-2 - i],
                        self.filters[i],
                        1
                    )
                )

                # Channel attention
                name = f"attention_{i}"
                setattr(
                    self,
                    name,
                    self.channel_attention(self.filters[i])
                )

        # Pre-output processing
        self.before_hm = self.conv_bn_relu("before_hm", self.filters[-1], self.filters[-1])
        self.before_sizes = self.conv_bn_relu("before_sizes", self.filters[-1], self.filters[-1])

        # Output layers
        self.hm = self.conv_bn_relu("hm", self.filters[-1], self.class_number, 3, "sigmoid")
        self.sizes = self.conv_bn_relu("sizes", self.filters[-1], 4, 3, None)

    def channel_attention(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 16, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 16, channels, 1),
            nn.Sigmoid()
        )

    def conv_bn_relu(self, name, input_num, output_num, kernel_size=3, activation="relu"):
        block = OrderedDict()
        padding = 1 if kernel_size == 3 else 0

        # Convolution with improved initialization
        block[f"conv_{name}"] = nn.Conv2d(
            input_num,
            output_num,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False  # Disable bias when using BatchNorm
        )

        # Batch normalization with optimized parameters
        block[f"bn_{name}"] = nn.BatchNorm2d(
            output_num,
            eps=1e-3,
            momentum=0.01
        )

        # Activation functions
        if activation == "relu":
            block[f"relu_{name}"] = nn.ReLU(inplace=True)
        elif activation == "sigmoid":
            block[f"sigmoid_{name}"] = nn.Sigmoid()

        return nn.Sequential(block)

    def connect_with_backbone(self, *backbone_out):
        used_out = [backbone_out[-i - 2] for i in range(self.connection_num)]
        x = backbone_out[-1]

        for i in range(len(self.filters)):
            # Process current feature level
            x = getattr(self, f"head_{i + 1}")(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

            # Add attention-weighted features from backbone
            if i < self.connection_num:
                name = f"after_{-2 - i}"
                x_ = getattr(self, name)(used_out[i])

                # Apply channel attention
                attention = getattr(self, f"attention_{i}")
                x_ = x_ * attention(x_)

                x = torch.add(x, x_)

        return x

    def forward(self, *backbone_out):
        self.last_shared_layer = self.connect_with_backbone(*backbone_out)

        # Heatmap branch
        x_hm = self.before_hm(self.last_shared_layer)
        hm_out = self.hm(x_hm)

        # Size branch
        x_sizes = self.before_sizes(self.last_shared_layer)
        sizes_out = self.sizes(x_sizes)

        # Combine outputs
        x = torch.cat((hm_out, sizes_out), dim=1)
        return x