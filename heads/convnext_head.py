from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtHead(nn.Module):
    def __init__(self, backbone_output_filters, class_number=20):
        super().__init__()
        self.connection_num = 4
        self.class_number = class_number
        self.backbone_output_filters = backbone_output_filters

        # Larger filter sizes for better feature extraction
        self.filters = [512, 256, 128, 64]
        head_filters = [self.backbone_output_filters[-1]] + self.filters

        # Create main processing blocks
        for i, filter_num in enumerate(self.filters):
            # Main feature processing path
            name = f"head_{i + 1}"
            setattr(
                self,
                name,
                self.conv_next_block(head_filters[i], head_filters[i + 1])
            )

            # Backbone connections
            if i < self.connection_num:
                name = f"after_{-2 - i}"
                setattr(
                    self,
                    name,
                    self.conv_next_block(
                        self.backbone_output_filters[-2 - i],
                        self.filters[i]
                    )
                )

                # Feature refinement
                name = f"refine_{i}"
                setattr(
                    self,
                    name,
                    self.feature_refinement(self.filters[i])
                )

        # Output processing blocks
        self.before_hm = self.conv_next_block(self.filters[-1], self.filters[-1])
        self.before_sizes = self.conv_next_block(self.filters[-1], self.filters[-1])

        # Final output layers
        self.hm = self.conv_bn_relu("hm", self.filters[-1], self.class_number, 3, "sigmoid")
        self.sizes = self.conv_bn_relu("sizes", self.filters[-1], 4, 3, None)

    def conv_next_block(self, dim_in, dim_out):
        return nn.Sequential(
            # Depthwise conv
            nn.Conv2d(dim_in, dim_in, kernel_size=7, padding=3, groups=dim_in),
            nn.LayerNorm(dim_in, eps=1e-6),
            # Pointwise convs
            nn.Conv2d(dim_in, dim_out, 1),
            nn.GELU(),
            nn.BatchNorm2d(dim_out)
        )

    def feature_refinement(self, channels):
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 1),
            nn.Sigmoid()
        )

    def conv_bn_relu(self, name, input_num, output_num, kernel_size=3, activation="relu"):
        block = OrderedDict()
        padding = 1 if kernel_size == 3 else 0

        block[f"conv_{name}"] = nn.Conv2d(
            input_num,
            output_num,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=False
        )

        block[f"bn_{name}"] = nn.BatchNorm2d(
            output_num,
            eps=1e-3,
            momentum=0.01
        )

        if activation == "relu":
            block[f"relu_{name}"] = nn.GELU()  # Using GELU for consistency with ConvNeXt
        elif activation == "sigmoid":
            block[f"sigmoid_{name}"] = nn.Sigmoid()

        return nn.Sequential(block)

    def connect_with_backbone(self, *backbone_out):
        used_out = [backbone_out[-i - 2] for i in range(self.connection_num)]
        x = backbone_out[-1]

        for i in range(len(self.filters)):
            # Process features
            x = getattr(self, f"head_{i + 1}")(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)

            # Connect with backbone features
            if i < self.connection_num:
                name = f"after_{-2 - i}"
                x_ = getattr(self, name)(used_out[i])

                # Apply feature refinement
                refine = getattr(self, f"refine_{i}")
                x_ = x_ * refine(x_)

                x = torch.add(x, x_)

        return x

    def forward(self, *backbone_out):
        self.last_shared_layer = self.connect_with_backbone(*backbone_out)

        # Heatmap branch with ConvNeXt-style processing
        x_hm = self.before_hm(self.last_shared_layer)
        hm_out = self.hm(x_hm)

        # Size branch with ConvNeXt-style processing
        x_sizes = self.before_sizes(self.last_shared_layer)
        sizes_out = self.sizes(x_sizes)

        # Combine outputs
        x = torch.cat((hm_out, sizes_out), dim=1)
        return x