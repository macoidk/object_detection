from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, backbone_output_filters, class_number=20):
        super().__init__()
        self.connection_num = 3
        self.class_number = class_number
        self.backbone_output_filters = backbone_output_filters
        self.filters = [128, 64, 32]
        head_filters = [self.backbone_output_filters[-1]] + self.filters

        for i, filter_num in enumerate(self.filters):
            name = f"head_{i+1}"
            setattr(
                self,
                name,
                self.conv_bn_relu(name, head_filters[i], head_filters[i + 1]),
            )
            # create connection with backbone
            if i < self.connection_num:
                name = f"after_{-2-i}"
                setattr(
                    self,
                    name,
                    self.conv_bn_relu(
                        name, self.backbone_output_filters[-2 - i], self.filters[i], 1
                    ),
                )

        self.before_hm = self.conv_bn_relu(
            "before_hm", self.filters[-1], self.filters[-1]
        )
        self.before_sizes = self.conv_bn_relu(
            "before_sizes", self.filters[-1], self.filters[-1]
        )

        self.hm = self.conv_bn_relu(
            "hm", self.filters[-1], self.class_number, 3, "sigmoid"
        )
        self.sizes = self.conv_bn_relu("hm", self.filters[-1], 4, 3, None)

    def conv_bn_relu(
        self, name, input_num, output_num, kernel_size=3, activation="relu"
    ):
        block = OrderedDict()
        padding = 1 if kernel_size == 3 else 0
        block["conv_" + name] = nn.Conv2d(
            input_num, output_num, kernel_size=kernel_size, stride=1, padding=padding
        )
        block["bn_" + name] = nn.BatchNorm2d(output_num, eps=1e-3, momentum=0.01)
        if activation == "relu":
            block["relu_" + name] = nn.ReLU()
        elif activation == "sigmoid":
            block["sigmoid_" + name] = nn.Sigmoid()
        return nn.Sequential(block)

    def connect_with_backbone(self, *backbone_out):
        used_out = [backbone_out[-i - 2] for i in range(self.connection_num)]
        x = backbone_out[-1]
        for i in range(len(self.filters)):
            x = getattr(self, "head_{}".format(i + 1))(x)
            x = F.interpolate(x, scale_factor=2, mode="nearest")
            if i < self.connection_num:
                name = f"after_{-2-i}"
                x_ = getattr(self, name)(used_out[i])
                x = torch.add(x, x_)
        return x

    def forward(self, *backbone_out):
        self.last_shared_layer = self.connect_with_backbone(self, *backbone_out)
        x = self.before_hm(self.last_shared_layer)
        hm_out = self.hm(x)

        x = self.before_sizes(self.last_shared_layer)
        sizes_out = self.sizes(x)

        x = torch.cat((hm_out, sizes_out), dim=1)
        return x
