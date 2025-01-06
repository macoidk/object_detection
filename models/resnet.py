import torch.nn as nn

import backbones.resnet_backbone as rb
from heads.centernet_head import Head
from losses.centernet_ttf import CenternetTTFLoss

input_height = input_width = 256


class ResnetModel(nn.Module):
    """
    To connect head with backbone
    """

    def __init__(self, class_number=20, backbone_name="resnet18"):
        super().__init__()
        self.class_number = class_number
        self.backbone = rb.create_resnet_backbone(name=backbone_name)
        self.head = Head(
            backbone_output_filters=self.backbone.filters, class_number=class_number
        )
        self.loss = CenternetTTFLoss(
            # todo (AA): is this "4" below the down_ratio parameter?
            #   shouldn't we pass it as an argument to initializer?
            #   shouldn't we pass input_height and input_width as arguments too?
            class_number,
            4,
            input_height // 4,
            input_width // 4,
        )

    def forward(self, x, gt=None):
        x = x / 0.5 - 1.0  # normalization
        out = self.backbone(x)
        pred = self.head(*out)

        if gt is None:
            return pred
        else:
            loss = self.loss(gt, pred)
            return loss
