import torch.nn as nn

from losses.eiou_loss import CenterNetEIoULoss
from backbones.imagenet_backbone import ImageNetBackbone
from heads.imagenet_head import ImageNetHead
from losses.centernet_ttf import CenternetTTFLoss

# Constants for input dimensions
input_height = input_width = 256


class ImageNetModel(nn.Module):

    def __init__(self, alpha=0.75, class_number=20, down_ratio=4):
        super().__init__()
        self.class_number = class_number
        self.down_ratio = down_ratio

        # Initialize backbone
        self.backbone = ImageNetBackbone(alpha)

        # Initialize head with backbone filter sizes
        self.head = ImageNetHead(
            backbone_output_filters=self.backbone.filters,
            class_number=class_number,
            down_ratio=down_ratio
        )

        self.loss = CenternetTTFLoss(

            class_number,
            4,
            input_height // 4,
            input_width // 4,
        )

    def forward(self, x, gt=None):

        x = x / 0.5 - 1.0

        backbone_features = self.backbone(x)

        predictions = self.head(*backbone_features)

        if gt is None:
            return predictions
        else:
            loss = self.loss(gt, predictions)

            return loss