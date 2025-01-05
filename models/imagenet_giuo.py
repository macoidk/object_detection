import torch.nn as nn

from backbones.imagenet_backbone import ImageNetBackbone
from heads.imagenet_head import ImageNetHead
from losses.giou_loss import CenterNetGIoULoss

# Константи для розмірів вхідного зображення
input_height = input_width = 256


class ImageNetModel(nn.Module):

    def __init__(self, alpha=1.0, class_number=20, down_ratio=4):
        super().__init__()
        self.class_number = class_number
        self.down_ratio = down_ratio

        self.backbone = ImageNetBackbone(alpha)

        self.head = ImageNetHead(
            backbone_output_filters=self.backbone.filters, class_number=class_number
        )

        self.loss = CenterNetGIoULoss(
            class_num=class_number,
            down_ratio=down_ratio,
            out_height=input_height // down_ratio,
            out_width=input_width // down_ratio,
            loss_dict={
                "lambda_giou": 2.0,  # Вага для GIoU loss
                "lambda_cls": 1.0,
                "alpha": 2.0,  # Параметр focal loss
                "gamma": 4.0,  # Параметр focal loss
            },
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
