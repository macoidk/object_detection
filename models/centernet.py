import torch.nn as nn

from backbones import create_backbone
from heads.centernet_head import Head
from losses.centernet_ttf import CenternetTTFLoss
from utils.config import IMG_HEIGHT, IMG_WIDTH


class ModelBuilder(nn.Module):
    """
    To connect head with backbone
    """

    def __init__(
        self,
        alpha=1.0,
        class_number=20,
        backbone: str = "default",
        backbone_weights: str = None,
    ):
        super().__init__()
        self.class_number = class_number
        self.backbone = create_backbone(backbone, alpha, backbone_weights)
        self.head = Head(
            backbone_output_filters=self.backbone.filters, class_number=class_number
        )
        self.loss = CenternetTTFLoss(
            class_number,
            4,
            IMG_HEIGHT // 4,
            IMG_WIDTH // 4,
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
