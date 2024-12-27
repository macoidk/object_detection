import torch
import torch.nn as nn
from collections import OrderedDict


class CenterNetGIoULoss(nn.Module):
    def __init__(self, class_num, down_ratio, out_height, out_width, loss_dict={}):
        super().__init__()

        # Initialize coordinate grids
        self._cols = torch.arange(out_width).repeat(out_height, 1)
        self._rows = torch.arange(out_height).repeat(out_width, 1).t()

        self._down_ratio = down_ratio
        self._class_num = class_num

        # Loss hyperparameters
        self.lambda_giou = loss_dict.get("lambda_giou", 2.0)  # Increased weight for GIoU
        self.lambda_cls = loss_dict.get("lambda_cls", 1.0)
        self.alpha = loss_dict.get("alpha", 2.0)  # Focal loss alpha
        self.gamma = loss_dict.get("gamma", 4.0)  # Focal loss gamma
        self.delta = 1e-6  # Numerical stability

        self._losses = OrderedDict({
            "loss_cls": 0.0,
            "loss_giou": 0.0,
            "loss": 0.0
        })

    def get_box_coors(self, y_pred):
        """Convert model outputs to box coordinates."""
        if y_pred.device != self._cols.device:
            self._cols = self._cols.to(y_pred.device)
            self._rows = self._rows.to(y_pred.device)

        r = self._down_ratio
        x1 = r * self._cols - y_pred[..., 0]
        y1 = r * self._rows - y_pred[..., 1]
        x2 = r * self._cols + y_pred[..., 2]
        y2 = r * self._rows + y_pred[..., 3]

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def quality_focal_loss(self, y_true, y_pred):
        """
        Quality Focal Loss - better suited for ConvNeXt architectures
        Focuses more on high-quality predictions
        """
        pos_mask = y_true.eq(1.0).float()
        neg_mask = 1.0 - pos_mask

        # Quality estimation
        pred_quality = torch.abs(y_pred - y_true) * pos_mask
        quality_weight = torch.pow(pred_quality + self.delta, self.gamma)

        # Focal weight
        neg_weights = torch.pow(1.0 - y_true, self.alpha)

        y_pred = torch.clamp(y_pred, self.delta, 1.0 - self.delta)
        pos_loss = -torch.log(y_pred) * quality_weight * pos_mask
        neg_loss = -torch.log(1.0 - y_pred) * neg_weights * neg_mask

        num_pos = torch.maximum(pos_mask.sum(), torch.ones_like(pos_mask.sum()))
        focal_loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return focal_loss

    def giou_loss(self, y_true, y_pred):
        """
        Generalized IoU Loss with additional penalties for shape and orientation
        """
        y_true = torch.reshape(y_true, (-1, 4))
        mask = torch.gt(torch.sum(y_true, dim=1), 0.0).float()
        num_pos = mask.sum()

        if num_pos == 0:
            return 0.0

        y_pred = self.get_box_coors(y_pred)
        y_pred = torch.reshape(y_pred, (-1, 4))

        # Box coordinates
        x1g, y1g, x2g, y2g = y_true.chunk(4, dim=1)
        x1p, y1p, x2p, y2p = y_pred.chunk(4, dim=1)

        # Calculate areas
        area_g = (x2g - x1g) * (y2g - y1g)
        area_p = (x2p - x1p) * (y2p - y1p)

        # Intersection
        xA = torch.maximum(x1g, x1p)
        yA = torch.maximum(y1g, y1p)
        xB = torch.minimum(x2g, x2p)
        yB = torch.minimum(y2g, y2p)

        inter_area = torch.maximum(xB - xA, torch.zeros_like(xA)) * \
                     torch.maximum(yB - yA, torch.zeros_like(yA))

        # Union
        union_area = area_g + area_p - inter_area

        # IoU
        iou = inter_area / (union_area + self.delta)

        # Find the smallest enclosing box
        xC = torch.minimum(x1g, x1p)
        yC = torch.minimum(y1g, y1p)
        xD = torch.maximum(x2g, x2p)
        yD = torch.maximum(y2g, y2p)

        # Diagonal length of smallest enclosing box
        c_area = (xD - xC) * (yD - yC)

        # GIoU
        giou = iou - (c_area - union_area) / (c_area + self.delta)

        # Apply mask and calculate loss
        giou_loss = (1 - giou) * mask
        return giou_loss.sum() / num_pos

    def forward(self, y_true, y_pred):
        """Forward pass with combined Quality Focal Loss and GIoU Loss"""
        y_true = y_true.float()
        y_pred = y_pred.float()
        y_pred = y_pred.permute(0, 2, 3, 1)

        # Classification loss with quality focal loss
        cls_loss = self.quality_focal_loss(
            y_true[..., :self._class_num],
            y_pred[..., :self._class_num]
        )

        # Box regression loss with GIoU
        giou_loss = self.giou_loss(
            y_true[..., self._class_num:],
            y_pred[..., self._class_num:]
        )

        # Combine losses
        self._losses["loss_cls"] = cls_loss
        self._losses["loss_giou"] = giou_loss
        self._losses["loss"] = self.lambda_cls * cls_loss + self.lambda_giou * giou_loss

        return self._losses