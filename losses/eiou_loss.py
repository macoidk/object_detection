from collections import OrderedDict

import torch
import torch.nn as nn


class CenterNetEIoULoss(nn.Module):
    def __init__(self, class_num, down_ratio, out_height, out_width, loss_dict={}):
        super().__init__()

        # Initialize coordinate grids
        self._cols = torch.arange(out_width).repeat(out_height, 1)
        self._rows = torch.arange(out_height).repeat(out_width, 1).t()

        self._down_ratio = down_ratio
        self._class_num = class_num

        """
        
        """

        # Loss hyperparameters optimized for ConvNeXt
        self.lambda_eiou = loss_dict.get("lambda_eiou", 2.5)
        self.lambda_cls = loss_dict.get("lambda_cls", 1.0)
        self.beta = loss_dict.get("beta", 1.0)  # EIoU weight parameter
        self.delta = 1e-6

        self._losses = OrderedDict({"loss_cls": 0.0, "loss_eiou": 0.0, "loss": 0.0})

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

    def varifocal_loss(self, y_true, y_pred):
        """
        Varifocal Loss - adaptive focal loss that works well with EIoU
        """
        pos_mask = y_true.eq(1.0).float()
        neg_mask = 1.0 - pos_mask

        # Quality estimation using IoU prediction
        pred_quality = y_pred * pos_mask
        quality_weight = torch.pow(pred_quality, 2)

        y_pred = torch.clamp(y_pred, self.delta, 1.0 - self.delta)
        pos_loss = -torch.log(y_pred) * quality_weight * pos_mask
        neg_loss = -torch.log(1.0 - y_pred) * torch.pow(y_pred, 2) * neg_mask

        num_pos = torch.maximum(pos_mask.sum(), torch.ones_like(pos_mask.sum()))
        focal_loss = (pos_loss.sum() + neg_loss.sum()) / num_pos
        return focal_loss

    def eiou_loss(self, y_true, y_pred):
        """
        Efficient IoU Loss - better geometric distance metric
        Includes penalties for aspect ratio and center point distance
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

        # Box centers
        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        cxp = (x1p + x2p) / 2
        cyp = (y1p + y2p) / 2

        # Box dimensions
        wg = x2g - x1g
        hg = y2g - y1g
        wp = x2p - x1p
        hp = y2p - y1p

        # Intersection
        xA = torch.maximum(x1g, x1p)
        yA = torch.maximum(y1g, y1p)
        xB = torch.minimum(x2g, x2p)
        yB = torch.minimum(y2g, y2p)

        inter_area = torch.maximum(xB - xA, torch.zeros_like(xA)) * torch.maximum(
            yB - yA, torch.zeros_like(yA)
        )

        # Areas
        area_g = wg * hg
        area_p = wp * hp
        union = area_g + area_p - inter_area

        # IoU
        iou = inter_area / (union + self.delta)

        # Distance between centers
        c_dist = ((cxg - cxp) ** 2 + (cyg - cyp) ** 2) / (
            torch.maximum(xB - xA, torch.zeros_like(xA)) ** 2
            + torch.maximum(yB - yA, torch.zeros_like(yA)) ** 2
            + self.delta
        )

        # Aspect ratio similarity
        ar_gt = torch.atan(wg / (hg + self.delta))
        ar_pred = torch.atan(wp / (hp + self.delta))
        ar_loss = 4 / (torch.pi**2) * (ar_gt - ar_pred) ** 2

        # Efficient IoU Loss
        eiou_loss = 1 - iou + self.beta * (c_dist + ar_loss)

        # Apply mask and calculate loss
        eiou_loss = eiou_loss * mask
        return eiou_loss.sum() / num_pos

    def forward(self, y_true, y_pred):
        """Forward pass combining Varifocal Loss and EIoU Loss"""
        y_true = y_true.float()
        y_pred = y_pred.float()
        y_pred = y_pred.permute(0, 2, 3, 1)

        # Classification loss with varifocal loss
        cls_loss = self.varifocal_loss(
            y_true[..., : self._class_num], y_pred[..., : self._class_num]
        )

        # Box regression loss with EIoU
        eiou_loss = self.eiou_loss(
            y_true[..., self._class_num :], y_pred[..., self._class_num :]
        )

        # Combine losses
        self._losses["loss_cls"] = cls_loss
        self._losses["loss_eiou"] = eiou_loss
        self._losses["loss"] = self.lambda_cls * cls_loss + self.lambda_eiou * eiou_loss

        return self._losses
