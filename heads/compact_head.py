from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F


class CompactHead(nn.Module):
    def __init__(self, backbone_output_filters, class_number=20, down_ratio=4):
        super().__init__()
        self.down_ratio = down_ratio
        self.connection_num = 4  # Зменшено для балансу ефективність/точність
        self.class_number = class_number
        self.backbone_output_filters = backbone_output_filters

        # Зменшені фільтри для мобільності
        self.filters = [128, 64, 32]
        head_filters = [self.backbone_output_filters[-1]] + self.filters

        # Основні блоки обробки
        for i, filter_num in enumerate(self.filters):
            # Основний шлях обробки
            name = f"head_{i + 1}"
            setattr(
                self,
                name,
                self.mobile_conv_block(name, head_filters[i], head_filters[i + 1])
            )

            # З'єднання з backbone
            if i < self.connection_num:
                name = f"after_{-2 - i}"
                setattr(
                    self,
                    name,
                    self.mobile_conv_block(
                        name,
                        self.backbone_output_filters[-2 - i],
                        self.filters[i],
                        1
                    )
                )

                # Полегшений механізм уваги
                name = f"attention_{i}"
                setattr(
                    self,
                    name,
                    self.light_attention(self.filters[i])
                )

        # Попередня обробка виходів
        self.before_hm = self.mobile_conv_block("before_hm", self.filters[-1], self.filters[-1])
        self.before_sizes = self.mobile_conv_block("before_sizes", self.filters[-1], self.filters[-1])

        # Вихідні шари
        self.hm = nn.Sequential(
            self.mobile_conv_block("hm", self.filters[-1], self.class_number, 3, "sigmoid"),
            nn.AdaptiveAvgPool2d((64, 64))
        )
        self.sizes = nn.Sequential(
            self.mobile_conv_block("sizes", self.filters[-1], 4, 3, None),
            nn.AdaptiveAvgPool2d((64, 64))
        )

    def light_attention(self, channels):
        """Полегшена версія channel attention"""
        return nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(8, channels // 16), 1),
            nn.ReLU6(inplace=True),
            nn.Conv2d(max(8, channels // 16), channels, 1),
            nn.Sigmoid()
        )

    def mobile_conv_block(self, name, input_num, output_num, kernel_size=3, activation="relu"):
        """Оптимізований для мобільних пристроїв conv block"""
        block = OrderedDict()
        padding = 1 if kernel_size == 3 else 0

        if input_num != output_num or kernel_size == 1:
            # Звичайна згортка для 1x1 або зміни розмірності
            block[f"conv_{name}"] = nn.Conv2d(
                input_num,
                output_num,
                kernel_size=kernel_size,
                stride=1,
                padding=padding
            )
        else:
            # Depthwise separable згортка для 3x3
            block[f"conv_dw_{name}"] = nn.Conv2d(
                input_num,
                input_num,
                kernel_size=kernel_size,
                stride=1,
                padding=padding,
                groups=input_num
            )
            block[f"conv_pw_{name}"] = nn.Conv2d(
                input_num,
                output_num,
                kernel_size=1
            )

        # Оптимізована batch normalization
        block[f"bn_{name}"] = nn.BatchNorm2d(
            output_num,
            eps=1e-3,
            momentum=0.01
        )

        # Активації
        if activation == "relu":
            block[f"relu_{name}"] = nn.ReLU6(inplace=True)
        elif activation == "sigmoid":
            block[f"sigmoid_{name}"] = nn.Sigmoid()

        return nn.Sequential(block)

    def connect_with_backbone(self, *backbone_out):
        used_out = [backbone_out[-i - 2] for i in range(self.connection_num)]
        x = backbone_out[-1]

        for i in range(len(self.filters)):
            # Обробка поточного рівня
            x = getattr(self, f"head_{i + 1}")(x)
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

            # Додавання features з backbone з attention
            if i < self.connection_num:
                name = f"after_{-2 - i}"
                x_ = getattr(self, name)(used_out[i])

                # Застосування уваги
                attention = getattr(self, f"attention_{i}")
                x_ = x_ * attention(x_)

                x = torch.add(x, x_)

        return x

    def forward(self, *backbone_out):
        self.last_shared_layer = self.connect_with_backbone(*backbone_out)

        # Гілка heatmap
        x_hm = self.before_hm(self.last_shared_layer)
        hm_out = self.hm(x_hm)

        # Гілка розмірів
        x_sizes = self.before_sizes(self.last_shared_layer)
        sizes_out = self.sizes(x_sizes)

        # Об'єднання виходів
        x = torch.cat((hm_out, sizes_out), dim=1)
        return x