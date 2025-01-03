from collections import OrderedDict
import torch.nn as nn
import numpy as np

class CompactBackbone(nn.Module):
    def __init__(self, alpha=0.75):  # Використовуємо alpha для балансу швидкість/точність
        super().__init__()
        self.block_num = 1
        self.alpha = alpha

        # Оптимізовані фільтри для мобільних пристроїв
        self.filters = np.array([
            32 * self.alpha,
            64 * self.alpha,
            128 * self.alpha,
            256 * self.alpha,
            512 * self.alpha]).astype("int")
        s = self.filters

        self.layer1 = self.conv_dw_sep(3, s[0], stride=2)  # stride 2
        self.layer2 = self.conv_dw_sep(s[0], s[0])

        self.layer3 = self.conv_dw_sep(s[0], s[1], stride=2)  # stride 4
        self.layer4 = self.conv_dw_sep(s[1], s[1])

        self.layer5 = self.conv_dw_sep(s[1], s[2], stride=2)  # stride 8
        self.layer6 = self.conv_dw_sep(s[2], s[2])

        self.layer7 = self.conv_dw_sep(s[2], s[3], stride=2)  # stride 16
        self.layer8 = self.conv_dw_sep(s[3], s[3])

        self.layer9 = self.conv_dw_sep(s[3], s[4], stride=2)  # stride 32

    def conv_dw_sep(self, in_ch, out_ch, stride=1):
        block = OrderedDict()
        block[f'dw_{self.block_num}'] = nn.Conv2d(
            in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch)
        block[f'bn1_{self.block_num}'] = nn.BatchNorm2d(in_ch)
        block[f'relu1_{self.block_num}'] = nn.ReLU6(inplace=True)


        block[f'pw_{self.block_num}'] = nn.Conv2d(
            in_ch, out_ch, kernel_size=1, stride=1)
        block[f'bn2_{self.block_num}'] = nn.BatchNorm2d(out_ch)
        block[f'relu2_{self.block_num}'] = nn.ReLU6(inplace=True)

        self.block_num += 1
        return nn.Sequential(block)

    def forward(self, x):
        out = self.layer1(x)
        out_stride_2 = self.layer2(out)

        out = self.layer3(out_stride_2)
        out_stride_4 = self.layer4(out)

        out = self.layer5(out_stride_4)
        out_stride_8 = self.layer6(out)

        out = self.layer7(out_stride_8)
        out_stride_16 = self.layer8(out)

        out_stride_32 = self.layer9(out_stride_16)

        return out_stride_2, out_stride_4, out_stride_8, out_stride_16, out_stride_32