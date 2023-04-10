"""mobilenetv2 in pytorch



[1] Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

    MobileNetV2: Inverted Residuals and Linear Bottlenecks
    https://arxiv.org/abs/1801.04381
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearBottleNeck(nn.Module):

    def __init__(self, in_channels, out_channels, stride, t=6, class_num=100):
        super().__init__()

        self.residual = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * t, 1),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, in_channels * t, 3, stride=stride, padding=1, groups=in_channels * t),
            nn.BatchNorm2d(in_channels * t),
            nn.ReLU6(inplace=True),

            nn.Conv2d(in_channels * t, out_channels, 1),
            nn.BatchNorm2d(out_channels)
        )

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):

        residual = self.residual(x)

        if self.stride == 1 and self.in_channels == self.out_channels:
            residual += x

        return residual

class MobileNetV2(nn.Module):

    def __init__(self, class_num=100):
        super().__init__()

        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU6(inplace=True)
        )

        self.stage1 = LinearBottleNeck(32, 16, 1, 1)
        self.stage2 = self._make_stage(2, 16, 24, 2, 6)
        self.stage3 = self._make_stage(3, 24, 32, 2, 6)
        self.stage4 = self._make_stage(4, 32, 64, 2, 6)
        self.stage5 = self._make_stage(3, 64, 96, 1, 6)
        self.stage6 = self._make_stage(3, 96, 160, 1, 6)
        self.stage7 = LinearBottleNeck(160, 320, 1, 6)

        self.conv1 = nn.Sequential(
            nn.Conv2d(320, 1280, 1),
            nn.BatchNorm2d(1280),
            nn.ReLU6(inplace=True)
        )

        self.conv2 = nn.Conv2d(1280, class_num, 1)

        # # 以下网络是修改部分
        # # 跑代码的时候，可以设置断点，比较前后四种channel是否对应或者合理
        # self.conv0 = nn.Conv2d(3, 32, kernel_size=1, padding=0)  # 把第一个channel的数值：3升到32，最后再变回3（考虑到变量in_channels=3）
        # self.conv00 = nn.Conv2d(32, 3, kernel_size=1, padding=0)  # 32变回3，改进模块结束
        # self.conv1_ = nn.Conv2d(32, 8, kernel_size=1, padding=0)  # 卷积模块1的x1
        # self.conv11 = nn.Conv2d(8, 8, kernel_size=1, padding=0)  # 卷积模块1的x1通道之后，第一个卷积1*1
        # self.conv12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 卷积模块1的x1通道之后，第二个卷积3*3
        # self.conv13 = nn.Conv2d(8, 8, kernel_size=5, padding=2)  # 卷积模块1的x1通道之后，第三个卷积5*5
        # self.conv2_ = nn.Conv2d(32, 16, kernel_size=1, padding=0)  # 卷积模块1的x2
        # self.conv3_ = nn.Conv2d(32, 8, kernel_size=1, padding=0)  # 卷积模块1的x3，和self.conv1一样，只不过这样写清清晰点
        # self.conv31 = nn.Conv2d(8, 2, kernel_size=1, padding=0)  # 卷积模块1的x3通道之后，第一个卷积1*1(1/16)
        # self.conv32 = nn.Conv2d(2, 2, kernel_size=3, padding=1)  # 卷积模块1的x3通道之后，第二个卷积3*3(1/16)
        # self.conv33 = nn.Conv2d(2, 8, kernel_size=1, padding=0)  # 卷积模块1的x3通道之后，第三个卷积1*1(1/4)

    def forward(self, x):
        # # 以下网络是修改部分
        # x = self.conv0(x)
        # # x1
        # x1 = self.conv1_(x)
        # x1 = torch.add(torch.add(self.conv11(x1), self.conv12(x1)), self.conv13(x1))
        # # x2
        # x2 = self.conv2_(x)
        # # x3
        # x3 = self.conv33(self.conv32(self.conv31(self.conv3_(x))))
        # # 卷积模块1输出
        # x = self.conv00(torch.cat([torch.cat([x1, x2], 1), x3], 1))  # 如果这里出错，也可能是torch.cat拼接的问题（按行拼，还是按列拼）

        x = self.pre(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.conv1(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = self.conv2(x)
        x = x.view(x.size(0), -1)

        return x

    def _make_stage(self, repeat, in_channels, out_channels, stride, t):

        layers = []
        layers.append(LinearBottleNeck(in_channels, out_channels, stride, t))

        while repeat - 1:
            layers.append(LinearBottleNeck(out_channels, out_channels, 1, t))
            repeat -= 1

        return nn.Sequential(*layers)

def mobilenetv2():
    return MobileNetV2()