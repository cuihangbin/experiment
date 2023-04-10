"""mobilenet in pytorch



[1] Andrew G. Howard, Menglong Zhu, Bo Chen, Dmitry Kalenichenko, Weijun Wang, Tobias Weyand, Marco Andreetto, Hartwig Adam

    MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications
    https://arxiv.org/abs/1704.04861
"""

import torch
import torch.nn as nn


class DepthSeperabelConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(
                input_channels,
                input_channels,
                kernel_size,
                groups=input_channels,
                **kwargs),
            nn.BatchNorm2d(input_channels),
            nn.ReLU(inplace=True)
        )

        self.pointwise = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x


class BasicConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.conv = nn.Conv2d(
            input_channels, output_channels, kernel_size, **kwargs)
        self.bn = nn.BatchNorm2d(output_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)

        return x


class MobileNet(nn.Module):

    """
    Args:
        width multipler: The role of the width multiplier α is to thin
                         a network uniformly at each layer. For a given
                         layer and width multiplier α, the number of
                         input channels M becomes αM and the number of
                         output channels N becomes αN.
    """

    def __init__(self, width_multiplier=1, class_num=100):
       super().__init__()

       alpha = width_multiplier
       self.stem = nn.Sequential(
           BasicConv2d(3, int(32 * alpha), 3, padding=1, bias=False),
           DepthSeperabelConv2d(
               int(32 * alpha),
               int(64 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv1 = nn.Sequential(
           DepthSeperabelConv2d(
               int(64 * alpha),
               int(128 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(128 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv2 = nn.Sequential(
           DepthSeperabelConv2d(
               int(128 * alpha),
               int(256 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(256 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv3 = nn.Sequential(
           DepthSeperabelConv2d(
               int(256 * alpha),
               int(512 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),

           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(512 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       #downsample
       self.conv4 = nn.Sequential(
           DepthSeperabelConv2d(
               int(512 * alpha),
               int(1024 * alpha),
               3,
               stride=2,
               padding=1,
               bias=False
           ),
           DepthSeperabelConv2d(
               int(1024 * alpha),
               int(1024 * alpha),
               3,
               padding=1,
               bias=False
           )
       )

       self.fc = nn.Linear(int(1024 * alpha), class_num)
       self.avg = nn.AdaptiveAvgPool2d(1)

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

        x = self.stem(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        x = self.avg(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def mobilenet(alpha=1, class_num=100):
    return MobileNet(alpha, class_num)

