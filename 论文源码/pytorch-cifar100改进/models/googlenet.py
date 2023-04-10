"""google net in pytorch



[1] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed,
    Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich.

    Going Deeper with Convolutions
    https://arxiv.org/abs/1409.4842v1
"""

import torch
import torch.nn as nn

class Inception(nn.Module):
    def __init__(self, input_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super().__init__()

        #1x1conv branch
        self.b1 = nn.Sequential(
            nn.Conv2d(input_channels, n1x1, kernel_size=1),
            nn.BatchNorm2d(n1x1),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 3x3conv branch
        self.b2 = nn.Sequential(
            nn.Conv2d(input_channels, n3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(n3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(n3x3),
            nn.ReLU(inplace=True)
        )

        #1x1conv -> 5x5conv branch
        #we use 2 3x3 conv filters stacked instead
        #of 1 5x5 filters to obtain the same receptive
        #field with fewer parameters
        self.b3 = nn.Sequential(
            nn.Conv2d(input_channels, n5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(n5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5, n5x5),
            nn.ReLU(inplace=True),
            nn.Conv2d(n5x5, n5x5, kernel_size=3, padding=1),
            nn.BatchNorm2d(n5x5),
            nn.ReLU(inplace=True)
        )

        #3x3pooling -> 1x1conv
        #same conv
        self.b4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(input_channels, pool_proj, kernel_size=1),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


class GoogleNet(nn.Module):

    def __init__(self, num_class=100):
        super().__init__()
        # self.prelayer = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 192, kernel_size=3, padding=1, bias=False),
        #     nn.BatchNorm2d(192),
        #     nn.ReLU(inplace=True),
        # )

        #although we only use 1 conv layer as prelayer,
        #we still use name a3, b3.......
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        ##"""In general, an Inception network is a network consisting of
        ##modules of the above type stacked upon each other, with occasional
        ##max-pooling layers with stride 2 to halve the resolution of the
        ##grid"""
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        #input feature size: 8*8*1024
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout2d(p=0.4)
        self.linear = nn.Linear(1024, num_class)

        # 以下网络是修改部分
        # 跑代码的时候，可以设置断点，比较前后四种channel是否对应或者合理
        self.conv0 = nn.Conv2d(3, 32, kernel_size=1, padding=0)  # 把第一个channel的数值：3升到32，最后再变回3（考虑到变量in_channels=3）
        self.conv00 = nn.Conv2d(32, 192, kernel_size=1, padding=0)  # 32变回3，改进模块结束
        self.conv1_ = nn.Conv2d(32, 8, kernel_size=1, padding=0)  # 卷积模块1的x1
        self.conv11 = nn.Conv2d(8, 8, kernel_size=1, padding=0)  # 卷积模块1的x1通道之后，第一个卷积1*1
        self.conv12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 卷积模块1的x1通道之后，第二个卷积3*3
        self.conv13 = nn.Conv2d(8, 8, kernel_size=5, padding=2)  # 卷积模块1的x1通道之后，第三个卷积5*5
        self.conv2_ = nn.Conv2d(32, 16, kernel_size=1, padding=0)  # 卷积模块1的x2
        self.conv3_ = nn.Conv2d(32, 8, kernel_size=1, padding=0)  # 卷积模块1的x3，和self.conv1一样，只不过这样写清清晰点
        self.conv31 = nn.Conv2d(8, 2, kernel_size=1, padding=0)  # 卷积模块1的x3通道之后，第一个卷积1*1(1/16)
        self.conv32 = nn.Conv2d(2, 2, kernel_size=3, padding=1)  # 卷积模块1的x3通道之后，第二个卷积3*3(1/16)
        self.conv33 = nn.Conv2d(2, 8, kernel_size=1, padding=0)  # 卷积模块1的x3通道之后，第三个卷积1*1(1/4)

    def forward(self, x):
        # 以下网络是修改部分
        x = self.conv0(x)
        # x1
        x1 = self.conv1_(x)
        x1 = torch.add(torch.add(self.conv11(x1), self.conv12(x1)), self.conv13(x1))
        # x2
        x2 = self.conv2_(x)
        # x3
        x3 = self.conv33(self.conv32(self.conv31(self.conv3_(x))))
        # 卷积模块1输出
        x = self.conv00(torch.cat([torch.cat([x1, x2], 1), x3], 1))  # 如果这里出错，也可能是torch.cat拼接的问题（按行拼，还是按列拼）

        # x = self.prelayer(x)
        x = self.maxpool(x)
        x = self.a3(x)
        x = self.b3(x)

        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)

        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        #"""It was found that a move from fully connected layers to
        #average pooling improved the top-1 accuracy by about 0.6%,
        #however the use of dropout remained essential even after
        #removing the fully connected layers."""
        x = self.avgpool(x)
        x = self.dropout(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x

def googlenet():
    return GoogleNet()


