"""vgg in pytorch


[1] Karen Simonyan, Andrew Zisserman

    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
'''VGG11/13/16/19 in Pytorch.'''

import torch
import torch.nn as nn

cfg = {
    'A' : [64,     'M', 128,      'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'B' : [64, 64, 'M', 128, 128, 'M', 256, 256,           'M', 512, 512,           'M', 512, 512,           'M'],
    'D' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256,      'M', 512, 512, 512,      'M', 512, 512, 512,      'M'],
    'E' : [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}

class VGG(nn.Module):

    def __init__(self, features, num_class=100):
        super().__init__()
        self.features = features

        self.classifier = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_class)
        )

        # # 以下网络是修改部分
        # # 跑代码的时候，可以设置断点，比较前后四种channel是否对应或者合理
        # self.conv0 = nn.Conv2d(3, 32, kernel_size=1, padding=0)  # 把第一个channel的数值：3升到32，最后再变回3（考虑到变量in_channels=3）
        # self.conv00 = nn.Conv2d(32, 3, kernel_size=1, padding=0)  # 32变回3，改进模块结束
        # self.conv1 = nn.Conv2d(32, 8, kernel_size=1, padding=0)  # 卷积模块1的x1
        # self.conv11 = nn.Conv2d(8, 8, kernel_size=1, padding=0)  # 卷积模块1的x1通道之后，第一个卷积1*1
        # self.conv12 = nn.Conv2d(8, 8, kernel_size=3, padding=1)  # 卷积模块1的x1通道之后，第二个卷积3*3
        # self.conv13 = nn.Conv2d(8, 8, kernel_size=5, padding=2)  # 卷积模块1的x1通道之后，第三个卷积5*5
        # self.conv2 = nn.Conv2d(32, 16, kernel_size=1, padding=0)  # 卷积模块1的x2
        # self.conv3 = nn.Conv2d(32, 8, kernel_size=1, padding=0)  # 卷积模块1的x3，和self.conv1一样，只不过这样写清清晰点
        # self.conv31 = nn.Conv2d(8, 2, kernel_size=1, padding=0)  # 卷积模块1的x3通道之后，第一个卷积1*1(1/16)
        # self.conv32 = nn.Conv2d(2, 2, kernel_size=3, padding=1)  # 卷积模块1的x3通道之后，第二个卷积3*3(1/16)
        # self.conv33 = nn.Conv2d(2, 8, kernel_size=1, padding=0)  # 卷积模块1的x3通道之后，第三个卷积1*1(1/4)

    def forward(self, x):
        # # 以下网络是修改部分
        # x = self.conv0(x)
        # # x1
        # x1 = self.conv1(x)
        # x1 = torch.add(torch.add(self.conv11(x1), self.conv12(x1)), self.conv13(x1))
        # # x2
        # x2 = self.conv2(x)
        # # x3
        # x3 = self.conv33(self.conv32(self.conv31(self.conv3(x))))
        # # 卷积模块1输出
        # x = self.conv00(torch.cat([torch.cat([x1, x2], 1), x3], 1))  # 如果这里出错，也可能是torch.cat拼接的问题（按行拼，还是按列拼）

        output = self.features(x)
        output = output.view(output.size()[0], -1)
        output = self.classifier(output)

        return output

def make_layers(cfg, batch_norm=False):
    layers = []

    input_channel = 3
    for l in cfg:
        if l == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            continue

        layers += [nn.Conv2d(input_channel, l, kernel_size=3, padding=1)]

        if batch_norm:
            layers += [nn.BatchNorm2d(l)]

        layers += [nn.ReLU(inplace=True)]
        input_channel = l

    return nn.Sequential(*layers)

def vgg11_bn():
    return VGG(make_layers(cfg['A'], batch_norm=True))

def vgg13_bn():
    return VGG(make_layers(cfg['B'], batch_norm=True))

def vgg16_bn():
    return VGG(make_layers(cfg['D'], batch_norm=True))

def vgg19_bn():
    return VGG(make_layers(cfg['E'], batch_norm=True))


