import torch
import torch.nn as nn
import torch.nn.functional as F

choices=['LeakyReLU', 'ReLU', 'Softplus']

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, act, stride=1):
        super(BasicBlock, self).__init__()
        choices=['LeakyReLU', 'ReLU', 'Softplus']
        assert act in choices
        self.arg_act=act
        if act==choices[0]:
            self.act=F.leaky_relu
        elif act==choices[1]:
            self.act=F.relu
        elif act==choices[2]:
            self.act=F.softplus
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, act, stride=1):
        super(Bottleneck, self).__init__()
        choices=['LeakyReLU', 'ReLU', 'Softplus']
        assert act in choices
        self.arg_act=act
        if act==choices[0]:
            self.act=F.leaky_relu
        elif act==choices[1]:
            self.act=F.relu
        elif act==choices[2]:
            self.act=F.softplus
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.act(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.act(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_channels, act, num_classes=10):
        super(ResNet, self).__init__()
        choices=['LeakyReLU', 'ReLU', 'Softplus']
        assert act in choices
        self.arg_act=act
        if act==choices[0]:
            self.act=F.leaky_relu
        elif act==choices[1]:
            self.act=F.relu
        elif act==choices[2]:
            self.act=F.softplus
        self.in_planes = num_channels[0]

        self.conv1 = nn.Conv2d(3, num_channels[0], kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels[0])
        self.layer1 = self._make_layer(block, num_channels[0], num_blocks[0], self.arg_act, stride=1)
        self.layer2 = self._make_layer(block, num_channels[1], num_blocks[1], self.arg_act, stride=2)
        self.layer3 = self._make_layer(block, num_channels[2], num_blocks[2], self.arg_act, stride=2)
        self.layer4 = self._make_layer(block, num_channels[3], num_blocks[3], self.arg_act, stride=2)
        self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(num_channels[3]*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, act, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, act, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.act(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNet18():
    return ResNet(BasicBlock, [2, 2, 2, 2], [64,128,256,512])


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3], [64,128,256,512])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3], [64,128,256,512])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3], [64,128,256,512])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3], [64,128,256,512])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()

