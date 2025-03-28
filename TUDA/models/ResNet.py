#==============================================
#ResNet50 主要在基于排序的质量评估（RUIQA）模块里被使用
#==============================================

import math
import torch as torch
import torch.nn as nn


__all__ = ['resnet50']


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        #
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        #
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        #
        out = self.conv3(out)
        out = self.bn3(out)
        #
        if self.downsample is not None:
            residual = self.downsample(x)
        #
        out += residual
        out = self.relu(out)
        #
        return out


class ResNetBackbone(nn.Module):

    def __init__(self, block, layers):
        super(ResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        #
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        #
        return nn.Sequential(*layers)

    def forward(self, x):  # 1,3,256,256
        x = self.conv1(x)  # 1,64,128,128
        x = self.bn1(x)  # 1,64,128,128
        x = self.relu(x)  # 1,64,128,128
        x = self.maxpool(x)  # 1,64,64,64
        #
        x = self.layer1(x)  # 1,256,64,64
        x = self.layer2(x)  # 1,512,32,32
        x = self.layer3(x)  # 1,1024,16,16
        x = self.layer4(x)  # 1,2048,8,8
        #
        x = self.avgpool(x)  # torch.Size([1, 2048, 16, 16])
        x = torch.flatten(x, 1)  # torch.Size([1, 2048*16*16])
        out = self.fc(x)
        return out


def resnet50(**kwargs):
    """Constructs a ResNet-50 model_hyper.
    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    return ResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)  # 16, 224


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #
    input = torch.FloatTensor(1, 3, 256, 256).cuda()  # torch.Size([1, 3, 224, 224])
    #
    net = resnet50().cuda()
    out = net(input)
    print('ok~')