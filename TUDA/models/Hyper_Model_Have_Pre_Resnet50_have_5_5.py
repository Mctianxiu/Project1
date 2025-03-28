#============================
#PUIQA：使用在 ImageNet 数据上预先训练的 ResNet50 网络作为我们的初始化骨干模型
#含有5x5的层
#============================
import cv2
import math
import torch as torch
import torch.nn as nn
from torch.nn import init
import torchvision.transforms as transforms


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

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetBackbone(nn.Module):

    def __init__(self, lda_out_channels, in_chn, block, layers, num_classes=1000):
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

        # local distortion aware module
        self.lda1_pool = nn.Sequential(
            nn.Conv2d(256, 16, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda1_fc = nn.Linear(16 * 64, lda_out_channels)

        self.lda2_pool = nn.Sequential(
            nn.Conv2d(512, 32, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda2_fc = nn.Linear(32 * 16, lda_out_channels)

        self.lda3_pool = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(7, stride=7),
        )
        self.lda3_fc = nn.Linear(64 * 4, lda_out_channels)

        self.lda4_pool = nn.AvgPool2d(7, stride=7)
        self.lda4_fc = nn.Linear(2048, 1024 - lda_out_channels * 3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # initialize
        nn.init.kaiming_normal_(self.lda1_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda2_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda3_pool._modules['0'].weight.data)
        nn.init.kaiming_normal_(self.lda1_fc.weight.data)
        nn.init.kaiming_normal_(self.lda2_fc.weight.data)
        nn.init.kaiming_normal_(self.lda3_fc.weight.data)
        nn.init.kaiming_normal_(self.lda4_fc.weight.data)

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

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)

        # the same effect as lda operation in the paper, but save much more memory
        lda_1 = self.lda1_pool(x)
        lda_1 = lda_1.view(x.size(0), -1)
        lda_1 = self.lda1_fc(lda_1)
        x = self.layer2(x)
        lda_2 = self.lda2_fc(self.lda2_pool(x).view(x.size(0), -1))
        x = self.layer3(x)
        lda_3 = self.lda3_fc(self.lda3_pool(x).view(x.size(0), -1))
        x = self.layer4(x)
        lda_4 = self.lda4_fc(self.lda4_pool(x).view(x.size(0), -1))

        out = torch.cat((lda_1, lda_2, lda_3, lda_4), 1)
        return out


def resnet50_backbone(lda_out_channels, in_chn, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model_hyper.

    Args:
        pretrained (bool): If True, returns a model_hyper pre-trained on ImageNet
    """
    model = ResNetBackbone(lda_out_channels, in_chn, Bottleneck, [3, 4, 6, 3], **kwargs)
    #
    if pretrained:
        print('~Pretrained Resnet50 no 5-5~')
        from models import ResNet
        model1 = ResNet.resnet50().cuda()
        model1.load_state_dict(torch.load('./Snapshots/Rank_train_Resnet50_0.00002_have_5_5_10/epoch200.pth'))
        save_model = model1.state_dict()
        #
        model_dict = model.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        model.load_state_dict(model_dict)
    return model


def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    # if isinstance(m, nn.Conv2d):
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data)
    elif classname.find('BatchNorm2d') != -1:
        init.uniform_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


class UWIQN(nn.Module):
    def __init__(self, inputsize):
        super(UWIQN, self).__init__()
        #
        self.inputsize = inputsize
        self.resnet50 = resnet50_backbone(128, self.inputsize, pretrained=True)
        #
        self.fc1 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,512),
            nn.ReLU(),
            nn.Linear(512, 4)
        )
        self.fc2 = nn.Linear(4, 1)

    def forward(self, x):
        x1 = self.resnet50(x)
        x1 = self.fc1(x1)
        y1 = self.fc2(x1)
        return y1


if __name__ == '__main__' :
    model = resnet50_backbone(32, 224, pretrained=True).cpu()
    # model = UWIQN(224,4)##.cuda()
    model.eval()
    img = cv2.imread('hp.bmp')
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    img = transform(img)##.cuda()
    img = img.unsqueeze(0)
    y = torch.rand(4)
    y = y.unsqueeze(0)
    result = model(img)
    print(result)



