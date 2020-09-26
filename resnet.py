import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torchvision.models as models

__all__ = ['ResNet', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes,
                               planes,
                               kernel_size=(1, 3, 3),
                               stride=(1, stride, stride),
                               padding=(0, 1, 1),
                               bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
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
            residual = self.downsample(residual)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, resi=False):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(3,
                               64,
                               kernel_size=(1, 7, 7),
                               stride=(1, 2, 2),
                               padding=(0, 3, 3),
                               bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3),
                                    stride=(1, 2, 2),
                                    padding=(0, 1, 1))
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.resi = resi
        if resi:

            self.beta = 1 / 8.0
            #self.inplanes=int(256*self.beta)
            #self.lateral_layer2=self._make_layer(block, int(64*self.beta), 1)
            #self.backward_layer2=self._make_layer(block, int(64*self.beta), 1)
            self.inplanes = int(512 * self.beta)
            self.lateral_layer3 = self._make_layer(block, int(128 * self.beta),
                                                   1)
            self.backward_layer3 = self._make_layer(block,
                                                    int(128 * self.beta), 1)
            self.inplanes = int(1024 * self.beta)
            self.lateral_layer4 = self._make_layer(block, int(256 * self.beta),
                                                   1)
            self.backward_layer4 = self._make_layer(block,
                                                    int(256 * self.beta), 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(
                    m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes,
                          planes * block.expansion,
                          kernel_size=1,
                          stride=(1, stride, stride),
                          bias=False),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def lateral_connection(self, input, lateral_layer, backward_layer=None):

        n, c, t, h, w = input.size()
        channels = int(c * self.beta)
        output = torch.zeros_like(input[:, :channels])
        output_back = torch.zeros_like(input[:, :channels])
        output[:, :,
               1:] = input[:, 2 * channels:3 *
                           channels, :-1] - input[:, 2 * channels:3 * channels,
                                                  1:]
        output_back[:, :, :-1] = input[:, 2 * channels:3 * channels,
                                       1:] - input[:, 2 * channels:3 *
                                                   channels, :-1]
        output = lateral_layer(output)
        output_back = backward_layer(output_back)
        out = torch.cat((output, output_back, input[:, channels * 2:]), 1)
        return out

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        #if self.resi:
        #    x = self.lateral_connection(x,self.lateral_layer2,self.backward_layer2)
        x = self.layer2(x)
        if self.resi:
            x = self.lateral_connection(x, self.lateral_layer3,
                                        self.backward_layer3)
        x = self.layer3(x)
        if self.resi:
            x = self.lateral_connection(x, self.lateral_layer4,
                                        self.backward_layer4)
        x = self.layer4(x)
        x = x.transpose(1, 2).contiguous()
        x = x.view((-1, ) + x.size()[2:])

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet50(**kwargs):
    """Constructs a ResNet-50 based model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet50'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln and not 'lateral' in layer_name:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        groups
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet101'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln and not 'lateral' in layer_name:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)
    return model
