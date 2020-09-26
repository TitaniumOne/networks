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


class STM_Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(STM_Bottleneck, self).__init__()
        channels = planes // 16
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.cstm_conv1 = nn.Conv3d(planes,
                                    planes,
                                    kernel_size=(3, 1, 1),
                                    groups=planes,
                                    padding=(1, 0, 0),
                                    bias=False)
        self.cstm_bn1 = nn.BatchNorm3d(planes)
        self.cstm_conv2 = nn.Conv3d(planes,
                                    planes,
                                    kernel_size=(1, 3, 3),
                                    padding=(0, 1, 1),
                                    stride=(1, stride, stride),
                                    bias=False)
        self.cstm_bn2 = nn.BatchNorm3d(planes)

        self.cmm_conv1 = nn.Conv3d(planes, channels, kernel_size=1, bias=False)
        self.cmm_bn1 = nn.BatchNorm3d(channels)
        self.cmm_conv = nn.Conv3d(channels,
                                  channels,
                                  kernel_size=(1, 3, 3),
                                  padding=(0, 1, 1),
                                  groups=channels,
                                  bias=False)
        self.cmm_conv2 = nn.Conv3d(channels,
                                   planes,
                                   kernel_size=1,
                                   stride=(1, stride, stride),
                                   bias=False)
        self.cmm_bn2 = nn.BatchNorm3d(planes)

        self.conv3 = nn.Conv3d(planes,
                               planes * self.expansion,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        cstm_x = self.cstm_conv1(x)
        cstm_x = self.cstm_bn1(cstm_x)
        cstm_x = self.cstm_conv2(cstm_x)
        cstm_x = self.cstm_bn2(cstm_x)

        cmm_x = self.cmm_conv1(x)
        cmm_x = self.cmm_bn1(cmm_x)
        cmm_z = self.cmm_conv(cmm_x)
        cmm_z = cmm_z[:, :, 1:] - cmm_x[:, :, :-1]
        cmm_y = torch.cat((cmm_z,
                           torch.zeros(cmm_x.size(0), cmm_x.size(1), 1,
                                       cmm_x.size(3), cmm_x.size(4)).cuda()),
                          2)
        cmm_y = self.cmm_conv2(cmm_y)
        cmm_y = self.cmm_bn2(cmm_y)

        out = cmm_y + cstm_x
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

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        for name, m in self.named_modules():
            if 'cstm_conv1' in name:
                ch = m.weight.size()[0]
                part1 = torch.cat(
                    (torch.ones_like(m.weight[:ch // 4, :, 0:1]),
                     torch.zeros_like(m.weight[:ch // 4, :, 1:])), 2)
                part2 = torch.cat(
                    (torch.zeros_like(m.weight[ch // 4:ch // 4 * 3, :, 0:1]),
                     torch.ones_like(m.weight[ch // 4:ch // 4 * 3, :, 1:2]),
                     torch.zeros_like(m.weight[ch // 4:ch // 4 * 3, :, 2:])),
                    2)
                part3 = torch.cat(
                    (torch.zeros_like(m.weight[ch // 4 * 3:, :, :2]),
                     torch.zeros_like(m.weight[ch // 4 * 3:, :, 2:])), 2)
                m.weight = torch.nn.Parameter(
                    torch.cat((part1, part2, part3), 0))

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

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
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
    model = ResNet(STM_Bottleneck, [3, 4, 6, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet50'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln and not 'cstm' in ln or 'cmm' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)

    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    Args:
        groups
    """
    model = ResNet(STM_Bottleneck, [3, 4, 23, 3], **kwargs)
    checkpoint = model_zoo.load_url(model_urls['resnet101'])
    layer_name = list(checkpoint.keys())
    for ln in layer_name:
        if 'conv' in ln or 'downsample.0.weight' in ln and not 'cstm' in ln or 'cmm' in ln:
            checkpoint[ln] = checkpoint[ln].unsqueeze(2)
    model.load_state_dict(checkpoint, strict=False)

    return model
