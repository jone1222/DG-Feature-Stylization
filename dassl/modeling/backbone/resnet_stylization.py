import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

_pair = _ntuple(2)

__all__ = ['resnet18_stylize', 'resnet34_stylize', 'resnet50_stylize', 'resnet101_stylize', 'resnet152_stylize']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

class FeatureStylizationBlock(nn.Module):
    def __init__(self, **kwargs):
        super(FeatureStylizationBlock, self).__init__()

        scaling_factor = kwargs['scaling_factor']
        encode_mode = kwargs['encode_mode']

        self.scaling_factor = scaling_factor
        self.encode_mode = encode_mode
        self.training = True

    def forward(self, x):
        LL, HH = self.encode_LL_HH(x, encode_mode=self.encode_mode)

        LL_transformed = self.stylize_feature(LL)
        x_transformed = self.decode_LL_HH(LL_transformed, HH)

        return x_transformed

    def encode_LL_HH(self, x, encode_mode = 'pooling', interpolate_mode='nearest'):
        if encode_mode == 'pooling':
            pooled = torch.nn.functional.avg_pool2d(x, 2)
            up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode=interpolate_mode)
            HH = x - up_pooled
            LL = up_pooled
        elif encode_mode == 'no_pooling':
            HH = torch.zeros_like(x)
            LL = x
        elif encode_mode == 'reverse_pooling':
            pooled = torch.nn.functional.avg_pool2d(x, 2)
            up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode=interpolate_mode)
            HH = up_pooled
            LL = x - up_pooled
        else:
            pooled = torch.nn.functional.avg_pool2d(x, 2)
            up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode=interpolate_mode)
            HH = x - up_pooled
            LL = up_pooled

        return LL, HH

    def stylize_feature(self, LL):
        B, C, H, W = LL.shape
        LL_cp = LL.view(C, -1)  # MEAN / STD : (C) -> batch-wise mean of each-channel

        # Calculate batch-wise statistics
        mean_LL = torch.mean(LL_cp, dim=1) # batch-wise mean
        std_LL = torch.std(LL_cp, dim=1) # batch-wise std

        # Calculate channel-wise statistics of mean and std vector
        mu_hat_LL = mean_LL.mean()
        sigma_hat_LL = mean_LL.std()
        mu_tilde_LL = std_LL.mean()
        sigma_tilde_LL = std_LL.std()

        # Sample new style vectors from the manipulated distribution
        mu_new = torch.normal(mu_hat_LL.view(1, 1).repeat(B, C), self.scaling_factor * sigma_hat_LL.view(1, 1).repeat(B, C)) #output : (B, C)
        sigma_new = torch.normal(mu_tilde_LL.view(1, 1).repeat(B, C), self.scaling_factor * sigma_tilde_LL.view(1,1).repeat(B, C))

        mu_new_reshape = mu_new.view(B, C, 1, 1).repeat(1, 1, H, W)
        sigma_new_reshpae = sigma_new.view(B, C, 1, 1).repeat(1, 1, H, W)

        # Equation 6 ~ Normalize original feature with batch statistics
        normalized_LL = (LL - mean_LL.view(1, C, 1, 1).repeat(B, 1, H, W)) / std_LL.view(1, C, 1, 1).repeat(B, 1, H, W)
        # Equation 6 ~ Affine transformation with sampled style vectors
        stylized_LL = sigma_new_reshpae * normalized_LL + mu_new_reshape

        return stylized_LL

    def decode_LL_HH(self, LL, HH):
        return HH + LL

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, planes * self.expansion, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
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

class Conv2d(_ConvNd):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(Conv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode='zeros')

    def forward(self, input):
        return F.conv2d(input, self.weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)



@BACKBONE_REGISTRY.register()
def resnet18_stylize(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = StylizeResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet34_stylize(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = StylizeResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet34'])
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet50_stylize(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = StylizeResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet50'])

        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet101_stylize(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = StylizeResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_zoo.load_url(model_urls['resnet101']))

        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet152_stylize(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = StylizeResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_zoo.load_url(model_urls['resnet152']))

        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
class StylizeResNet(Backbone):
    def __init__(self, block, layers, in_features=256, num_classes=1000, **kwargs):
        self.inplanes = 64
        self.in_features = in_features
        self.num_classes = num_classes
        super(StylizeResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], )
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self._out_features = 512 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


        self.stylization_layer = FeatureStylizationBlock(**kwargs)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def featuremaps(self, x, stylization_layer_idx = 2):

        assert 0 <= stylization_layer_idx <= 4, 'Target stylization layer should be between 0 and 4'

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        if stylization_layer_idx == 0:
            x_tr = self.stylization_layer(x)

        x = self.layer1(x)
        if stylization_layer_idx == 1:
            x_tr = self.stylization_layer(x)
        elif stylization_layer_idx < 1:
            x_tr = self.layer1(x_tr)

        x = self.layer2(x)
        if stylization_layer_idx == 2:
            x_tr = self.stylization_layer(x)
        elif stylization_layer_idx < 2:
            x_tr = self.layer2(x_tr)

        x = self.layer3(x)
        if stylization_layer_idx == 3:
            x_tr = self.stylization_layer(x)
        elif stylization_layer_idx < 3:
            x_tr = self.layer3(x_tr)

        x = self.layer4(x)
        if stylization_layer_idx == 4:
            x_tr = self.stylization_layer(x)
        elif stylization_layer_idx < 4:
            x_tr = self.layer4(x_tr)

        if not self.stylization_layer.training:
            x_tr = x

        return x, x_tr

    def forward(self, x, stylization_layer_idx = 2):

        x, x_tr = self.featuremaps(x, stylization_layer_idx)

        x = x.mean(3).mean(2)  # global average pooling
        x = x.view(x.size(0), -1)

        x_tr = x_tr.mean(3).mean(2)  # global average pooling
        x_tr = x_tr.view(x_tr.size(0), -1)

        return x, x_tr