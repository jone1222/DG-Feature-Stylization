import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter
from torch.nn.modules.conv import _ConvNd
from torch.nn.modules.utils import _ntuple
from torch.nn.modules.batchnorm import _BatchNorm

from collections import OrderedDict
import operator
from itertools import islice

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

_pair = _ntuple(2)

__all__ = ['resnet18_InfoDrop', 'resnet34_InfoDrop', 'resnet50_InfoDrop', 'resnet101_InfoDrop', 'resnet152_InfoDrop']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def remove_module_prefix(state_dict):
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    return new_state_dict

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

dropout_layers = 0.5  # how many layers to apply InfoDrop to
finetune_wo_infodrop = False  # when finetuning without InfoDrop, turn this on

class Info_Dropout(nn.Module):
    def __init__(self, indim, outdim, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, if_pool=False, pool_kernel_size=2, pool_stride=None,
                 pool_padding=0, pool_dilation=1):
        super(Info_Dropout, self).__init__()
        if groups != 1:
            raise ValueError('InfoDropout only supports groups=1')

        self.indim = indim
        self.outdim = outdim
        self.if_pool = if_pool
        self.drop_rate = 1.5
        self.temperature = 0.03
        self.band_width = 1.0
        self.radius = 3

        self.patch_sampling_num = 9

        self.all_one_conv_indim_wise = nn.Conv2d(self.patch_sampling_num, self.patch_sampling_num,
                                                 kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation,
                                                 groups=self.patch_sampling_num, bias=False)
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise = nn.Conv2d(self.patch_sampling_num, outdim, kernel_size=1, padding=0, bias=False)
        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


        if if_pool:
            self.pool = nn.MaxPool2d(pool_kernel_size, pool_stride, pool_padding, pool_dilation)

        self.padder = nn.ConstantPad2d((padding + self.radius, padding + self.radius + 1,
                                         padding + self.radius, padding + self.radius + 1), 0)

    def initialize_parameters(self):
        self.all_one_conv_indim_wise.weight.data = torch.ones_like(self.all_one_conv_indim_wise.weight, dtype=torch.float)
        self.all_one_conv_indim_wise.weight.requires_grad = False

        self.all_one_conv_radius_wise.weight.data = torch.ones_like(self.all_one_conv_radius_wise.weight, dtype=torch.float)
        self.all_one_conv_radius_wise.weight.requires_grad = False


    def forward(self, x_old, x):
        if finetune_wo_infodrop:
            return x

        with torch.no_grad():
            distances = []
            padded_x_old = self.padder(x_old)
            sampled_i = torch.randint(-self.radius, self.radius + 1, size=(self.patch_sampling_num,)).tolist()
            sampled_j = torch.randint(-self.radius, self.radius + 1, size=(self.patch_sampling_num,)).tolist()
            for i, j in zip(sampled_i, sampled_j):
                tmp = padded_x_old[:, :, self.radius: -self.radius - 1, self.radius: -self.radius - 1] - \
                      padded_x_old[:, :, self.radius + i: -self.radius - 1 + i,
                      self.radius + j: -self.radius - 1 + j]
                distances.append(tmp.clone())
            distance = torch.cat(distances, dim=1)
            batch_size, _, h_dis, w_dis = distance.shape
            distance = (distance**2).view(-1, self.indim, h_dis, w_dis).sum(dim=1).view(batch_size, -1, h_dis, w_dis)
            distance = self.all_one_conv_indim_wise(distance)
            distance = torch.exp(
                -distance / distance.mean() / 2 / self.band_width ** 2)  # using mean of distance to normalize
            prob = (self.all_one_conv_radius_wise(distance) / self.patch_sampling_num) ** (1 / self.temperature)

            if self.if_pool:
                prob = -self.pool(-prob)  # min pooling of probability
            prob /= prob.sum(dim=(-2, -1), keepdim=True)


            batch_size, channels, h, w = x.shape

            random_choice = random_sample(prob, sampling_num=int(self.drop_rate * h * w))

            random_mask = torch.ones((batch_size * channels, h * w), device='cuda:0')
            random_mask[torch.arange(batch_size * channels, device='cuda:0').view(-1, 1), random_choice] = 0

        return x * random_mask.view(x.shape)

# class Conv2d(nn.Conv2d):
#     def __init__(self, *args, **kwargs):
#         super(Conv2d, self).__init__(*args, **kwargs)
#
#     def forward(self, x, domain_label):
#         return F.conv2d(x, self.weight, self.bias, self.stride,
#                          self.padding, self.dilation, self.groups), domain_label

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
def resnet18_InfoDrop(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InfoDropResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet18']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet34_InfoDrop(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InfoDropResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_zoo.load_url(model_urls['resnet34']))
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet34']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet50_InfoDrop(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InfoDropResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_zoo.load_url(model_urls['resnet50']))
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet34']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet101_InfoDrop(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InfoDropResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_zoo.load_url(model_urls['resnet101']))
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet34']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet152_InfoDrop(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = InfoDropResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_zoo.load_url(model_urls['resnet152']))
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet34']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model


def _update_initial_weights_dsbn(state_dict, num_classes=1000, num_domains=2, dsbn_type='all'):
    new_state_dict = state_dict.copy()

    for key, val in state_dict.items():
        update_dict = False
        if ((('bn' in key or 'downsample.1' in key) and dsbn_type == 'all') or
                (('bn1' in key) and dsbn_type == 'partial-bn1')):
            update_dict = True

        if (update_dict):
            if 'weight' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-6] + 'bns.{}.weight'.format(d)] = val.data.clone()

            elif 'bias' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-4] + 'bns.{}.bias'.format(d)] = val.data.clone()

            if 'running_mean' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-12] + 'bns.{}.running_mean'.format(d)] = val.data.clone()

            if 'running_var' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-11] + 'bns.{}.running_var'.format(d)] = val.data.clone()

            if 'num_batches_tracked' in key:
                for d in range(num_domains):
                    new_state_dict[
                        key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(d)] = val.data.clone()

    if num_classes != 1000 or len([key for key in new_state_dict.keys() if 'fc' in key]) > 1:
        key_list = list(new_state_dict.keys())
        for key in key_list:
            if 'fc' in key:
                print('pretrained {} are not used as initial params.'.format(key))
                del new_state_dict[key]

    return new_state_dict

def _update_initial_weights_InfoDrop(state_dict, num_classes=1000, num_domains=2, dsbn_type='all'):
    new_state_dict = state_dict.copy()

    for key, val in state_dict.items():
        update_dict = False
        if ((('bn' in key or 'downsample.1' in key) and dsbn_type == 'all') or
                (('bn1' in key) and dsbn_type == 'partial-bn1')):
            update_dict = True

        if (update_dict):
            if 'weight' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-6] + 'bns.{}.weight'.format(d)] = val.view(1, -1, 1, 1).data.clone()

            elif 'bias' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-4] + 'bns.{}.bias'.format(d)] = val.view(1, -1, 1, 1).data.clone()

            if 'running_mean' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-12] + 'bns.{}.running_mean'.format(d)] = val.view(1, -1, 1).data.clone()

            if 'running_var' in key:
                for d in range(num_domains):
                    new_state_dict[key[0:-11] + 'bns.{}.running_var'.format(d)] = val.view(1, -1, 1).data.clone()

            if 'num_batches_tracked' in key:
                for d in range(num_domains):
                    new_state_dict[
                        key[0:-len('num_batches_tracked')] + 'bns.{}.num_batches_tracked'.format(d)] = val.data.clone()

    if num_classes != 1000 or len([key for key in new_state_dict.keys() if 'fc' in key]) > 1:
        key_list = list(new_state_dict.keys())
        for key in key_list:
            if 'fc' in key:
                print('pretrained {} are not used as initial params.'.format(key))
                del new_state_dict[key]

    return new_state_dict


@BACKBONE_REGISTRY.register()
class InfoDropResNet(Backbone):
    def __init__(self, block, layers, in_features=256, num_classes=1000):
        self.inplanes = 64
        self.in_features = in_features
        self.num_classes = num_classes
        super(InfoDropResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.info_dropout = Info_Dropout(3, 64, kernel_size=7, stride=2, padding=3, if_pool=True,
                                         pool_kernel_size=3, pool_stride=2, pool_padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self._out_features = 512 * block.expansion

        #   # self.avgpool = nn.AvgPool2d(7, stride=1)
        # if self.in_features != 0:
        #     self.fc1 = nn.Linear(512 * block.expansion, self.in_features)
        #     self.fc2 = nn.Linear(self.in_features, num_classes)
        # else:
        #     self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def _make_layer(self, block, planes, blocks, stride=1, if_dropout=False):
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

    def is_patch_based(self):
        return False

    def get_last_layer_params(self):
        return list(self.class_classifier.parameters()) + list(self.jigsaw_classifier.parameters())

    def featuremaps(self, x):
        x_old = x.clone()
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        if dropout_layers > 0:
            x_new = self.info_dropout(x_old, x)

        x_new = self.layer1(x_new)
        x_new = self.layer2(x_new)
        x_new = self.layer3(x_new)
        x_new = self.layer4(x_new)

        return x_new

    def forward(self, x):
        x = self.featuremaps(x)

        x = x.mean(3).mean(2)  # global average pooling
        x = x.view(x.size(0), -1)

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        # self.bn1 = ModifiedBatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        # self.bn2 = ModifiedBatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out )
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out )

        if self.downsample is not None:
            residual = self.downsample(x )

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        # self.bn1 = ModifiedBatchNorm2d(planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        # self.bn2 = ModifiedBatchNorm2d(planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        # self.bn3 = ModifiedBatchNorm2d(planes * 4)
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
