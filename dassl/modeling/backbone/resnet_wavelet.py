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
from torchvision.transforms import RandomPerspective
import numpy as np
import itertools

from .build import BACKBONE_REGISTRY
from .backbone import Backbone
from ..ops.wct import feature_wct_simple, SwitchWhiten2d
from ..ops.mixup import mixup_xonly

_pair = _ntuple(2)

__all__ = ['resnet18_texture', 'resnet34_texture', 'resnet50_texture', 'resnet101_texture', 'resnet152_texture']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}
dropout_layers = 0.5  # how many layers to apply InfoDrop to
finetune_wo_infodrop = False  # when finetuning without InfoDrop, turn this on

def random_sample(prob, sampling_num):
    batch_size, channels, h, w = prob.shape
    return torch.multinomial((prob.view(batch_size * channels, -1) + 1e-8), sampling_num, replacement=True)

def gaussian_blur(x, kernel_size=9):
    B, C, W, H = x.size()

    if B == 0:
        return x

    channels = C
    x_cord = torch.arange(kernel_size + 0.)
    x_grid = x_cord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1)
    mean = (kernel_size - 1) // 2
    diff = -torch.sum((xy_grid - mean) ** 2., dim=-1)
    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels,
                                      kernel_size=kernel_size, groups=channels, bias=False)

    gaussian_filter.weight.requires_grad = False

    sigma = 16
    variance = sigma ** 2.
    gaussian_kernel = (1. / (2. * math.pi * variance)) * torch.exp(diff / (2 * variance))
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)
    gaussian_kernel = gaussian_kernel.cuda()
    gaussian_filter.weight.data = gaussian_kernel
    output = gaussian_filter(torch.nn.functional.pad(x, (mean, mean, mean, mean), "replicate"))

    return output

def random_perspective(distortion_scale=0.5, p=0.5, interpolation=2, fill=0):
    rand_pers = RandomPerspective(distortion_scale, p, interpolation, fill)
    return rand_pers

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
        # self.temperature = 0.03
        self.temperature = 0.1
        self.band_width = 1.0
        # self.radius = 3
        self.radius = 1

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

        return x * random_mask.view(x.shape), random_mask.view(x.shape)


# class TPS(nn.Module):
#
#     def __init__(self, span_range_height, span_range_width, grid_height, grid_width):
#         super(self, TPS).__init__()
#         r1 = span_range_height
#         r2 = span_range_width
#         assert r1 < 1 and r2 < 1  # if >= 1, arctanh will cause error in BoundedGridLocNet
#         target_control_points = torch.Tensor(list(itertools.product(
#             np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
#             np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
#         )))
#         Y, X = target_control_points.split(1, dim=1)
#         target_control_points = torch.cat([X, Y], dim=1)
#
#         self.tps = TPSGridGen(args.image_height, args.image_width, target_control_points)
#
#     def forward(self, x):
#         batch_size = x.size(0)
#         source_control_points = self.loc_net(x)
#         source_coordinate = self.tps(source_control_points)
#         grid = source_coordinate.view(batch_size, self.args.image_height, self.args.image_width, 2)
#         transformed_x = grid_sample(x, grid)
#
#         return transformed_x

class TextureTransformBlock(nn.Module):
    def __init__(self, **kwargs):
        super(TextureTransformBlock, self).__init__()

        transform_LL = kwargs['transform_LL']
        transform_HH = kwargs['transform_HH']
        LL_permute_scale = kwargs['LL_permute_scale']
        HH_kernel_size = kwargs['HH_kernel_size']
        encode_mode = kwargs['encode_mode']

        use_growing = kwargs['use_growing']

        self.LL_permute_scale = LL_permute_scale
        self.HH_kernel_size = HH_kernel_size

        self.LL_permute_scale_max = self.LL_permute_scale
        self.use_growing = use_growing

        self.encode_mode = encode_mode

        self.LL_transform_ratio = kwargs['LL_transform_ratio']
        self.HH_transform_ratio = kwargs['HH_transform_ratio']
        self.BOTH_transform_ratio = kwargs['BOTH_transform_ratio']

        assert (transform_LL in ['drop', 'permute']), 'The transform must be either drop or permute'
        if transform_LL == 'drop':
            self.transform_LL = self.drop_LL
        elif transform_LL == 'permute':
            self.transform_LL = self.permute_LL
        else:
            self.transform_LL = self.drop_LL

        assert (transform_HH in ['blur', 'perspective']), 'The transform must be either blur or perspective'
        if transform_HH == 'blur':
            self.transform_HH = gaussian_blur
        elif transform_HH == 'perspective':
            self.transform_HH = random_perspective()
        else:
            self.transform_HH = gaussian_blur

        self.training = True

        # self.info_dropout0 = Info_Dropout(64, 64, kernel_size=7, stride=2, padding=3, if_pool=True,
        #                                  pool_kernel_size=3, pool_stride=2, pool_padding=1)

        # self.info_dropout0 = Info_Dropout(64, 64, kernel_size=5, stride=1, padding=2, if_pool=True,
        #                                  pool_kernel_size=3, pool_stride=2, pool_padding=1)

        self.info_dropout0 = Info_Dropout(64, 64, kernel_size=1, stride=1, padding=0, if_pool=False)

    def forward(self, x, cur_epoch_ratio = 0):
        B = x.shape[0]
        # LL, HH = self.encode_LL_HH(x, encode_mode='infodrop')
        # LL, HH = self.encode_LL_HH(x, encode_mode='pooling')
        LL, HH = self.encode_LL_HH(x, encode_mode=self.encode_mode)

        LL_transform_idx = int(self.LL_transform_ratio * B)
        HH_transform_idx = LL_transform_idx + int(self.HH_transform_ratio * B)
        BOTH_transform_idx = HH_transform_idx + int(self.BOTH_transform_ratio * B)

        LL_transformed = self.transform_LL(LL[:LL_transform_idx], cur_epoch_ratio=cur_epoch_ratio)
        x_transformed_LL = self.decode_LL_HH(LL_transformed, HH[:LL_transform_idx])

        if self.HH_transform_ratio > 0:
            HH_transformed = self.transform_HH(HH[LL_transform_idx:HH_transform_idx])
            x_transformed_HH = self.decode_LL_HH(LL[LL_transform_idx:HH_transform_idx], HH_transformed)
        else:
            x_transformed_HH = self.decode_LL_HH(LL[LL_transform_idx:HH_transform_idx], HH[LL_transform_idx:HH_transform_idx])

        if self.BOTH_transform_ratio > 0:
            BOTH_transformed_LL = self.transform_LL(LL[HH_transform_idx:BOTH_transform_idx])
            BOTH_transformed_HH = self.transform_HH(HH[HH_transform_idx:BOTH_transform_idx])
            x_transformed_BOTH = self.decode_LL_HH(BOTH_transformed_LL, BOTH_transformed_HH)
        else:
            x_transformed_BOTH = self.decode_LL_HH(LL[HH_transform_idx:BOTH_transform_idx], HH[HH_transform_idx:BOTH_transform_idx])

        x_transformed = torch.cat([x_transformed_LL, x_transformed_HH, x_transformed_BOTH, x[BOTH_transform_idx:]], dim=0)

        #mixup
        # x_transformed = mixup_xonly(x, x_transformed, beta=1.0)


        # loss_sim = self.calculate_loss(x_transformed, LL_transformed)

        return x_transformed

    def encode_LL_HH(self, x, encode_mode = 'pooling', interpolate_mode='nearest'):
        if encode_mode == 'pooling':
            pooled = torch.nn.functional.avg_pool2d(x, 2)
            up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode=interpolate_mode)
            HH = x - up_pooled
            LL = up_pooled
        elif encode_mode == 'no_pooling':
            # pooled = torch.nn.functional.avg_pool2d(x, 2)
            # up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode=interpolate_mode)
            HH = torch.zeros_like(x)
            LL = x
        elif encode_mode == 'reverse_pooling':
            pooled = torch.nn.functional.avg_pool2d(x, 2)
            up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode=interpolate_mode)
            HH = up_pooled
            LL = x - up_pooled
        elif encode_mode == 'infodrop':
            HH, prob = self.get_infodrop(x)
            LL = x - HH
        elif encode_mode == 'svd':
            # import pdb; pdb.set_trace()
            pooled = torch.nn.functional.avg_pool2d(x, 2)
            up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode=interpolate_mode)
            HH = x - up_pooled
            LL = up_pooled

            whiten_cF, wm_c, c_mean = SwitchWhiten2d(LL)
            LL = (whiten_cF, wm_c, c_mean)
        else:
            pooled = torch.nn.functional.avg_pool2d(x, 2)
            up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode=interpolate_mode)
            HH = x - up_pooled
            LL = up_pooled

        # self.HH = HH.detach()
        # self.HH = prob.detach()

        return LL, HH

    def visualize_HH(self, x_org, HH):
        # Channel_wise Pooling --> B x 1 x H x W
        HH = torch.mean(HH, dim=1, keepdim=True)
        B, C, H, W = HH.shape

        # min max normalization
        HH = HH.view(B, -1)
        HH = HH - HH.min(dim=1)[0].unsqueeze(1)
        HH = HH / HH.max(dim=1)[0].unsqueeze(1)
        HH = HH.view(B, 1, H, W)

        B, C, H, W = HH.shape
        HH_up = torch.nn.functional.interpolate(HH, size=[x_org.size(2), x_org.size(3)], mode='bilinear').repeat(1, 3, 1, 1)
        img_out = torch.cat([x_org, HH_up], dim=3)
        from torchvision.utils import make_grid, save_image
        save_image(make_grid(img_out, nrow=8), "./HH_img_k1_art_prob_rad1_temp0.1.jpg")

    def drop_LL(self, LL, ratio=0.5):
        mask_distrib = torch.distributions.Bernoulli(probs=ratio)
        mask = mask_distrib.sample(LL.size()).cuda()

        return mask * LL

    def permute_LL(self, LL, cur_epoch_ratio = 0):
        B, C, H, W = LL.shape
        LL_cp = LL.view(C, -1)  # MEAN / STD : (C) -> batch-wise mean of each-channel

        mean_LL = torch.mean(LL_cp, dim=1) # mean will be used for bias
        std_LL = torch.std(LL_cp, dim=1) # std will be used for scaling
        self.mean_LL = mean_LL.detach()
        self.std_LL = std_LL.detach()

        bias = torch.normal(mean_LL.mean().view(1,1).repeat(B, C), self.LL_permute_scale * mean_LL.std().view(1,1).repeat(B, C)).view(B, C, 1, 1).repeat(1, 1, H, W) #output : (B, C)
        scale = torch.normal(std_LL.mean().view(1,1).repeat(B, C), self.LL_permute_scale * std_LL.std().view(1,1).repeat(B, C)).view(B, C, 1, 1).repeat(1, 1, H, W)

        permuted_LL = scale * ((LL - mean_LL.view(1, C, 1, 1).repeat(B, 1, H, W)) / std_LL.view(1, C, 1, 1).repeat(B, 1, H, W)) + bias
        #
        # torch.normal(mean_LL, std_LL, size=)
        return permuted_LL

    def permute_LL_new(self, LL, cur_epoch_ratio = 0):
        B, C, H, W = LL.shape
        LL_cp = LL.view(C, -1)  # MEAN / STD : (C) -> batch-wise mean of each-channel

        mean_LL = torch.mean(LL_cp, dim=1) # mean will be used for bias
        std_LL = torch.std(LL_cp, dim=1) # std will be used for scaling

        bias = torch.normal(mean_LL.mean().view(1,1).repeat(B, C), self.LL_permute_scale * mean_LL.std().view(1,1).repeat(B, C)).view(B, C, 1, 1).repeat(1, 1, H, W) #output : (B, C)
        scale = torch.normal(std_LL.mean().view(1,1).repeat(B, C), self.LL_permute_scale * std_LL.std().view(1,1).repeat(B, C)).view(B, C, 1, 1).repeat(1, 1, H, W)

        permuted_LL = scale * ((LL - mean_LL.view(1, C, 1, 1).repeat(B, 1, H, W)) / std_LL.view(1, C, 1, 1).repeat(B, 1, H, W)) + bias
        #
        # torch.normal(mean_LL, std_LL, size=)
        return permuted_LL

    def decode_LL_HH(self, LL, HH):
        return HH + LL

    def decode_svd_LL_HH(self, LL, HH, c_mean, H, W):
        N, C, _ = HH.shape
        wm = LL
        whiten_cF = HH
        targetFeature = torch.bmm(torch.inverse(wm), whiten_cF)

        targetFeature = targetFeature.view(N, C, H, W)
        targetFeature = targetFeature + c_mean.unsqueeze(2).expand_as(targetFeature)
        # targetFeature.clamp_(cont_min, cont_max)

        return targetFeature

    def calculate_loss(self, x, x_tr):
        B, C, H, W = x.shape

        x = x.view(B, -1)
        x_tr = x_tr.view(B, -1)

        loss_sim = torch.mean(F.cosine_similarity(x, x_tr))

        return loss_sim

    def get_infodrop(self, x):
        with torch.no_grad():
            old_out = x.clone()

            output = self.info_dropout0(old_out, x)
            return output

class SVDTextureTransformBlock(nn.Module):
    def __init__(self, **kwargs):
        super(SVDTextureTransformBlock, self).__init__()

        LL_permute_scale = kwargs['LL_permute_scale']

        self.LL_permute_scale = LL_permute_scale


        self.training = True

    def forward(self, x):
        pooled = torch.nn.functional.avg_pool2d(x, 2)
        up_pooled = torch.nn.functional.interpolate(pooled, size=[x.size(2), x.size(3)], mode='nearest')
        HH = x - up_pooled
        LL = up_pooled

        content_feat = LL
        N, C, H, W = content_feat.size()
        cont_min = content_feat.min().item()
        cont_max = content_feat.max().item()

        whiten_cF, wm_c, c_mean = SwitchWhiten2d(content_feat)
        wm_s, s_mean = self.randomize_texture(wm_c, c_mean)

        targetFeature = torch.bmm(torch.inverse(wm_s), whiten_cF)
        targetFeature = targetFeature.view(N, C, H, W)
        targetFeature = targetFeature + s_mean.unsqueeze(2).expand_as(targetFeature)
        targetFeature.clamp_(cont_min, cont_max)


        new_x = HH + targetFeature
        return new_x

    def randomize_texture(self, wm, mean):

        B, C, _ = wm.shape
        wm_triu = torch.triu(wm)
        wm_trid = wm - wm_triu # get elements without diagonal

        wm_diag = torch.diagonal(wm, dim1=1, dim2=2)
        wm_diag = torch.eye(C).cuda().unsqueeze(0).repeat(B, 1, 1) * wm_diag.unsqueeze(2).repeat(1, 1, C)

        # wm = wm.view(B, -1)
        wm_trid = wm_trid.view(B, -1)
        num_elem_tri = C * (C-1) / 2
        wm_trid_mean = (torch.sum(wm_trid, dim=1) / num_elem_tri)
        wm_trid_std = torch.sqrt(torch.sum(torch.pow(wm_trid - wm_trid_mean.unsqueeze(1).repeat(1, C*C), 2), dim=1) / num_elem_tri)
        # .unsqueeze(2).repeat(1, 1, C)
        wm_new = torch.normal(wm_trid_mean.view(B, 1, 1).repeat(1, C, C), 10 * wm_trid_std.view(B, 1, 1).repeat(1, C, C))
        wm_triu_new = torch.triu(wm_new)
        wm_trid_new = wm_triu_new.transpose(dim0=2, dim1=1)
        wm_diag_new = torch.diagonal(wm_new, dim1=1, dim2=2).unsqueeze(2).repeat(1, 1, C)
        wm_diag_new = torch.eye(C).cuda().unsqueeze(0).repeat(B, 1, 1) * wm_diag_new

        wm_new = wm_triu_new + wm_trid_new - (wm_diag_new * 2) + wm_diag

        mean_new = mean
        return wm_new, mean_new

    def calculate_loss(self, x, x_tr):
        B, C, H, W = x.shape

        x = x.view(B, -1)
        x_tr = x_tr.view(B, -1)

        loss_sim = torch.mean(F.cosine_similarity(x, x_tr))

        return loss_sim


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
def resnet18_texture(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TextureResNet(BasicBlock, [2, 2, 2, 2], **kwargs)

    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet18'])
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet18']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet34_texture(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TextureResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet34'])
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet34']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet50_texture(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TextureResNet(Bottleneck, [3, 4, 6, 3], **kwargs)

    if pretrained:
        pretrain_dict = model_zoo.load_url(model_urls['resnet50'])
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet34']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet101_texture(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TextureResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pretrain_dict = model_zoo.load_url(model_zoo.load_url(model_urls['resnet101']))
        # updated_state_dict = _update_initial_weights_texture(model_zoo.load_url(model_urls['resnet34']),
        #                                                   num_classes=model.num_classes,
        #                                                   num_domains=model.num_domains)
        model.load_state_dict(pretrain_dict, strict=False)

    return model

@BACKBONE_REGISTRY.register()
def resnet152_texture(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = TextureResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
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

def _update_initial_weights_texture(state_dict, num_classes=1000, num_domains=2, dsbn_type='all'):
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
class TextureResNet(Backbone):
    def __init__(self, block, layers, in_features=256, num_classes=1000, **kwargs):
        self.inplanes = 64
        self.in_features = in_features
        self.num_classes = num_classes
        super(TextureResNet, self).__init__()
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


        self.transform_layer = TextureTransformBlock(**kwargs)
        # self.transform_layer = SVDTextureTransformBlock(**kwargs)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Conv2d(self.inplanes, planes * block.expansion,
                       kernel_size=1, stride=stride, bias=False),
                # ModifiedBatchNorm2d(planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(
                block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def featuremaps(self, x, cur_epoch_ratio=0):
        # x_org = x.detach()

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # self.transform_layer.training=True
        # if self.transform_layer.training:
        #     x_tr, loss_sim = self.transform_layer(x)

        # self.transform_layer.visualize_HH(x_org, self.transform_layer.HH)
        # import pdb; pdb.set_trace()

        x = self.layer1(x)

        x = self.layer2(x)

        if self.transform_layer.training:
            x_tr = self.transform_layer(x, cur_epoch_ratio)
            self.feat_org = x
            self.feat_tr = x_tr

        x = self.layer3(x)

        x = self.layer4(x)


        if self.transform_layer.training:

            # x_tr = self.layer1(x_tr)
            #
            # x_tr = self.layer2(x_tr)

            # x_tr, loss_sim = self.transform_layer(x_tr)

            x_tr = self.layer3(x_tr)
            x_tr = self.layer4(x_tr)

        else:
            x_tr = x

        return x, x_tr

    def forward_from_layer(self, x, start_layer=0):

        if start_layer <= 0:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

        if start_layer <= 1:
            x = self.layer1(x)
        if start_layer <= 2:
            x = self.layer2(x)
        if start_layer <= 3:
            x = self.layer3(x)
        if start_layer <= 4:
            x = self.layer4(x)

        x = x.mean(3).mean(2)  # global average pooling
        x = x.view(x.size(0), -1)

        return x

    def forward(self, x, cur_epoch_ratio=0):

        x, x_tr = self.featuremaps(x, cur_epoch_ratio=cur_epoch_ratio)

        x = x.mean(3).mean(2)  # global average pooling
        x = x.view(x.size(0), -1)

        x_tr = x_tr.mean(3).mean(2)  # global average pooling
        x_tr = x_tr.view(x_tr.size(0), -1)

        # if self.in_features != 0:
        #     x = self.fc1(x)
        #     feat = x
        #     x = self.fc2(x)
        # else:
        #     x = self.fc(x)
        #     feat = x

        return x, x_tr