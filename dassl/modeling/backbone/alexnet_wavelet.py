import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from .build import BACKBONE_REGISTRY
from .backbone import Backbone

from .resnet_wavelet import TextureTransformBlock

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
}


class AlexNet_texture(Backbone):

    def __init__(self, **kwargs):
        super().__init__()
        self.features = nn.ModuleDict({
            'conv1': nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            'relu1':nn.ReLU(inplace=True),
            'pool1':nn.MaxPool2d(kernel_size=3, stride=2),
            'conv2':nn.Conv2d(64, 192, kernel_size=5, padding=2),
            'relu2':nn.ReLU(inplace=True),
            'pool2':nn.MaxPool2d(kernel_size=3, stride=2),
            'conv3':nn.Conv2d(192, 384, kernel_size=3, padding=1),
            'relu3':nn.ReLU(inplace=True),
            'conv4':nn.Conv2d(384, 256, kernel_size=3, padding=1),
            'relu4':nn.ReLU(inplace=True),
            'conv5':nn.Conv2d(256, 256, kernel_size=3, padding=1),
            'relu5':nn.ReLU(inplace=True),
            'pool5':nn.MaxPool2d(kernel_size=3, stride=2),
            }
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        # Note that self.classifier outputs features rather than logits
        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * 6 * 6, 4096), nn.ReLU(inplace=True),
            nn.Dropout(), nn.Linear(4096, 4096), nn.ReLU(inplace=True)
        )

        self._out_features = 4096
        self.transform_block = TextureTransformBlock(**kwargs)

    def featuremaps(self, x, cur_epoch_ratio=0):
        after_transform = False
        x_tr = x.detach()
        for name, module in self.features.items():
            x = module(x)
            if after_transform:
                x_tr = module(x_tr)

            if name == 'pool2':
                x_tr = self.transform_block(x, cur_epoch_ratio=cur_epoch_ratio)
                after_transform = True

        return x, x_tr

    def forward(self, x, cur_epoch_ratio=0):
        import pdb; pdb.set_trace()
        feat, feat_tr = self.featuremaps(x, cur_epoch_ratio=0)
        feat = self.avgpool(feat)
        feat = torch.flatten(feat, 1)
        feat = self.classifier(feat)

        feat_tr = self.avgpool(feat_tr)
        feat_tr = torch.flatten(feat_tr, 1)
        feat_tr = self.classifier(feat_tr)

        return feat, feat_tr


def init_pretrained_weights(model, model_url):
    pretrain_dict = model_zoo.load_url(model_url)
    model.load_state_dict(pretrain_dict, strict=False)


@BACKBONE_REGISTRY.register()
def alexnet_texture(pretrained=True, **kwargs):
    model = AlexNet_texture(**kwargs)

    if pretrained:
        init_pretrained_weights(model, model_urls['alexnet'])

    return model
