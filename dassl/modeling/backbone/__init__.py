from .build import build_backbone, BACKBONE_REGISTRY # isort:skip
from .backbone import Backbone # isort:skip

from .vgg import vgg16
from .resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from .alexnet import alexnet
from .mobilenetv2 import mobilenetv2
from .wide_resnet import wide_resnet_28_2
from .cnn_digitsdg import cnn_digitsdg
from .efficientnet import (
    efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
    efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
)
from .shufflenetv2 import (
    shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5,
    shufflenet_v2_x2_0
)
from .cnn_digitsingle import cnn_digitsingle
from .preact_resnet18 import preact_resnet18
from .cnn_digit5_m3sda import cnn_digit5_m3sda
from .resnet_wavelet import resnet18_texture, resnet34_texture, resnet50_texture, resnet101_texture, resnet152_texture
from .resnet_infodrop import resnet18_InfoDrop, resnet34_InfoDrop, resnet50_InfoDrop
from .alexnet_wavelet import alexnet_texture