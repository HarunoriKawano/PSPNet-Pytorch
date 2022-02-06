from torch import nn

from feature_map import FeatureMapConvolution
from resnet50 import ResNet50
from pyramid_pooling import PyramidPooling
from decoder import Decoder


class PSPNet(nn.Module):
    def __init__(self, class_num):
        super(PSPNet, self).__init__()

        image_size = 475
        image_size_pyramid = 60

        self.feature_conv = FeatureMapConvolution()

        self.res_net50 = ResNet50()

        self.pyramid_pool = PyramidPooling(
            in_channels=2048, pool_sizes=[6, 3, 2, 1],
            height=image_size_pyramid, width=image_size_pyramid
        )

        self.decoder = Decoder(
            height=image_size, width=image_size, class_num=class_num
        )

    def forward(self, x):
        feature_map = self.feature_conv(x)
        resnet_features = self.res_net50(feature_map)
        pyramid_features = self.pyramid_pool(resnet_features)
        output = self.decoder(pyramid_features)

        return output
