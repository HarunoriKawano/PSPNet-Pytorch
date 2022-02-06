from torch import nn


class ResNet50(nn.Module):

    def __init__(self):
        super(ResNet50, self).__init__()
        self.feature_block1 = ResidualBlockPSP(
            n_blocks=3, in_channels=128, mid_channels=64,
            out_channels=256, stride=1, dilation=1
        )
        self.feature_block2 = ResidualBlockPSP(
            n_blocks=4, in_channels=256, mid_channels=128,
            out_channels=512, stride=2, dilation=1
        )
        self.feature_block3 = ResidualBlockPSP(
            n_blocks=6, in_channels=512, mid_channels=256,
            out_channels=1024, stride=1, dilation=2
        )
        self.feature_block4 = ResidualBlockPSP(
            n_blocks=3, in_channels=1024, mid_channels=512,
            out_channels=2048, stride=1, dilation=4
        )

    def forward(self, x):
        feature = self.feature_block1(x)
        feature = self.feature_block2(feature)
        feature = self.feature_block3(feature)
        feature = self.feature_block4(feature)

        return feature


class ResidualBlockPSP(nn.Sequential):
    def __init__(self, n_blocks, in_channels, mid_channels, out_channels, stride, dilation):
        super(ResidualBlockPSP, self).__init__()

        # BottleNek
        self.add_module(
            "block1",
            BottleNeck(
                in_channels, mid_channels, out_channels, stride, dilation
            )
        )

        # loop BottleNekIdentify
        for i in range(n_blocks - 1):
            self.add_module(
                "block" + str(i + 2),
                BottleNeckIdentify(
                    out_channels, mid_channels, dilation
                )
            )


class BottleNeck(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels, stride, dilation):
        super(BottleNeck, self).__init__()

        self.conv1 = Conv2DBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )
        self.conv2 = Conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, bias=False
        )
        self.conv3 = Conv2DBatchNorm(
            mid_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )

        # skip union
        self.residual_conv = Conv2DBatchNorm(
            in_channels, out_channels, kernel_size=1, stride=stride, padding=0, dilation=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.conv2(conv)
        conv = self.conv3(conv)

        residual = self.residual_conv(x)

        return self.relu(conv + residual)


class BottleNeckIdentify(nn.Module):
    def __init__(self, in_channels, mid_channels, dilation):
        super(BottleNeckIdentify, self).__init__()

        self.cbr_1 = Conv2DBatchNormRelu(
            in_channels, mid_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )

        self.cbr_2 = Conv2DBatchNormRelu(
            mid_channels, mid_channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation, bias=False
        )

        self.cb_3 = Conv2DBatchNorm(
            mid_channels, in_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv = self.conv1(x)
        conv = self.conv2(conv)
        conv = self.conv3(conv)

        return self.relu(conv + x)


class Conv2DBatchNorm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNorm, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        outputs = self.batch_norm(x)

        return outputs


class Conv2DBatchNormRelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, dilation, bias):
        super(Conv2DBatchNormRelu, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride, padding, dilation, bias=bias
        )
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        outputs = self.relu(x)

        return outputs
