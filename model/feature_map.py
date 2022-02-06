from torch import nn


class FeatureMapConvolution(nn.Module):
    def __init__(self):
        super(FeatureMapConvolution, self).__init__()

        self.conv1 = Conv2DBatchNormRelu(
            in_channels=3, out_channels=64, kernel_size=3, stride=2, padding=1, dilation=1, bias=False
        )

        self.conv2 = Conv2DBatchNormRelu(
            in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
        )

        self.conv3 = Conv2DBatchNormRelu(
            in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
        )

        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        outputs = self.max_pool(x)

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
