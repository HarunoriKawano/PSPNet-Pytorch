from torch import nn
from torch.nn import functional as F


class Decoder(nn.Module):

    def __init__(self, height, width, class_num):
        super(Decoder, self).__init__()

        # size of image to use in forward
        self.height = height
        self.width = width

        self.conv = Conv2DBatchNormRelu(
            in_channels=4096, out_channels=512, kernel_size=3, stride=1, padding=1, dilation=1, bias=False
        )
        self.dropout = nn.Dropout2d(p=0.1)
        self.classification = nn.Conv2d(
            in_channels=512, out_channels=class_num, kernel_size=(1, 1), stride=(1, 1), padding=0
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.dropout(x)
        x = self.classification(x)
        output = F.interpolate(
            x, size=(self.height, self.width), mode="bilinear", align_corners=True
        )

        return output


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
