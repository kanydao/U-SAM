import torch.nn as nn
import torch


def get_activation(activation_type):
    activation_type = activation_type.lower()
    if hasattr(nn, activation_type):
        return getattr(nn, activation_type)()
    else:
        return nn.ReLU()


def _make_nConv(in_channels, out_channels, nb_Conv, activation='ReLU', stride=1):
    layers = []
    layers.append(ConvBatchNorm(in_channels, out_channels, activation, stride))

    for _ in range(nb_Conv - 1):
        layers.append(ConvBatchNorm(out_channels, out_channels, activation))
    return nn.Sequential(*layers)


class ConvBatchNorm(nn.Module):
    """(convolution => [BN] => ReLU)"""

    def __init__(self, in_channels, out_channels, activation='ReLU', stride=1):
        super(ConvBatchNorm, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=3, padding=1, stride=stride)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = get_activation(activation)

    def forward(self, x):
        # return self.conv(x)
        out = self.conv(x)
        out = self.norm(out)
        return self.activation(out)


class DownBlock(nn.Module):
    """Downscaling with maxpool convolution"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU'):
        super(DownBlock, self).__init__()
        self.maxpool = nn.MaxPool2d(2)
        self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation, stride=1)

    def forward(self, x):
        out = self.maxpool(x)
        return self.nConvs(out)


class UpBlock(nn.Module):
    """Upscaling then conv"""

    def __init__(self, in_channels, out_channels, nb_Conv, activation='ReLU', use_prompt=False, use_skip=False):
        super(UpBlock, self).__init__()

        # self.up = nn.Upsample(scale_factor=2)
        self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, (2, 2), 2)

        self.use_skip = use_skip
        if use_skip:
            self.nConvs = _make_nConv(in_channels, out_channels, nb_Conv, activation)
        else:
            self.nConvs = _make_nConv(in_channels // 2, out_channels, nb_Conv, activation)

        self.use_prompt = use_prompt
        if self.use_prompt:
            self.compress = ChannelPool()
            kernel_size = 7
            self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x, skip_x=None):
        x = self.up(x)

        if self.use_skip:
            if self.use_prompt:
                compress = self.compress(x)
                scale = torch.sigmoid(self.spatial(compress))
                x = torch.cat([x, skip_x * scale], dim=1)  # dim 1 is the channel dimension
            else:
                x = torch.cat([x, skip_x], dim=1)  # dim 1 is the channel dimension

        return self.nConvs(x)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat([torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)], dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class UNet(nn.Module):
    def __init__(self, n_channels=3):
        '''
        n_channels : number of channels of the input.
                        By default 3, because we have RGB images
        n_labels : number of channels of the ouput.
                      By default 3 (2 labels + 1 for the background)
        '''
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = 1
        # Question here
        in_channels = 32
        bt_channels = 768
        self.inc = ConvBatchNorm(n_channels, in_channels)  # 32
        self.down1 = DownBlock(in_channels, in_channels * 2, nb_Conv=2)  # 64
        self.down2 = DownBlock(in_channels * 2, in_channels * 4, nb_Conv=2)  # 128
        self.down3 = DownBlock(in_channels * 4, in_channels * 8, nb_Conv=2)  # 256
        self.down4 = DownBlock(in_channels * 8, bt_channels, nb_Conv=2)  # 768

        # self.up4 = UpBlock(in_channels*16, in_channels*4, nb_Conv=2)
        # self.up3 = UpBlock(in_channels*8, in_channels*2, nb_Conv=2)
        # self.up2 = UpBlock(in_channels*4, in_channels, nb_Conv=2)
        # self.up1 = UpBlock(in_channels*2, in_channels, nb_Conv=2)
        # self.outc = nn.Conv2d(in_channels, n_classes, kernel_size=(1,1))
        # if n_classes == 1:
        #     self.last_activation = nn.Sigmoid()
        # else:
        #     self.last_activation = None

    def forward(self, x):
        # Question here
        x = x.float()
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = x5.permute(0, 2, 3, 1)
        # print(x5.shape)
        # exit(0)
        # x = self.up4(x5, x4)
        # x = self.up3(x, x3)
        # x = self.up2(x, x2)
        # x = self.up1(x, x1)
        # if self.last_activation is not None:
        #     logits = self.last_activation(self.outc(x))
        # else:
        #     logits = self.outc(x)

        return x, [x1, x2, x3, x4]
