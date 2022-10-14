import torch
from torch import nn





class BasicBlock(nn.Module):
    def __init__(self, channel):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class DownsampleBlock(nn.Module):
    def __init__(self, channel_in, channel):
        super(DownsampleBlock, self).__init__()
        self.conv1 = nn.Conv3d(channel_in, channel, kernel_size=3, stride=2, padding=1, bias=True)
        self.bn1 = nn.BatchNorm3d(channel)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channel, channel, kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2 = nn.BatchNorm3d(channel)
        self.downsample = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        residual = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += torch.cat((residual, residual), 1)
        out = self.relu(out)

        return out


class ResNet_single(nn.Module):
    def __init__(self, in_channel=1):
        super(ResNet_single, self).__init__()
        self.conv1 = nn.Conv3d(in_channel, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.bn1 = nn.BatchNorm3d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True)
        self.bn2 = nn.BatchNorm3d(64)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)

        self.layer10 = BasicBlock(64)
        self.layer11 = DownsampleBlock(64, 128)

        self.layer20 = BasicBlock(128)
        self.layer21 = DownsampleBlock(128, 256)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.output_head = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)

    def forward(self, img):
        x = self.conv1(img)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.maxpool(x)

        x = self.layer10(x)
        x = self.layer11(x)

        x = self.layer20(x)
        x = self.layer21(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)

        output = self.output_head(x)

        return output


