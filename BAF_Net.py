import torch
from torch import nn
import torch.nn.functional as F
import math



class depthwise_separable_conv(nn.Module):
    def __init__(self, in_channel, out_channel, stride=(1,1,1)):
        super(depthwise_separable_conv, self).__init__()

        self.depthwise = nn.Conv3d(in_channel, in_channel, kernel_size=3, padding=1, groups=in_channel, stride=stride)
        self.pointwise = nn.Conv3d(in_channel, out_channel, kernel_size=1)

    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        return out



class Scaled_SELayer(nn.Module):
    """
    Change 2D SELayer to 3D SELayer

    Add a scaled shortcut connection
    """
    def __init__(self, channel, reduction=8):
        super(Scaled_SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.max_pool = nn.AdaptiveMaxPool3d((1, 1, 1))
        self.fc_avg = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )
        self.fc_max = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        identity = x

        b, c, _, _, _ = x.size()
        y_avg = self.avg_pool(x).view(b, c)
        y_avg = self.fc_avg(y_avg).view(b, c, 1, 1, 1)

        y_max = self.max_pool(x).view(b, c)
        y_max = self.fc_max(y_max).view(b, c, 1, 1, 1)

        y = y_max + y_avg

        scaled_map = x * y.expand_as(x) + identity

        return scaled_map



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



class BAF_Net(nn.Module):
    def __init__(self, in_channel=2):
        super(BAF_Net, self).__init__()

        # pet stream
        self.pet_block1 = nn.Sequential(
            nn.Conv3d(in_channel, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        )

        self.pet_block2 = nn.Sequential(
            BasicBlock(64),
            DownsampleBlock(64, 128)
        )

        self.pet_block3 = nn.Sequential(
            BasicBlock(128),
            DownsampleBlock(128, 256)
        )

        self.avgpool_pet = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.output_head_pet = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )


        # ct stream
        self.ct_block1 = nn.Sequential(
            nn.Conv3d(in_channel, 32, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 64, kernel_size=(1, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1), bias=True),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=(1, 2, 2), padding=1)
        )

        self.ct_block2 = nn.Sequential(
            BasicBlock(64),
            DownsampleBlock(64, 128)
        )

        self.ct_block3 = nn.Sequential(
            BasicBlock(128),
            DownsampleBlock(128, 256)
        )

        self.avgpool_ct = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.output_head_ct = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
        )

        # Conv Block for pet/ct information interaction
        self.Bridge1 = depthwise_separable_conv(64 * 2, 32)
        self.se1 = Scaled_SELayer(32)

        self.Bridge2 = depthwise_separable_conv(128 * 2 + 32, 64)
        self.se2 = Scaled_SELayer(64)

        self.Bridge3 = depthwise_separable_conv(256 * 2 + 64, 128)
        self.se3 = Scaled_SELayer(128)

        # Transition layer for fuse feature better fuse
        self.Transition1 = nn.Conv3d(128 + 64, 64, kernel_size=1)
        self.se4 = Scaled_SELayer(64)

        self.Transition2 = nn.Conv3d(128 + 64 + 32, 32, kernel_size=1)
        self.se5 = Scaled_SELayer(32)

        self.Transition3 = nn.Conv3d(64 + 32, 16, kernel_size=1)

        # avgPool
        self.avgpool_fuse = nn.AdaptiveAvgPool3d((1, 1, 1))

        # fuse feature output head
        self.output_head_fuse = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1)
        )

        self.pet_weight = nn.Parameter(torch.ones(1,))
        self.ct_weight = nn.Parameter(torch.ones(1,))
        self.fuse_weight = nn.Parameter(torch.ones(1,))

        # init weight
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.xavier_normal_(m.weight.data, gain=1)
                nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight.data, 1)
                nn.init.constant_(m.bias.data, 0)


    def forward(self, pet, ct):

        z, y, x = pet.shape[2], pet.shape[3], pet.shape[4]
        z_2x, y_2x, x_2x = math.ceil(z / 1), math.ceil(y / 2), math.ceil(x / 2)
        z_4x, y_4x, x_4x = math.ceil(z / 2), math.ceil(y / 4), math.ceil(x / 4)
        z_8x, y_8x, x_8x = math.ceil(z / 4), math.ceil(y / 8), math.ceil(x / 8)
        pet_stage1_features = self.pet_block1(pet)   #(1,64,16,80,80)
        ct_stage1_features = self.ct_block1(ct)

        fuse_stage_1_map1 = torch.cat((pet_stage1_features, ct_stage1_features), dim=1)
        fuse_stage_1_map1[fuse_stage_1_map1.isnan()] = 0.
        fuse_stage_1_map1[fuse_stage_1_map1.isinf()] = 1.
        fuse_stage_1_map1 = self.Bridge1(fuse_stage_1_map1)    #(1,32,16,80,80)
        fuse_stage_1_map1 = self.se1(fuse_stage_1_map1)

        fuse_stage_1_map1_BUTD = F.interpolate(fuse_stage_1_map1, size=(z_4x, y_4x, x_4x), mode='trilinear', align_corners=True)   #(1,32,8,40,40)

        pet_stage2_features = self.pet_block2(pet_stage1_features)   #(1,128,8,40,40)
        ct_stage2_features = self.ct_block2(ct_stage1_features)

        fuse_stage_2_map1 = torch.cat((pet_stage2_features, ct_stage2_features, fuse_stage_1_map1_BUTD), dim=1)
        fuse_stage_2_map1[fuse_stage_2_map1.isnan()] = 0.
        fuse_stage_2_map1[fuse_stage_2_map1.isinf()] = 1.
        fuse_stage_2_map1 = self.Bridge2(fuse_stage_2_map1)   #(1,64,24,16,16)
        fuse_stage_2_map1 = self.se2(fuse_stage_2_map1)

        fuse_stage_2_map1_BUTD = F.interpolate(fuse_stage_2_map1, size=(z_8x, y_8x, x_8x), mode='trilinear', align_corners=True)

        pet_stage3_features = self.pet_block3(pet_stage2_features)    #(1,128,12,8,8)
        ct_stage3_features = self.ct_block3(ct_stage2_features)

        fuse_stage_3_map1 = torch.cat((pet_stage3_features, ct_stage3_features, fuse_stage_2_map1_BUTD), dim=1)
        fuse_stage_3_map1[fuse_stage_3_map1.isnan()] = 0.
        fuse_stage_3_map1[fuse_stage_3_map1.isinf()] = 1.
        fuse_stage_3_map1 = self.Bridge3(fuse_stage_3_map1)    #(1,128,12,8,8)
        fuse_stage_3_map1 = self.se3(fuse_stage_3_map1)

        # Implement the fluid connection of fuse features
        # map1 donates fuse_stage_1, map2 donates fuse_stage_2, map3 donates fuse_stage_3
        map3_for_map2 = F.interpolate(fuse_stage_3_map1, size=(z_4x, y_4x, x_4x), mode='trilinear', align_corners=True)
        fuse_stage_2_map2 = torch.cat((map3_for_map2, fuse_stage_2_map1), dim=1)    #(1,192,24,16,16)
        fuse_stage_2_map2[fuse_stage_2_map2.isnan()] = 0.
        fuse_stage_2_map2[fuse_stage_2_map2.isinf()] = 1.
        fuse_stage_2_map2 = self.Transition1(fuse_stage_2_map2)     #(1,64,24,16,16)
        fuse_stage_2_map2 = self.se4(fuse_stage_2_map2)

        map3_for_map1 = F.interpolate(fuse_stage_3_map1, size=(z_2x, y_2x, x_2x), mode='trilinear', align_corners=True)
        map2_for_map1 = F.interpolate(fuse_stage_2_map1, size=(z_2x, y_2x, x_2x), mode='trilinear', align_corners=True)
        fuse_stage_1_map2 = torch.cat((map3_for_map1, map2_for_map1, fuse_stage_1_map1), dim=1)      #(1,224,48,32,32)
        fuse_stage_1_map2[fuse_stage_1_map2.isnan()] = 0.
        fuse_stage_1_map2[fuse_stage_1_map2.isinf()] = 1.
        fuse_stage_1_map2 = self.Transition2(fuse_stage_1_map2)   #(1,32,48,32,32)
        fuse_stage_1_map2 = self.se5(fuse_stage_1_map2)

        fuse_stage_2_map2 = F.interpolate(fuse_stage_2_map2, size=(z_2x, y_2x, x_2x), mode='trilinear', align_corners=True)
        fuse_stage_1_map3 = torch.cat((fuse_stage_2_map2, fuse_stage_1_map2), dim=1)   #(1,96,48,32,32)
        fuse_stage_1_map3[fuse_stage_1_map3.isnan()] = 0.
        fuse_stage_1_map3[fuse_stage_1_map3.isinf()] = 1.
        fuse_stage_1_map3 = self.Transition3(fuse_stage_1_map3) #(1,16,48,32,32)

        fuse_features = self.avgpool_fuse(fuse_stage_1_map3)
        fuse_features = fuse_features.view(fuse_features.size(0), -1)
        fuse_features[fuse_features.isnan()] = 0.
        fuse_features[fuse_features.isinf()] = 1.
        fuse_output = self.output_head_fuse(fuse_features)
        fuse_output[fuse_output.isnan()] = 0.
        fuse_output[fuse_output.isinf()] = 1.

        pet_features = self.avgpool_pet(pet_stage3_features)
        pet_features = pet_features.view(pet_features.size(0), -1)
        pet_features[pet_features.isnan()] = 0.
        pet_features[pet_features.isinf()] = 1.
        pet_output = self.output_head_pet(pet_features)
        pet_output[pet_output.isnan()] = 0.
        pet_output[pet_output.isinf()] = 1.

        ct_features = self.avgpool_ct(ct_stage3_features)
        ct_features = ct_features.view(ct_features.size(0), -1)
        ct_features[ct_features.isnan()] = 0.
        ct_features[ct_features.isinf()] = 1.
        ct_output = self.output_head_ct(ct_features)
        ct_output[ct_output.isnan()] = 0.
        ct_output[ct_output.isinf()] = 1.

        output = (fuse_output * self.fuse_weight + pet_output * self.pet_weight + ct_output * self.ct_weight) / 3

        return output






if __name__ == '__main__':
    pet = torch.randn((1,1,64,96,96))
    ct = torch.randn((1,1,64,96,96))

    model = BAF_Net(in_channel=2)
    output = model(pet, ct)
    print(output.shape)

    #network parameters
    print('net total parameters:', sum(param.numel() for param in model.parameters()) / 1e6)



