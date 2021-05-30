import torch
from torch import Tensor
import torch.nn as nn
import torchvision.models



def conv_block(in_channels, out_channels, kernel, padding):

    return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias = False),
            nn.InstanceNorm2d(out_channels),
            nn.ELU(inplace=False),
        )



class convrelu(nn.Module):

    def __init__(self, in_channels, out_channels, kernel, num_layers = 2):
        super().__init__()

        self.conv_layers = self.create_block(in_channels, out_channels, kernel, num_layers)

        # Skip connection
        if num_layers == 1:
            self.skip = False
        else:
            self.identity_block = self.create_block(in_channels, out_channels, 1, 1)
            self.skip = True


    def create_block(self, in_channels, out_channels, kernel, num_layers):

        layers = []
        mid = num_layers // 2
        last_out_layer = in_channels
        padding = kernel // 2
        for i in range(num_layers):

            if i != mid:
                layers.append(conv_block(last_out_layer, last_out_layer, kernel, padding))
            else:
                layers.append(conv_block(last_out_layer, out_channels, kernel, padding))
                last_out_layer = out_channels

        return nn.Sequential(*layers)

    def forward(self, x):

        out = self.conv_layers(x)

        if self.skip:
            out = out + self.identity_block(x)

        return out


class ResNetUNet34(nn.Module):

    def __init__(self, pretrained = True):
        super().__init__()

        if pretrained:
            self.base_model = torchvision.models.resnet34(pretrained=pretrained)
        else:
            self.base_model = torchvision.models.resnet34(pretrained=pretrained, norm_layer = nn.InstanceNorm2d)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 1)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 1)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 1)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 1)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 1)
        self.layer5 = convrelu(512, 1024, 3, 2)


        self.conv_up5 = convrelu(1024, 512, 3, 2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = convrelu(512, 256, 3, 3)
        self.conv_up3 = convrelu(256, 128, 3, 6)
        self.conv_up2 = convrelu(128, 64, 3, 4)
        self.conv_up1 = convrelu(64, 64, 3, 3)
        # self.conv_up0 = convrelu(64, 64, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        # self.conv_original_size2 = convrelu(64, 64, 3, 1, 1)

        self.conv_mask = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2), nn.Conv2d(16, 1, 5, padding = 2))

        #self.conv_mask = convrelu(64, 1, 3, 3)
        #self.conv_depth = convrelu(64, 1, 3, 3)

        self.conv_depth = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2),nn.Conv2d(16, 1, 5, padding = 2))

        self.sig = nn.Sigmoid()

    # def make_layers(self, num_blocks = None):



    def forward(self, input, encoder = False):

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        #print(layer1.size())
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        #print(layer3.size())
        layer4 = self.layer4(layer3)
        layer5 = self.layer5(layer4)
        layer5_up = self.conv_up5(layer5)
        layer4 = self.layer4_1x1(layer4)

        x = self.upsample(layer4)
        #print('encoded')
        x = self.conv_up4(x)

        layer3 = self.layer3_1x1(layer3)
        #print(x.size(), layer3.size())
        x = x + layer3
        #x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        #print('up1')
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        #print(x.size(), layer2.size())
        x = x + layer2
        #x = torch.cat([x, layer2], dim=1)

        x = self.conv_up2(x)

        x = self.upsample(x)

        layer1 = self.layer1_1x1(layer1)
        #print(x.size(), layer1.size())
        x = x + layer1
        #x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        #print(x.size(), layer0.size())
        x = x + layer0
        #x = torch.cat([x, layer0], dim=1)

        x = self.upsample(x)
        x = x + x_original
        # x = self.conv_original_size2(x)

        mask = self.sig(self.conv_mask(x))
        depth = self.sig(self.conv_depth(x))

        if encoder:
            return depth, mask, layer5
        else:
            return depth, mask


class ResNetUNet50(nn.Module):

    def __init__(self, pretrained = True):
        super().__init__()

        if pretrained:
            self.base_model = torchvision.models.resnet50(pretrained=pretrained)
        else:
            self.base_model = torchvision.models.resnet50(pretrained=pretrained, norm_layer = nn.InstanceNorm2d)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 1)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 1)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 256, 1, 1)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 512, 1, 1)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        #self.layer4_1x1 = convrelu(2048, 512, 1, 0)
        self.layer4_1x1 = convrelu(2048, 2048, 1, 1)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = convrelu(2048, 512, 3, 3)
        self.conv_up3 = convrelu(512, 256, 3, 6)
        self.conv_up2 = convrelu(256, 256, 3, 4)
        self.conv_up1 = convrelu(256, 64, 3, 3)
        # self.conv_up0 = convrelu(64, 64, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        # self.conv_original_size2 = convrelu(64, 64, 3, 1, 1)

        #self.conv_mask = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2),nn.Conv2d(16, 1, 5, padding = 2))

        self.conv_mask = convrelu(64, 64, 3, 3)
        self.conv_depth = convrelu(64, 64, 3, 3)

        #self.conv_depth = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2),nn.Conv2d(16, 1, 5, padding = 2))

        self.sig = nn.Sigmoid()

    # def make_layers(self, num_blocks = None):



    def forward(self, input, encoder = False):

        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        #print(layer1.size())
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        #print(layer3.size())
        layer4 = self.layer4(layer3)
        layer4 = self.layer4_1x1(layer4)

        x = self.upsample(layer4)
        #print('encoded')
        x = self.conv_up4(x)

        layer3 = self.layer3_1x1(layer3)
        #print(x.size(), layer3.size())
        x = x + layer3
        #x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        #print('up1')
        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        #print(x.size(), layer2.size())
        x = x + layer2
        #x = torch.cat([x, layer2], dim=1)

        x = self.conv_up2(x)

        x = self.upsample(x)

        layer1 = self.layer1_1x1(layer1)
        #print(x.size(), layer1.size())
        x = x + layer1
        #x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        #print(x.size(), layer0.size())
        x = x + layer0
        #x = torch.cat([x, layer0], dim=1)

        x = self.upsample(x)
        x = x + x_original
        # x = self.conv_original_size2(x)

        mask = self.sig(self.conv_mask(x))
        depth = self.sig(self.conv_depth(x))

        if encoder:
            return depth, mask, layer4
        else:
            return depth, mask


class NOCS_decoder50(nn.Module):
    '''
    NOCS decoder class
    '''
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = convrelu(2048, 512, 3, 3)
        self.conv_up3 = convrelu(512, 256, 3, 6)
        self.conv_up2 = convrelu(256, 128, 3, 4)
        self.conv_up1 = convrelu(128, 64, 3, 3)
        # self.conv_up0 = convrelu(128,  64, 3, 3)
        self.conv_original_size2 = convrelu(64, 3, 3, 1)

        self.sig = nn.Sigmoid()

    def forward(self, input):

        #print(x.shape)
        x = self.upsample(input)

        x = self.conv_up4(x)
        x = self.upsample(x)

        x = self.conv_up3(x)
        x = self.upsample(x)

        x = self.conv_up2(x)
        x = self.upsample(x)

        x = self.conv_up1(x)
        x = self.upsample(x)

        x = self.sig(self.conv_original_size2(x))

        return x

class NOCS_decoder(nn.Module):
    '''
    NOCS decoder class
    '''
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up5 = convrelu(1024, 512, 3, 2)
        self.conv_up4 = convrelu(512, 512, 3, 3)
        self.conv_up3 = convrelu(512, 256, 3, 6)
        self.conv_up2 = convrelu(256, 128, 3, 4)
        self.conv_up1 = convrelu(128, 64, 3, 3)
        # self.conv_up0 = convrelu(128,  64, 3, 3)
        self.conv_original_size2 = convrelu(64, 3, 3, 3)

        self.sig = nn.Sigmoid()

    def forward(self, input):

        #print(x.shape)

        x = self.conv_up5(input)
        x = self.upsample(x)

        x = self.conv_up4(x)
        x = self.upsample(x)

        x = self.conv_up3(x)
        x = self.upsample(x)

        x = self.conv_up2(x)
        x = self.upsample(x)

        x = self.conv_up1(x)
        x = self.upsample(x)

        x = self.sig(self.conv_original_size2(x))

        return x

