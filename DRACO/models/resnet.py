import torch
import torch.nn as nn
from torchvision import models
import kornia.filters
# borrowed from https://github.com/usuyama/pytorch-unet/blob/master/pytorch_resnet18_unet.ipynb and edited


def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(

        nn.Conv2d(in_channels, in_channels, kernel, padding=padding, bias = False),
        nn.ELU(inplace=True),
        nn.InstanceNorm2d(in_channels),


       # nn.Conv2d(in_channels, in_channels, kernel, padding=padding, bias = False),
       # nn.ELU(inplace=True),
       # nn.InstanceNorm2d(in_channels),

        nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias = False),
        nn.ELU(inplace=True),
        nn.InstanceNorm2d(out_channels),


    )

class ResNetUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.base_model = models.resnet18(pretrained=True)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last_mask = nn.Conv2d(64, 1, 3, 1)
        self.conv_last_depth = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2), nn.Conv2d(16, 1, 5, padding = 2), nn.AvgPool2d(5, padding = 2))

        self.sig = nn.Sigmoid()

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out_mask = self.sig(self.conv_last_mask(x))
        out_depth = self.sig(self.conv_last_depth(x))
        print(out_depth.size())
        return out_depth, out_mask


class ResNetUNet50(nn.Module):

    def __init__(self, pretrained = True):
        super().__init__()

        if pretrained:
            self.base_model = models.resnet50(pretrained=pretrained)
        else:
            self.base_model = models.resnet50(pretrained=pretrained, norm_layer = nn.InstanceNorm2d)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 256, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 512, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        #self.layer4_1x1 = convrelu(2048, 512, 1, 0)
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = convrelu(2048, 512, 3, 1)
        self.conv_up3 = convrelu(512, 256, 3, 1)
        self.conv_up2 = convrelu(256, 256, 3, 1)
        self.conv_up1 = convrelu(256, 64, 3, 1)
        self.conv_up0 = convrelu(64, 64, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64, 64, 3, padding = 1)


        #self.conv_mask = nn.Conv2d(64, 1, 3, padding = 1)

        self.conv_mask = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2), nn.Conv2d(16, 1, 5, padding = 2))

        #self.conv_depth = nn.Conv2d(64, 1, 3, padding = 1)
        self.sig = nn.Sigmoid()
        #self.conv_depth = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2), nn.Conv2d(16, 1, 5, padding = 2), nn.AvgPool2d(3, stride = 1, padding = 1))

        self.conv_depth = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2), nn.Conv2d(16, 1, 5, padding = 2))
        self.median_filter = kornia.filters.median_blur
        self.gaussian_filter = kornia.filters.gaussian_blur2d

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
        x = self.conv_up0(x)


        x = self.upsample(x)
        x = x + x_original
        x = self.conv_original_size2(x)

        mask = self.sig(self.conv_mask(x))
        # SLURM 2
        #depth = self.sig(self.median_filter(self.conv_depth(x), (3,3)))
        #depth = self.sig(self.conv_depth(x))
        #1
        #depth = self.sig(self.gaussian_filter(self.conv_depth(x), (3,3), (0.9,0.9)))
        #################FINAL CHANGE
        depth = self.sig(self.conv_depth(x))

        if encoder:
            return depth, mask, layer4
        else:
            return depth, mask


class convrelu_nocs(nn.Module):
    '''
    Convolution layer for NOCS decoder
    Conv layer with skip connections
    '''

    def __init__(self, in_channels, out_channels, kernel, padding):
        super().__init__()

        self.elu = nn.ELU(inplace=True)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel, padding=padding, bias = False)
        self.norm_1 = nn.InstanceNorm2d(in_channels)

        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel, padding=padding, bias = False)
        self.norm_2 = nn.InstanceNorm2d(out_channels)

        self.conv_3 = nn.Conv2d(out_channels, out_channels, kernel, padding=padding, bias = False)
        self.norm_3 = nn.InstanceNorm2d(out_channels)

        self.conv_identity = nn.Conv2d(in_channels, out_channels, 1, padding=padding, bias = False)
        self.norm_identity = nn.InstanceNorm2d(out_channels)

    def forward(self, input):

        x_original = input
        #print(input.shape)
        x = self.conv_1(input)
        x = self.elu(x)
        x = self.norm_1(x)


        x = self.conv_2(x)
        x = self.elu(x)
        x = self.norm_2(x)


        x = self.conv_3(x)
        x = self.elu(x)
        x = self.norm_3(x)

        # Check if the number of channels are same
        if x_original.shape[1] != x.shape[1]:
            x_original = self.norm_identity(self.elu(self.conv_identity(x_original)))

        # Making sure the size is same
        x_original = nn.functional.interpolate(x_original, size = (x.shape[2], x.shape[3]))

        x = x + x_original

        return x

class NOCS_decoder(nn.Module):
    '''
    NOCS decoder class
    '''
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_up4 = convrelu_nocs(2048, 512, 3, 1)
        self.conv_up3 = convrelu_nocs(512, 256, 3, 1)
        self.conv_up2 = convrelu_nocs(256, 256, 3, 1)
        self.conv_up1 = convrelu_nocs(256, 64, 3, 1)
        self.conv_up0 = convrelu_nocs(64, 64, 3, 1)

        #self.conv_original_size0 = convrelu_nocs(3, 64, 3, 1)
        #self.conv_original_size1 = convrelu_nocs(64, 64, 3, 1)
        self.conv_original_size2 = convrelu_nocs(64, 3, 3, padding = 1)

        self.sig = nn.Sigmoid()

    def forward(self, input):

        x = self.upsample(input)
        #print(x.shape)
        x = self.conv_up4(x)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = self.sig(self.conv_original_size2(x))

        return x


class NOCS_decoder_2(nn.Module):
    '''
    NOCS decoder class
    '''
    def __init__(self):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_enc = convrelu_nocs(2048, 1024, 3, 1)
        self.conv_up4 = convrelu_nocs(1024, 512, 3, 1)
        self.conv_up3 = convrelu_nocs(512, 256, 3, 1)
        self.conv_up2 = convrelu_nocs(256, 256, 3, 1)
        self.conv_up1 = convrelu_nocs(256, 64, 3, 1)
        self.conv_up0 = convrelu_nocs(64, 64, 3, 1)

        self.conv_original_size2 = convrelu_nocs(64, 3, 3, padding = 1)

        self.sig = nn.Sigmoid()

    def forward(self, input):

        x = self.conv_enc(input)
        x = self.upsample(x)
        #print(x.shape)
        x = self.conv_up4(x)
        x = self.conv_up3(x)

        x = self.upsample(x)
        x = self.conv_up2(x)

        x = self.upsample(x)
        x = self.conv_up1(x)

        x = self.upsample(x)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = self.sig(self.conv_original_size2(x))

        return x



class ResNet50UNetCat(nn.Module):

    def __init__(self, n_class=2):
        super().__init__()

        self.base_model = models.resnet50(pretrained=True)

        self.base_layers = list(self.base_model.children())

        self.layer0 = nn.Sequential(*self.base_layers[:3]) # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)

        self.layer1 = nn.Sequential(*self.base_layers[3:5]) # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(256, 256, 1, 0)

        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(512, 512, 1, 0)

        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(1024, 1024, 1, 0)

        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(2048, 2048, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.conv_up3 = convrelu(2048 + 1024, 1024, 3, 1)
        self.conv_up2 = convrelu(1024 + 512, 512, 3, 1)
        self.conv_up1 = convrelu(512 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_mask = nn.Conv2d(64, 1, 3, padding = 1)
        self.sig = nn.Sigmoid()
        self.conv_depth = nn.Sequential(nn.Conv2d(64, 32, 5, padding = 2), nn.Conv2d(32, 16, 5, padding = 2), nn.Conv2d(16, 1, 5, padding = 2), nn.AvgPool2d(3, stride = 1, padding = 1))

    def forward(self, input):
        x_original = self.conv_original_size0(input)
        x_original = self.conv_original_size1(x_original)

        layer0 = self.layer0(input)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.layer4_1x1(layer4)
        x = self.upsample(layer4)
        layer3 = self.layer3_1x1(layer3)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        layer2 = self.layer2_1x1(layer2)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        mask = self.sig(self.conv_mask(x))
        depth = self.sig(self.conv_depth(x))

        return depth, mask

class ResNetTruncEncoder(nn.Module):

    def __init__(self, net):
        super().__init__()

        self.base_model = net
        self.base_layers = list(self.base_model.children())
        self.layer0 = nn.Sequential(self.base_layers[1], self.base_layers[3]) # size=(N, 64, x.H/2, x.W/2)

    def forward(self, input):

        layer0 = self.layer0(input)

        return layer0

if __name__=="__main__":

    model = ResNetUNet50(pretrained=True)
    decoder = NOCS_decoder()

    x = torch.randn(2, 3, 480, 640)

    depth, mask, encoded_features = model(x, encoder = True)

    print(encoded_features.shape)
    nocs = decoder(encoded_features)

    print(nocs.shape)
