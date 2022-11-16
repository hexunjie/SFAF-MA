import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import sys
import torchvision.models as models
#sys.path.append('../data/')

"""https://github.com/ShreyasSkandanS/PST900_thermal_rgb"""

def convrelu(in_channels, out_channels, kernel, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel, padding=padding),
        nn.ReLU(inplace=True),
    )

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(
            ninput,
            noutput - ninput,
            (3, 3),
            stride=2,
            padding=1,
            bias=True
        )
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat(
            [self.conv(input), self.pool(input)],
            1
        )
        output = self.bn(output)
        return F.relu(output)


class non_bottleneck_1d(nn.Module):
    def __init__(self, chann, dropprob, dilated):
        super().__init__()
        self.conv3x1_1 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1, 0),
            bias=True
        )
        self.conv1x3_1 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1),
            bias=True
        )
        self.bn1 = nn.BatchNorm2d(chann, eps=1e-03)
        self.conv3x1_2 = nn.Conv2d(
            chann,
            chann,
            (3, 1),
            stride=1,
            padding=(1 * dilated, 0),
            bias=True,
            dilation=(dilated, 1)
        )
        self.conv1x3_2 = nn.Conv2d(
            chann,
            chann,
            (1, 3),
            stride=1,
            padding=(0, 1 * dilated),
            bias=True,
            dilation=(1, dilated)
        )
        self.bn2 = nn.BatchNorm2d(chann, eps=1e-03)
        self.dropout = nn.Dropout2d(dropprob)

    def forward(self, input):
        output = self.conv3x1_1(input)
        output = F.relu(output)
        output = self.conv1x3_1(output)
        output = self.bn1(output)
        output = F.relu(output)

        output = self.conv3x1_2(output)
        output = F.relu(output)
        output = self.conv1x3_2(output)
        output = self.bn2(output)

        if (self.dropout.p != 0):
            output = self.dropout(output)

        return F.relu(output + input)  # +input = identity (residual connection)


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(ninput, noutput, 3, 1, 1)
        )

        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class RGBNet(nn.Module):
    def __init__(self, n_channels, n_class):
        super().__init__()

        self.base_model = models.resnet18(pretrained=False)
        self.base_layers = list(self.base_model.children())
        # print(self.base_layers[0])
        # self.base_layers[0] = nn.Conv2d(4,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
        # print(self.base_layers[0])
        self.layer0 = nn.Sequential(*self.base_layers[:3])  # size=(N, 64, x.H/2, x.W/2)
        self.layer0_1x1 = convrelu(64, 64, 1, 0)
        self.layer1 = nn.Sequential(*self.base_layers[3:5])  # size=(N, 64, x.H/4, x.W/4)
        self.layer1_1x1 = convrelu(64, 64, 1, 0)
        self.layer2 = self.base_layers[5]  # size=(N, 128, x.H/8, x.W/8)
        self.layer2_1x1 = convrelu(128, 128, 1, 0)
        self.layer3 = self.base_layers[6]  # size=(N, 256, x.H/16, x.W/16)
        self.layer3_1x1 = convrelu(256, 256, 1, 0)
        self.layer4 = self.base_layers[7]  # size=(N, 512, x.H/32, x.W/32)
        self.layer4_1x1 = convrelu(512, 512, 1, 0)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        self.upsample_360_23 = nn.Upsample(size=[30, 40], mode='bilinear', align_corners=True)
        self.upsample_360_45 = nn.Upsample(size=[60, 80], mode='bilinear', align_corners=True)
        self.conv_up3 = convrelu(256 + 512, 512, 3, 1)
        self.conv_up2 = convrelu(128 + 512, 256, 3, 1)
        self.conv_up1 = convrelu(64 + 256, 256, 3, 1)
        self.conv_up0 = convrelu(64 + 256, 128, 3, 1)

        self.conv_original_size0 = convrelu(3, 64, 3, 1)
        self.conv_original_size1 = convrelu(64, 64, 3, 1)
        self.conv_original_size2 = convrelu(64 + 128, 64, 3, 1)

        self.conv_last = nn.Conv2d(64, n_class, 1)

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
        #print(x.shape)
        layer3 = self.layer3_1x1(layer3)
        #print(layer3.shape)
        x = self.upsample_360_23(x)
        #
        # print('layer3:', layer3.shape,'x',x.shape)
        x = torch.cat([x, layer3], dim=1)
        x = self.conv_up3(x)

        x = self.upsample(x)
        #print(x.shape)
        layer2 = self.layer2_1x1(layer2)
        #print(layer2.shape)
        x = self.upsample_360_45(x)
        # print('layer2:', layer2.shape,'x',x.shape)
        x = torch.cat([x, layer2], dim=1)
        x = self.conv_up2(x)

        x = self.upsample(x)
        layer1 = self.layer1_1x1(layer1)
        # print('layer1:', layer1.shape,'x',x.shape)
        x = torch.cat([x, layer1], dim=1)
        x = self.conv_up1(x)

        x = self.upsample(x)
        layer0 = self.layer0_1x1(layer0)
        # print('layer0:', layer0.shape,'x',x.shape)
        x = torch.cat([x, layer0], dim=1)
        x = self.conv_up0(x)

        x = self.upsample(x)
        # print('x_original:', x_original.shape,'x',x.shape)
        x = torch.cat([x, x_original], dim=1)
        x = self.conv_original_size2(x)

        out = self.conv_last(x)

        return out

class PSTNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()
        #rgb_net_weights = torch.load("../data/weights/pstnet/model_best.pth")
        rgb_net = RGBNet(3, 5)
        #rgb_net = torch.nn.DataParallel(rgb_net)
        #rgb_net = rgb_net.cuda()
        #rgb_net = self.load_my_state_dict(rgb_net, rgb_net_weights)
        for p in rgb_net.parameters():
            p.requires_grad = False

        self.rgb_net = rgb_net

        # ================== ENCODER ==================
        self.initial_block_enc = DownsamplerBlock(9, 32)

        self.layers_enc = nn.ModuleList()
        self.layers_enc.append(DownsamplerBlock(32, 64))

        # Encoder 5 Stack
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1))
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1))
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1))
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1))
        self.layers_enc.append(non_bottleneck_1d(64, 0.03, 1))

        self.layers_enc.append(DownsamplerBlock(64, 128))
        # Encoder 2 Stack
        # 1)
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 2))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 4))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 8))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 16))
        # 2)
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 2))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 4))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 8))
        self.layers_enc.append(non_bottleneck_1d(128, 0.3, 16))

        # ================== DECODER ==================
        self.layers_dec = nn.ModuleList()
        self.layers_dec.append(UpsamplerBlock(128, 64))
        self.layers_dec.append(non_bottleneck_1d(64, 0, 1))
        self.layers_dec.append(non_bottleneck_1d(64, 0, 1))

        self.layers_dec.append(UpsamplerBlock(64, 16))
        self.layers_dec.append(non_bottleneck_1d(16, 0, 1))
        self.layers_dec.append(non_bottleneck_1d(16, 0, 1))

        # Final output
        self.output_conv = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, n_class, 3, 1, 1)
        )

    def forward(self, input):

        rgb_out = self.rgb_net(input[:, 0:3, :,:])

        for bidx in range(0, rgb_out.shape[0]):
            tensor_min = torch.min(rgb_out[bidx, ...])
            tensor_max = torch.max(rgb_out[bidx, ...])
            tensor_range = tensor_max - tensor_min
            rgb_out[bidx, ...] = (rgb_out[bidx, ...] - tensor_min) / tensor_range

        input = torch.cat([rgb_out, input[:, 3, ...].unsqueeze(1), input[:, 0:3, ...]], 1)
        # import pdb; pdb.set_trace()

        output = self.initial_block_enc(input)
        for layer_enc in self.layers_enc:
            output = layer_enc(output)
        for layer_dec in self.layers_dec:
            output = layer_dec(output)
        output = self.output_conv(output)
        return output

    def load_my_state_dict(self, g_model, state_dict):
        own_state = g_model.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                print('[ERROR] Could not load weights correctly!')
                continue
            own_state[name].copy_(param)
        return g_model
