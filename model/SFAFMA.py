# coding:utf-8
# By XunJie He
# Email: hxj990319@163.com

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
import  math
import torch.utils.model_zoo as model_zoo

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, filters):
        super().__init__()

        #B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, filters, 1)
        self.norm3 = nn.BatchNorm2d(filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self,x):
        x=self.conv1(x)
        x=self.norm1(x)
        x=self.relu1(x)
        x=self.deconv2(x)
        x=self.norm2(x)
        x=self.relu2(x)
        x=self.conv3(x)
        x=self.norm3(x)
        x=self.relu3(x)
        return x



class SFAFMA(nn.Module):
    def __init__(self, n_class):
        super(SFAFMA, self).__init__()
        resnet_raw_model1 = models.resnet152(pretrained=True)
        resnet_raw_model2 = models.resnet152(pretrained=True)

        filters = [64, 256, 512, 1024, 2048]

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)

        #Thermal Encoder
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        #RGB Encoder
        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        
        self.conv0 = nn.Conv2d(128, 64, 1, 1)
        self.conv1 = nn.Conv2d(512, 256, 1, 1)
        self.conv2 = nn.Conv2d(1024, 512, 1, 1)
        self.conv3 = nn.Conv2d(2048, 1024, 1, 1)
        self.conv4 = nn.Conv2d(4096, 2048, 1, 1)
        self.encoder_relu = nn.ReLU(inplace=True)
        self.encoder_gap = nn.AdaptiveAvgPool2d((1,1))
        self.encoder_downsampling1 = nn.Conv2d(64, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling2 = nn.Conv2d(256, 512, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling3 = nn.Conv2d(512, 1024, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling4 = nn.Conv2d(1024, 2048, kernel_size=7, stride=2, padding=3, bias=False)

        #Decoder
        self.decoder4 = DecoderBlock(filters[4], filters[3])
        self.decoder3 = DecoderBlock(filters[3], filters[2])
        self.decoder2 = DecoderBlock(filters[2], filters[1])
        self.decoder1 = DecoderBlock(filters[1], filters[0])
        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, 3, stride=2)
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, 3)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, n_class, 2, padding=1)



    def forward(self, input):
        rgb = input[:, :3]
        thermal = input[:, 3:]

        verbose = False

        # encoder
        rgb = self.encoder_rgb_conv1(rgb)
        rgb = self.encoder_rgb_bn1(rgb)
        rgb = self.encoder_rgb_relu(rgb) # (240, 320)

        thermal = self.encoder_thermal_conv1(thermal)
        thermal = self.encoder_thermal_bn1(thermal)
        thermal = self.encoder_thermal_relu(thermal)# (240, 320)
        #e0 = self.conv0(torch.cat([rgb,thermal],dim=1))    #64
        e0 = rgb + thermal
        #print(e0.shape)
        e00 = self.encoder_relu(e0)
        e01 = self.encoder_gap(e00)
        #print(e01)
        e03 = e01*rgb + (1-e01)*thermal
        #print(e03.shape)

        rgb = self.encoder_rgb_maxpool(e0)
        thermal = self.encoder_thermal_maxpool(thermal) # (120, 160)

        rgb = self.encoder_rgb_layer1(rgb)
        thermal = self.encoder_thermal_layer1(thermal) # (120, 160)
        #e1 =self.conv1(torch.cat([rgb,thermal],dim=1))     #256
        e1=rgb+thermal
        #print(e1.shape)
        e10 = self.encoder_relu(e1)
        e11 = self.encoder_gap(e10)
        #print(e11.shape)
        e12 = self.encoder_downsampling1(e03)
        #print(e12.shape)
        e13 = e12 + e11*rgb + (1-e11)*thermal

        rgb = self.encoder_rgb_layer2(e1)
        thermal = self.encoder_thermal_layer2(thermal) # (60, 80)
        #e2 = self.conv2(torch.cat([rgb,thermal],dim=1))      #512
        e2 =rgb+thermal
        e20 =self.encoder_relu(e2)
        e21 = self.encoder_gap(e20)
        e22 = self.encoder_downsampling2(e13)
        e23 = e22 + e21 * rgb + (1 - e21) * thermal


        rgb = self.encoder_rgb_layer3(e2)
        thermal = self.encoder_thermal_layer3(thermal)  # (30, 40)
        #e3 = self.conv3(torch.cat([rgb,thermal],dim=1))      #1024
        e3 = rgb+thermal
        e30 = self.encoder_relu(e3)
        e31 = self.encoder_gap(e30)
        e32 = self.encoder_downsampling3(e23)
        e33 = e32 + e31 * rgb + (1 - e31) * thermal


        rgb = self.encoder_rgb_layer4(e3)
        thermal = self.encoder_thermal_layer4(thermal) # (15, 20)
        e4 = rgb+thermal
        #e4 = self.conv4(torch.cat([rgb,thermal],dim=1))     #2048
        e40 = self.encoder_relu(e4)
        e41 = self.encoder_gap(e40)
        e42 = self.encoder_downsampling4(e33)
        e43 = e42 + e41 * rgb + (1 - e41) * thermal

        ######################################################################

        # decoder
        d4 = self.decoder4(e43) +e33
        d3 = self.decoder3(d4) +e23
        d2 = self.decoder2(d3) +e13
        d1 = self.decoder1(d2) +e03

        fuse = self.finaldeconv1(d1)
        fuse = self.finalrelu1(fuse)
        fuse = self.finalconv2(fuse)
        fuse = self.finalrelu2(fuse)
        fuse = self.finalconv3(fuse)

        return fuse

def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
    sfafma = SFAFMA(9).cuda(0)
    input = torch.cat((rgb, thermal), dim=1)
    sfafma(input)

if __name__ == '__main__':
    unit_test()
