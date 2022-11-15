# coding:utf-8
# By XunJie He
# Email: hxj990319@163.com

import torch
import torch.nn as nn
import cv2
import numpy as np
from torch.autograd import Variable
from torchvision import models
import torch.nn.functional as F
from PIL import Image
#from pyheatmap.heatmap import HeatMap
import math
import torch.utils.model_zoo as model_zoo
from thop import profile
import pandas
import seaborn as sns
import numpy as np
import os
import torch
import matplotlib.pyplot as plt

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super().__init__()

        # B, C, H, W -> B, C/4, H, W
        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C/4, H, W
        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3,
                                          stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        # B, C/4, H, W -> B, C, H, W
        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x

class ASPP_origin(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP_origin, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=18, dilation=18)
        #self.atrous_block30 = nn.Conv2d(in_channel, depth, 3, 1, padding=30, dilation=30)
        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        image_features = self.mean(x)  #[2,2048,1,1]
        image_features = self.conv(image_features)  #[2,2048,1,1]
        image_features = F.upsample(image_features, size=size, mode='bilinear')  #[2,2048,15,20]

        atrous_block1 = self.atrous_block1(x) #[2,2048,15,20]
        atrous_block6 = self.atrous_block6(x)  #[2,2048,15,20]
        atrous_block12 = self.atrous_block12(x) #[2,2048,15,20]
        atrous_block18 = self.atrous_block18(x) #[2,2048,15,20]
       # atrous_block30 = self.atrous_block30(x)  # [2,2048,15,20]
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, int(in_channel // 4), kernel_size=1, stride=1, padding=0)
        self.context1 = nn.Sequential(
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=1, dilation=1),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
        )
        self.context2 = nn.Sequential(
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=8, dilation=8),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
        )
        self.context3 = nn.Sequential(
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=8, dilation=8),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
        )
        self.context4 = nn.Sequential(
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=2, dilation=2),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=4, dilation=4),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(in_channel // 4), int(in_channel // 4), 3, 1, padding=8, dilation=8),
            nn.BatchNorm2d(int(in_channel // 4)),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv2d(int(in_channel // 4), depth, kernel_size=1, stride=1, padding=0)
        self.dropout = nn.Dropout(p=0.5)
    def forward(self, x):
        img = self.conv1(x)
        #print(img.shape)
        img = self.context1(img)+self.context2(img)+self.context3(img)+ self.context4(img)
        #print(img.shape)
        part1 = self.conv2(img)
        out = self.dropout(x+part1)
        #print(out.shape)
        return out


class SFAFMA(nn.Module):
    def __init__(self, n_class):
        super(SFAFMA, self).__init__()
        resnet_raw_model1 = models.resnet101(pretrained=True) #preweights of backbone
        resnet_raw_model2 = models.resnet101(pretrained=True)

        filters = [64, 256, 512, 1024, 2048]
        #filters = [64, 64, 128, 256, 512]

        self.encoder_thermal_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_thermal_conv1.weight.data = torch.unsqueeze(torch.mean(resnet_raw_model1.conv1.weight.data, dim=1),
                                                                 dim=1)
        self.encoder_thermal_bn1 = resnet_raw_model1.bn1
        self.encoder_thermal_relu = resnet_raw_model1.relu
        self.encoder_thermal_maxpool = resnet_raw_model1.maxpool
        self.encoder_thermal_layer1 = resnet_raw_model1.layer1
        self.encoder_thermal_layer2 = resnet_raw_model1.layer2
        self.encoder_thermal_layer3 = resnet_raw_model1.layer3
        self.encoder_thermal_layer4 = resnet_raw_model1.layer4

        ########  RGB ENCODER  ########

        self.encoder_rgb_conv1 = resnet_raw_model2.conv1
        self.encoder_rgb_bn1 = resnet_raw_model2.bn1
        self.encoder_rgb_relu = resnet_raw_model2.relu
        self.encoder_rgb_maxpool = resnet_raw_model2.maxpool
        self.encoder_rgb_layer1 = resnet_raw_model2.layer1
        self.encoder_rgb_layer2 = resnet_raw_model2.layer2
        self.encoder_rgb_layer3 = resnet_raw_model2.layer3
        self.encoder_rgb_layer4 = resnet_raw_model2.layer4

        """
        self.conv0 = nn.Conv2d(16, 64, 1, 1)
        self.conv1 = nn.Conv2d(16, 64, 1, 1)
        self.conv2 = nn.Conv2d(32, 128, 1, 1)
        self.conv3 = nn.Conv2d(64, 256, 1, 1)
        self.conv4 = nn.Conv2d(128, 512, 1, 1)
        """
        self.conv0 = nn.Conv2d(16, 64, 1, 1)
        self.conv1 = nn.Conv2d(64, 256, 1, 1)
        self.conv2 = nn.Conv2d(128, 512, 1, 1)
        self.conv3 = nn.Conv2d(256, 1024, 1, 1)
        self.conv4 = nn.Conv2d(512, 2048, 1, 1)



        self.encoder_relu = nn.ReLU(inplace=True)
        self.encoder_gap = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder_downsampling1 = nn.Conv2d(64,256, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling2 = nn.Conv2d(256,512, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling3 = nn.Conv2d(512,1024, kernel_size=7, stride=2, padding=3, bias=False)
        self.encoder_downsampling4 = nn.Conv2d(1024,2048, kernel_size=7, stride=2, padding=3, bias=False)

        channels = [64, 256, 512, 1024, 2048]
        #channels =[64, 64, 128, 256, 512]
        self.local_att0 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[0], int(channels[0] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[0] // 4)),
            nn.ReLU(inplace=True),
        )
        # 全局注意力
        self.global_att0 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[0], int(channels[0] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[0] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att1 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[1], int(channels[1] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[1] // 4)),
            nn.ReLU(inplace=True),
        )
        # 全局注意力
        self.global_att1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[1], int(channels[1] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[1] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att2 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[2], int(channels[2] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[2] // 4)),
            nn.ReLU(inplace=True),
        )
        # 全局注意力
        self.global_att2 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[2], int(channels[2] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[2] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att3 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[3], int(channels[3] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[3] // 4)),
            nn.ReLU(inplace=True),
        )
        # 全局注意力
        self.global_att3 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[3], int(channels[3] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[3] // 4)),
            nn.ReLU(inplace=True),
        )

        self.local_att4 = nn.Sequential(
            nn.MaxPool2d(1, padding=0, stride=1),
            nn.Conv2d(channels[4], int(channels[4] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[4] // 4)),
            nn.ReLU(inplace=True),
        )
        # 全局注意力
        self.global_att4 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels[4], int(channels[4] // 4), kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(int(channels[4] // 4)),
            nn.ReLU(inplace=True),
        )
        self.sigmoid = nn.Sigmoid()
        self.aspp2 = ASPP(filters[2], filters[2])
        self.aspp3 = ASPP(filters[3], filters[3])
        self.aspp4= ASPP(filters[4],filters[4])
        # Decoder
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

        x_list=list()
        x_2=list()

        # encoder
        ######################################################################
         # (480, 640)
        ######################################################################
        rgb = self.encoder_rgb_conv1(rgb)
        # (240, 320)
        rgb = self.encoder_rgb_bn1(rgb)
        # (240, 320)
        rgb = self.encoder_rgb_relu(rgb)
        # (240, 320)


        thermal = self.encoder_thermal_conv1(thermal)
        # (240, 320)
        thermal = self.encoder_thermal_bn1(thermal)
        # (240, 320)
        thermal = self.encoder_thermal_relu(thermal)
        # (240, 320)

        # e0 = self.conv0(torch.cat([rgb,thermal],dim=1))    #64
        e0 = rgb + thermal

        e00 = self.local_att0(e0)
        e01 = self.global_att0(e0)
        w0 = self.sigmoid(self.conv0(e00+e01))
        e02 = rgb * w0 + thermal * (1 - w0)



        #print(e0.shape)
        #[2,96,240,320]
        #x_list.append(e0)
        #e00 = self.encoder_relu(e0)
        #e01 = self.encoder_gap(e00)
        #e03 = e01 * rgb + (1 - e01) * thermal

        # img_feats1=HeatMap(e03[1,1,:,:])
        # print(np.array(e03[1,1,:,:]*255).astype(int))

        # e04=torch.ceil(e03[1,1,:,:]*255) #how to build a new picture
        # print(e04)
        # img_feats1 = cv2.normalize(e04, None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # img_feats1 = cv2.applyColorMap(img_feats1, cv2.COLORMAP_JET)
        # cv2.imshow('heatimage1', img_feats1)
        # cv2.waitKey(2000)

        rgb = self.encoder_rgb_maxpool(e0)
        thermal = self.encoder_thermal_maxpool(thermal)
        #[2,96,120,160]
        rgb = self.encoder_rgb_layer1(rgb)
        thermal = self.encoder_thermal_layer1(thermal)
        #[2,384,120,160]
        # e1 =self.conv1(torch.cat([rgb,thermal],dim=1))     #256
        e1 = rgb + thermal
        e10 = self.local_att1(e1)
        e11 = self.global_att1(e1)
        w1 = self.sigmoid(self.conv1(e10+e11))
        e12 = rgb * w1 + thermal * (1 - w1)



        #x_list.append(e1)
        #print(x_list)
        #e10 = self.encoder_relu(e1)
        #e11 = self.encoder_gap(e10)
        #e12 = self.encoder_downsampling1(e03)
        #e13 = e12 + e11 * rgb + (1 - e11) * thermal

        # img_feats2 = cv2.normalize(e13[1, 1, :, :], None, 255, 0, cv2.NORM_MINMAX, cv2.CV_8UC1)
        # img_feats2 = cv2.applyColorMap(img_feats2, cv2.COLORMAP_JET)
        # cv2.imshow('heatimage2', img_feats2)
        # cv2.waitKey(2000)

        ######################################################################

        # [2,192,60,80]
        rgb = self.encoder_rgb_layer2(e1)
        thermal = self.encoder_thermal_layer2(thermal)
        #[2,768,60,80]

        e2 = rgb + thermal
        #e2 = self.aspp2(e2)
        e20 = self.local_att2(e2)
        e21 = self.global_att2(e2)
        w2 = self.sigmoid(self.conv2(e20+e21))
        e22 = rgb * w2 + thermal * (1 - w2)


        rgb = self.encoder_rgb_layer3(e2)
        thermal = self.encoder_thermal_layer3(thermal)
        e3 = rgb + thermal
        e30 = self.local_att3(e3)
        e31 = self.global_att3(e3)
        w3 = self.sigmoid(self.conv3(e30 + e31))
        e32 = rgb * w3 + thermal * (1 - w3)

        rgb = self.encoder_rgb_layer4(e3)
        thermal = self.encoder_thermal_layer4(thermal)
        # [2,2208,15,20]
        e4 = rgb + thermal
        e40 = self.local_att4(e4)
        e41 = self.global_att4(e4)
        w4 = self.sigmoid(self.conv4(e40 + e41))
        e42 = rgb * w4 + thermal * (1 - w4)
        e43=self.aspp4(e42)

        ######################################################################

        # decoder

        d4 = self.decoder4(e43)+e32
        #d4 = self.conv3(torch.cat([d4,e3], dim=1))
        d3 = self.decoder3(d4)+e22
        #d3 = self.conv2(torch.cat([d3, e2], dim=1))
        d2 = self.decoder2(d3)+e12
        #d2 = self.conv1(torch.cat([d2, e1], dim=1))
        d1 = self.decoder1(d2)+e02
        #d1 = self.conv0(torch.cat([d1, e0], dim=1))
        """

        d4 = self.decoder4(e42) + e32
        d3 = self.decoder3(d4) + e22
        d2 = self.decoder2(d3) + e12
        d1 = self.decoder1(d2) + e02
        """
        fuse = self.finaldeconv1(d1)
        fuse = self.finalrelu1(fuse)
        fuse = self.finalconv2(fuse)
        fuse = self.finalrelu2(fuse)
        fuse = self.finalconv3(fuse)

        return fuse

"""
def densenet_testing():
    import time
    net = models.densenet161(pretrained=True).cuda()

    input_data = torch.randn(2,3,640,480).cuda()

    for i in range(100000):
        torch.cuda.synchronize()
        start_time = time.time()
        output_data = net(input_data)
        torch.cuda.synchronize()
        end_time = time.time()
        print(' [INFO] cost time {:.2f} ms'.format(1000*(end_time-start_time)))
"""
class testm(nn.Module):
    def __init__(self, n_class):
        super(testm, self).__init__()

        self.local_att = nn.Sequential(

            nn.Conv2d(3, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
        )
        # 全局注意力
        self.global_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(2, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(64),
        )
    def forward(self, input):
        rgb = input[:, :3]
        thermal = input[:, 3:]
        e = rgb+thermal
        e1 = self.local_att(e)
        e2 = self.global_att(e)
        output=e1+e2
        return output

def get_parameter_number(net):
    total_num = sum(p.numel() for p in net.parameters())
    trainable_num = sum(p.numel() for p in net.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def unit_test():
    num_minibatch = 2
    rgb = torch.randn(num_minibatch, 3, 480, 640).cuda(0)
    thermal = torch.randn(num_minibatch, 1, 480, 640).cuda(0)
    sfafma = SFAFMA(9).cuda(0)
    #print(sfafma)
    #test = testm(9).cuda(0)
    #para_num_dict=get_parameter_number(test)
    #print(para_num_dict['Total'])
    input = torch.cat((rgb, thermal), dim=1)
    flops, params = profile(sfafma, inputs=(input,))
    print(flops / 1e9, params / 1e6)
    sfafma(input)


if __name__ == '__main__':
    unit_test()
