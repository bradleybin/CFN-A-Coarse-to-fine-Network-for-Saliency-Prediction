# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 09:14:07 2019
"""
import torch
import torch.nn.functional as F
import torch.nn as nn
class DAnet(nn.Module):
    def __init__(self,norm_kwargs=None, **kwargs):
        super(DAnet, self).__init__()
        in_channels = 64
        inter_channels = in_channels // 4
        #print(inter_channels,in_channels)
        self.conv_p1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, kernel_size = 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
        self.conv_p2 = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, kernel_size = 3, padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(True)
        )
    def forward(self, x):
        feat_p = self.conv_p1(x)
        feat_p = self.conv_p2(feat_p)
        return feat_p

class EMLNet(nn.Module):
    def __init__(self):
        super(EMLNet, self).__init__()
        self.conv1_1 = nn.Conv2d(168+64, 64, 3, 1, 1, bias = False)
        self.conv1_2 = nn.Conv2d(1344+64, 64, 3, 1, 1, bias = False)
        self.conv1_3 = nn.Conv2d(2688+64, 64, 3, 1, 1, bias = False)
        self.conv1_4 = nn.Conv2d(4032, 64, 1, 1, 0, bias = False)

        self.upsample = nn.Upsample(scale_factor = 2, mode ='bilinear' , align_corners = True)
        self.fddropout = nn.Dropout2d(p=0.5)
        self.DAnet = DAnet()
        self.conv1_8 = nn.Conv2d(16, 1, 1, 1, 0, bias = False)
        self.upsample8 = nn.Upsample(scale_factor = 4, mode ='bilinear' , align_corners = True)
        
    def forward(self, x1,x2,x3,x4):
        x1 = F.relu(x1)
        x2 = F.relu(x2)
        x3 = F.relu(x3)
        x4 = F.relu(x4)
        
        x4 =  self.conv1_4(x4)
        x4 = F.relu(x4)
        
        x3 = torch.cat([x4, x3], dim=1)
        x3 = self.conv1_3(x3)#
        x3 = F.relu(x3)
        x3 =  self.upsample(x3)
        
        x2 = torch.cat([x3, x2], dim=1)
        x2 = self.conv1_2(x2)#
        x2 = F.relu(x2)
        x2 =  self.upsample(x2)
        x2 =  self.upsample(x2)

        x1 = torch.cat([x2, x1], dim=1)
        x1 = self.conv1_1(x1)#
        x1 = F.relu(x1)

        x = x1
        x = self.fddropout(x)
        x = self.DAnet(x)

        x = self.conv1_8(x)
        x = torch.relu(x)
        x = self.upsample8(x)
        
        return x