import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from mi.nets import MIFCNet, MI1x1ConvNet

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, feat):
        return feat.view(feat.size(0), -1)

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        self.mutual_info_loss = mutual_info_loss
        local_units = 1024
        mi_units = 1024
        global_units = 64
        self.local_net = MI1x1ConvNet(local_units, mi_units)
        self.global_net = MIFCNet(global_units, mi_units)
        
        conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        conv3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )  
        
        self.blocks = nn.ModuleList([
            conv1,
            conv2,
            conv3
        ])
    
    
    def forward(self, x):
        
        features = x
        
        for block in zip(self.blocks):
            features = block(features)
            
        features = self.local_net(features)

        vector = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)

        vector = self.global_net(vector)
        
        return features, vector

    
