#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn
import torch.nn.functional as F

from pointnet2_patch_encode.Model.encoder import Encoder
from pointnet2_patch_encode.Model.resnet_decoder import ResNetDecoder


class ClassMsg(nn.Module):

    def __init__(self, feature_size=1024, normal_channel=False):
        super(ClassMsg, self).__init__()
        self.feature_size = feature_size

        self.encoder = Encoder(feature_size, normal_channel)
        self.decoder = ResNetDecoder(feature_size, relu_in=True)
        return

    def forward(self, xyz):
        l3_points = self.encoder(xyz)
        x = self.decoder(l3_points)
        return x, l3_points


class get_loss(nn.Module):

    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):
        total_loss = F.nll_loss(pred, target)
        return total_loss
