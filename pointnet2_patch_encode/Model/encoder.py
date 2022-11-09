#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from pointnet2_patch_encode.Model.utils.set_abstraction import SetAbstraction
from pointnet2_patch_encode.Model.utils.set_abstraction_msg import SetAbstractionMsg


class Encoder(nn.Module):

    def __init__(self, feature_size=1024, normal_channel=True):
        super(Encoder, self).__init__()
        self.feature_size = feature_size
        self.normal_channel = normal_channel

        in_channel = 3 if normal_channel else 0
        self.sa1 = SetAbstractionMsg(
            512, [0.1, 0.2, 0.4], [16, 32, 128], in_channel,
            [[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = SetAbstractionMsg(
            128, [0.2, 0.4, 0.8], [32, 64, 128], 320,
            [[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = SetAbstraction(None, None, None, 640 + 3,
                                  [256, 512, feature_size], True)
        return

    def forward(self, xyz):
        B = xyz.shape[0]

        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None

        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        _, l3_points = self.sa3(l2_xyz, l2_points)

        x = l3_points.view(B, self.feature_size)
        return x
