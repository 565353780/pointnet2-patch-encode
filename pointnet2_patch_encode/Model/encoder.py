#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from pointnet2_patch_encode.Model.utils.set_abstraction import SetAbstraction
from pointnet2_patch_encode.Model.utils.set_abstraction_msg import SetAbstractionMsg


class Encoder(nn.Module):

    def __init__(self,
                 min_radius=0.01,
                 min_sample_point_num=1,
                 min_mlp_channel=4,
                 feature_size=128,
                 normal_channel=True):
        super(Encoder, self).__init__()

        self.feature_size = feature_size
        self.normal_channel = normal_channel

        in_channel = 3 if normal_channel else 0
        self.sa1 = SetAbstractionMsg(
            min_sample_point_num * 32,
            [min_radius, min_radius * 2, min_radius * 4], [
                min_sample_point_num, min_sample_point_num * 2,
                min_sample_point_num * 8
            ], in_channel,
            [[min_mlp_channel, min_mlp_channel, min_mlp_channel * 2],
             [min_mlp_channel * 2, min_mlp_channel * 2, min_mlp_channel * 4],
             [min_mlp_channel * 2, min_mlp_channel * 3, min_mlp_channel * 4]])
        self.sa2 = SetAbstractionMsg(
            min_sample_point_num * 8,
            [min_radius * 2, min_radius * 4, min_radius * 8], [
                min_sample_point_num * 2, min_sample_point_num * 4,
                min_sample_point_num * 8
            ], min_mlp_channel * 10,
            [[min_mlp_channel * 2, min_mlp_channel * 2, min_mlp_channel * 4],
             [min_mlp_channel * 4, min_mlp_channel * 4, min_mlp_channel * 8],
             [min_mlp_channel * 4, min_mlp_channel * 4, min_mlp_channel * 8]])
        self.sa3 = SetAbstraction(
            None, None, None, min_mlp_channel * 20 + 3,
            [min_mlp_channel * 8, min_mlp_channel * 16, feature_size], True)
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
