#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from pointnet2_patch_encode.Model.utils.resnet_block import ResNetBlock


class ResNetDecoder(nn.Module):

    def __init__(self, feats, relu_in=False):
        super().__init__()

        self.feats = [1, 8, 16, 32, 64, feats]

        self.network = nn.Sequential(
            nn.ReLU() if relu_in else nn.Identity(),
            nn.ConvTranspose3d(self.feats[5], self.feats[4], 2),

            # 4 x 4 x 4
            ResNetBlock(self.feats[4]),
            nn.ConvTranspose3d(self.feats[4],
                               self.feats[3],
                               4,
                               stride=2,
                               padding=1),

            # 8 x 8 x 8
            ResNetBlock(self.feats[3]),
            nn.ConvTranspose3d(self.feats[3],
                               self.feats[2],
                               4,
                               stride=2,
                               padding=1),

            # 16 x 16 x 16
            ResNetBlock(self.feats[2]),
            nn.ConvTranspose3d(self.feats[2],
                               self.feats[1],
                               4,
                               stride=2,
                               padding=1),

            # 32 x 32 x 32
            ResNetBlock(self.feats[1]),
            nn.ConvTranspose3d(self.feats[1],
                               self.feats[1],
                               4,
                               stride=2,
                               padding=1),

            # 32 x 32 x 32
            nn.ReLU(inplace=True),
            nn.ConvTranspose3d(self.feats[1],
                               self.feats[0],
                               7,
                               stride=1,
                               padding=3))
        return

    def forward(self, x):
        if x.ndim < 5:
            x = x.reshape(*x.size(), *(1 for _ in range(5 - x.ndim)))
        return self.network(x)
