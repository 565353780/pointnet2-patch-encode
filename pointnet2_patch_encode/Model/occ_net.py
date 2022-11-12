#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.nn as nn

from pointnet2_patch_encode.Model.resnet_decoder import ResNetDecoder
from pointnet2_patch_encode.Model.resnet_encoder import ResNetEncoder


class OccNet(nn.Module):

    def __init__(self, feature_size=256):
        super(OccNet, self).__init__()
        self.feature_size = feature_size

        self.encoder = ResNetEncoder(feature_size, relu_out=True)
        self.decoder = ResNetDecoder(feature_size, relu_in=True)

        self.mse_loss = nn.MSELoss(reduction='mean')
        return

    def loss(self, data):
        data['losses']['loss_occ'] = self.mse_loss(
            data['inputs']['occ'], data['predictions']['feature'])
        return data

    def forward(self, data):
        data['predictions']['encode'] = self.encoder(data['inputs']['occ'])
        data['predictions']['feature'] = self.decoder(
            data['predictions']['encode'])

        if self.training:
            data = self.loss(data)
        return data
