#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
from tqdm import trange

from pointnet2_patch_encode.Model.class_msg import ClassMsg


class Trainer(object):

    def __init__(self):
        self.model = ClassMsg().cuda()
        self.model.train()
        return

    def processData(self, point_array):
        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['inputs']['point_array'] = torch.tensor(
            point_array.astype(np.float32)).transpose(2, 1).cuda()

        return data

    def trainStep(self, data):
        return data

    def train(self):
        for i in trange(100):
            points = np.random.randn(1000, 32, 3)
            data = self.processData(point_array)
        return True
