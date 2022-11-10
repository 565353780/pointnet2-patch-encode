#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

from pointnet2_patch_encode.Model.class_msg import ClassMsg


class Detector(object):

    def __init__(self):
        self.model = ClassMsg().cuda()
        self.model.eval()
        return

    def detect(self, point_array):
        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        data['inputs']['point_array'] = torch.tensor(
            point_array.astype(np.float32)).transpose(2, 1).cuda()

        data = self.model(data)
        return data
