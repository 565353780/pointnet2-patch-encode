#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np

from pointnet2_patch_encode.Model.class_msg import ClassMsg


class Detector(object):

    def __init__(self):
        self.model = ClassMsg(936, False).cuda()
        self.model.eval()
        return

    def detect(self, point_array):
        inputs = torch.tensor(point_array.astype(np.float32)).transpose(
            2, 1).cuda()
        pred, trans_feat = self.model(inputs)
        return pred, trans_feat
