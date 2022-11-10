#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import trange

from pointnet2_patch_encode.Module.detector import Detector


def demo():
    detector = Detector()

    points = np.random.randn(1000, 32, 3)

    for i in trange(100):
        pred, trans_feat = detector.detect(points)
        #  print(pred.shape)
        #  print(trans_feat.shape)
    return True
