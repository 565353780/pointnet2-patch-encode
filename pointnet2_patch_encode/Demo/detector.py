#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from pointnet2_patch_encode.Module.detector import Detector


def demo():
    detector = Detector()

    points = np.random.randn(3232, 3)

    points = points.reshape(1, -1, 3)

    pred, trans_feat = detector.detect(points)
    print(pred.shape)
    print(trans_feat.shape)
    return True
