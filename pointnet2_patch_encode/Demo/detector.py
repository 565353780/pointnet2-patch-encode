#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tqdm import trange

from pointnet2_patch_encode.Module.detector import Detector


def demo():
    detector = Detector()

    points = np.random.randn(1000, 32, 3)

    for i in trange(100):
        data = detector.detect(points)
        #  print(data['predictions']['encode'].shape)
        #  print(data['predictions']['feature'].shape)
    return True
