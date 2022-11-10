#!/usr/bin/env python
# -*- coding: utf-8 -*-

from random import random

import numpy as np


def getSamplePointArray(point_array,
                        min_sample_scale=0.0,
                        max_sample_scale=1.0):
    assert point_array.shape[0] > 0
    assert 0.0 <= min_sample_scale <= 1.0
    assert 0.0 <= max_sample_scale <= 1.0
    assert min_sample_scale <= max_sample_scale

    point_idx_array = np.arange(0, point_array.shape[0])

    random_sample_scale = min_sample_scale + (max_sample_scale -
                                              min_sample_scale) * random()

    sample_point_num = int(random_sample_scale * point_array.shape[0])
    sample_point_num = max(sample_point_num, 1)
    sample_point_num = min(sample_point_num, point_array.shape[0])

    sample_point_idx_array = np.random.choice(point_idx_array,
                                              sample_point_num,
                                              replace=False)

    sample_points = point_array[sample_point_idx_array]
    return sample_points
