#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import ceil

from pointnet2_patch_encode.Data.bbox import BBox


def getCropPointArray(point_array, bbox):
    mask_x = (point_array[:, 0] >= bbox.min_point.x) &\
        (point_array[:, 0] <= bbox.max_point.x)
    mask_y = (point_array[:, 1] >= bbox.min_point.y) &\
        (point_array[:, 1] <= bbox.max_point.y)
    mask_z = (point_array[:, 2] >= bbox.min_point.z) &\
        (point_array[:, 2] <= bbox.max_point.z)
    mask = mask_x & mask_y & mask_z

    crop_point_array = point_array[mask]
    return crop_point_array


def getCropPointArrayList(point_array,
                          crop_patch_size=0.1,
                          crop_patch_move_step=0.05):
    crop_patch_num = ceil(1.0 / crop_patch_move_step)
    start_center_value = 0.5 * crop_patch_move_step

    crop_point_array_list = []
    crop_bbox_list = []

    for i in range(crop_patch_num):
        center_x = start_center_value + i * crop_patch_move_step
        for j in range(crop_patch_num):
            center_y = start_center_value + j * crop_patch_move_step
            for k in range(crop_patch_num):
                center_z = start_center_value + k * crop_patch_move_step
                center = np.array([center_x, center_y, center_z])
                min_point = center - crop_patch_size / 0.5
                max_point = center + crop_patch_size / 0.5
                bbox = BBox.fromList([min_point, max_point])
                crop_point_array = getCropPointArray(point_array, bbox)
                if crop_point_array.shape[0] > 0:
                    crop_point_array_list.append(crop_point_array)
                    crop_bbox_list.append(bbox)
    return crop_point_array_list, crop_bbox_list
