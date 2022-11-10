#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from multiprocessing import Pool

import numpy as np
import open3d as o3d
from tqdm import tqdm

from pointnet2_patch_encode.Method.render import getOccPointCloud


def getPointArrayOcc(point_array, bbox, occ_size=32, render=False):
    assert occ_size > 0

    x_step = bbox.diff_point.x / occ_size
    y_step = bbox.diff_point.y / occ_size
    z_step = bbox.diff_point.z / occ_size

    occ = np.zeros([occ_size, occ_size, occ_size], dtype=np.float32)
    for i in range(occ_size):
        min_point_x = bbox.min_point.x + i * x_step
        max_point_x = min_point_x + x_step
        mask_x = (point_array[:, 0] >= min_point_x) &\
            (point_array[:, 0] <= max_point_x)
        for j in range(occ_size):
            min_point_y = bbox.min_point.y + j * y_step
            max_point_y = min_point_y + y_step
            mask_y = (point_array[:, 1] >= min_point_y) &\
                (point_array[:, 1] <= max_point_y)
            mask_xy = mask_x & mask_y
            for k in range(occ_size):
                min_point_z = bbox.min_point.z + k * z_step
                max_point_z = min_point_z + z_step
                mask_z = (point_array[:, 2] >= min_point_z) &\
                    (point_array[:, 2] <= max_point_z)
                mask = mask_xy & mask_z

                mask_point_array = point_array[mask]
                if mask_point_array.shape[0] > 0:
                    occ[i][j][k] = 1.0

    if not render:
        return occ

    input_pcd = o3d.geometry.PointCloud()
    colors = np.zeros_like(point_array)
    colors[:, 1] = 1.0
    input_pcd.points = o3d.utility.Vector3dVector(point_array)
    input_pcd.colors = o3d.utility.Vector3dVector(colors)
    occ_pcd = getOccPointCloud(bbox, occ)
    o3d.visualization.draw_geometries([input_pcd, occ_pcd])
    return occ


def getPointArrayOccWithInputs(inputs):
    point_array, bbox, occ_size = inputs
    return getPointArrayOcc(point_array, bbox, occ_size)


def getPointArrayOccWithPool(point_array_list, bbox_list, occ_size=32):
    assert len(point_array_list) > 0

    inputs_list = []
    for point_array, bbox in zip(point_array_list, bbox_list):
        inputs_list.append([point_array, bbox, occ_size])
    with Pool(os.cpu_count()) as pool:
        result = list(
            tqdm(pool.imap(getPointArrayOccWithInputs, inputs_list),
                 total=len(inputs_list)))
    return result
