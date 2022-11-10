#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import open3d as o3d


def getOccPointCloud(bbox, occ):
    assert occ.shape[0] > 0

    x_step = bbox.diff_point.x / occ.shape[0]
    y_step = bbox.diff_point.y / occ.shape[1]
    z_step = bbox.diff_point.z / occ.shape[2]

    point_list = []
    occ_list = []

    for i in range(occ.shape[0]):
        point_x = bbox.min_point.x + (i + 0.5) * x_step
        for j in range(occ.shape[1]):
            point_y = bbox.min_point.y + (j + 0.5) * y_step
            for k in range(occ.shape[2]):
                point_z = bbox.min_point.z + (k + 0.5) * z_step

                point_list.append([point_x, point_y, point_z])
                occ_list.append(occ[i][j][k])

    point_array = np.array(point_list)
    occ_array = np.array(occ_list)

    valid_point_idx = np.where(occ_array > 0.5)[0]

    valid_point_array = point_array[valid_point_idx]
    valid_color_array = np.zeros_like(valid_point_array)
    valid_color_array[:, 0] = 1.0

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(valid_point_array)
    pcd.colors = o3d.utility.Vector3dVector(valid_color_array)
    return pcd


def renderOcc(bbox, occ):
    occ_pcd = getOccPointCloud(bbox, occ)
    o3d.visualization.draw_geometries([occ_pcd])
    return True
