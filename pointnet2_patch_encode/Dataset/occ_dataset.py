#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from copy import deepcopy

import numpy as np
import open3d as o3d
import torch
from torch.utils.data import Dataset

from pointnet2_patch_encode.Data.bbox import BBox

from pointnet2_patch_encode.Method.sample import getSamplePointArray
from pointnet2_patch_encode.Method.occ import getPointArrayOcc


class OccDataset(Dataset):

    def __init__(self, dataset_manager, unit_size=0.1, input_size=0.3):
        self.dataset_manager = dataset_manager
        self.unit_size = unit_size
        self.input_size = input_size

        self.scannet_scene_name_list = self.dataset_manager.getScanNetSceneNameList(
        )

        # collect all scene-object udf files as list-dict
        self.scannet_scene_name = None
        self.points_list = []
        self.test_points_list = []
        return

    def reset(self):
        self.scannet_scene_name = None
        self.points_list = []
        self.test_points_list = []
        return True

    def loadScene(self, scannet_scene_name):
        assert scannet_scene_name in self.scannet_scene_name_list

        self.reset()

        sample_scale = 0.5

        self.scannet_scene_name = scannet_scene_name
        self.scannet_scene_object_file_name_list = self.dataset_manager.getScanNetObjectFileNameList(
            self.scannet_scene_name)

        for scannet_scene_object_file_name in self.scannet_scene_object_file_name_list:
            shapenet_model_tensor_dict = self.dataset_manager.getShapeNetModelTensorDict(
                self.scannet_scene_name, scannet_scene_object_file_name, False)

            scannet_object_file_path = shapenet_model_tensor_dict[
                'scannet_object_file_path']
            shapenet_model_file_path = shapenet_model_tensor_dict[
                'shapenet_model_file_path']

            object_points = np.array(
                o3d.io.read_point_cloud(scannet_object_file_path).points)
            sample_object_points = getSamplePointArray(object_points,
                                                       sample_scale)

            self.points_list.append(sample_object_points)
            self.test_points_list.append(sample_object_points.reshape(
                1, -1, 3))

            cad_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
            pcd = cad_mesh.sample_points_uniformly(20000)
            cad_points = np.array(pcd.points)
            sample_cad_points = getSamplePointArray(cad_points, sample_scale)

            self.points_list.append(sample_cad_points)
            self.test_points_list.append(sample_cad_points.reshape(1, -1, 3))

            bbox = BBox.fromList([[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]])
            occ = getPointArrayOcc(sample_cad_points, bbox)
            exit()
        return True

    def loadSceneByIdx(self, scannet_scene_name_idx):
        assert scannet_scene_name_idx <= len(self.scannet_scene_name_list)

        return self.loadScene(
            self.scannet_scene_name_list[scannet_scene_name_idx])

    def __getitem__(self, index, test=False):
        assert self.scannet_scene_name is not None

        assert index <= len(self.points_list)

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}
        if test:
            data['inputs']['point_array'] = torch.tensor(
                self.test_points_list[index].astype(np.float32)).transpose(
                    2, 1)
        else:
            data['inputs']['point_array'] = torch.tensor(
                self.points_list[index].astype(np.float32)).transpose(1, 0)
        return data

    def __len__(self):
        return len(self.points_list)
