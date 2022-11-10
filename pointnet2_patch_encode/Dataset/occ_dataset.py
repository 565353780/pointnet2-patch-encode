#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import pickle
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset

from pointnet2_patch_encode.Method.crop import getCropPointArrayList
from pointnet2_patch_encode.Method.occ import getPointArrayOccListWithPool
from pointnet2_patch_encode.Method.sample import getSamplePointArray


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
        self.bbox_list = []
        self.occ_list = []
        return

    def reset(self):
        self.scannet_scene_name = None
        self.points_list = []
        self.bbox_list = []
        self.occ_list = []
        return True

    def loadScene(self, scannet_scene_name, print_progress=False):
        assert scannet_scene_name in self.scannet_scene_name_list

        self.reset()

        crop_patch_size = 0.1
        crop_patch_move_step = 0.05
        crop_min_point_num = 32
        sample_scale = 0.5

        self.scannet_scene_name = scannet_scene_name
        self.scannet_scene_object_file_name_list = self.dataset_manager.getScanNetObjectFileNameList(
            self.scannet_scene_name)

        occ_save_folder_path = "/home/chli/chLi/pointnet2_patch_encode/occ/" + scannet_scene_name + "/"
        os.makedirs(occ_save_folder_path, exist_ok=True)

        save_idx = 0
        for scannet_scene_object_file_name in self.scannet_scene_object_file_name_list:
            occ_file_path = occ_save_folder_path + str(save_idx) + ".pkl"

            if os.path.exists(occ_file_path):
                occ_dict = pickle.load(open(occ_file_path, "rb"))
                self.points_list += occ_dict['object_points_list']
                self.bbox_list += occ_dict['object_bbox_list']
                self.occ_list += occ_dict['object_occ_list']

                self.points_list += occ_dict['cad_points_list']
                self.bbox_list += occ_dict['cad_bbox_list']
                self.occ_list += occ_dict['cad_occ_list']
                save_idx += 1
                continue

            occ_dict = {}

            shapenet_model_tensor_dict = self.dataset_manager.getShapeNetModelTensorDict(
                self.scannet_scene_name, scannet_scene_object_file_name, False)

            scannet_object_file_path = shapenet_model_tensor_dict[
                'scannet_object_file_path']
            trans_matrix_inv = shapenet_model_tensor_dict[
                'trans_matrix_inv'].numpy()
            shapenet_model_file_path = shapenet_model_tensor_dict[
                'shapenet_model_file_path']

            pcd = o3d.io.read_point_cloud(scannet_object_file_path)
            pcd.transform(trans_matrix_inv)
            object_points = np.array(pcd.points)
            sample_object_points = getSamplePointArray(object_points,
                                                       sample_scale)

            object_crop_points_list, object_crop_bbox_list = getCropPointArrayList(
                sample_object_points, crop_patch_size, crop_patch_move_step,
                crop_min_point_num)

            if len(object_crop_points_list) > 0:
                self.points_list += object_crop_points_list
                self.bbox_list += object_crop_bbox_list
                object_crop_occ_list = getPointArrayOccListWithPool(
                    object_crop_points_list,
                    object_crop_bbox_list,
                    print_progress=print_progress)
                self.occ_list += object_crop_occ_list
                occ_dict['object_points_list'] = object_crop_points_list
                occ_dict['object_bbox_list'] = object_crop_bbox_list
                occ_dict['object_occ_list'] = object_crop_occ_list

            cad_mesh = o3d.io.read_triangle_mesh(shapenet_model_file_path)
            pcd = cad_mesh.sample_points_uniformly(20000)
            cad_points = np.array(pcd.points)
            sample_cad_points = getSamplePointArray(cad_points, sample_scale)

            cad_crop_points_list, cad_crop_bbox_list = getCropPointArrayList(
                sample_cad_points, crop_patch_size, crop_patch_move_step,
                crop_min_point_num)

            if len(cad_crop_points_list) > 0:
                self.points_list += cad_crop_points_list
                self.bbox_list += cad_crop_bbox_list
                cad_crop_occ_list = getPointArrayOccListWithPool(
                    cad_crop_points_list,
                    cad_crop_bbox_list,
                    print_progress=print_progress)
                self.occ_list += cad_crop_occ_list
                occ_dict['cad_points_list'] = cad_crop_points_list
                occ_dict['cad_bbox_list'] = cad_crop_bbox_list
                occ_dict['cad_occ_list'] = cad_crop_occ_list

            pickle.dump(occ_dict, open(occ_file_path, "wb"))
            save_idx += 1
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
                self.points_list[index].reshape(1, -1, 3).astype(
                    np.float32)).transpose(2, 1)
            data['inputs']['bbox'] = torch.tensor(
                self.bbox_list[index].toArray().reshape(1, 2,
                                                        3).astype(np.float32))
            data['inputs']['occ'] = torch.tensor(self.occ_list[index].reshape(
                1, 1, 32, 32, 32).astype(np.float32))
        else:
            data['inputs']['point_array'] = torch.tensor(
                self.points_list[index].astype(np.float32)).transpose(1, 0)
            data['inputs']['bbox'] = torch.tensor(
                self.bbox_list[index].toArray().astype(np.float32))
            data['inputs']['occ'] = torch.tensor(self.occ_list[index].reshape(
                1, 32, 32, 32).astype(np.float32))
        return data

    def __len__(self):
        return len(self.points_list)
