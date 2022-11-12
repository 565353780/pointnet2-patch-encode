#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch
import random
import numpy as np
from tqdm import tqdm
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from pointnet2_patch_encode.Model.class_msg import ClassMsg

from pointnet2_patch_encode.Dataset.occ_dataset import OccDataset

from pointnet2_patch_encode.Method.device import toCuda, toCpu, toNumpy
from pointnet2_patch_encode.Method.time import getCurrentTime
from pointnet2_patch_encode.Method.path import createFileFolder, renameFile, removeFile
from pointnet2_patch_encode.Method.io import saveDataList, loadDataList
from pointnet2_patch_encode.Method.render import renderDataList

from pointnet2_patch_encode.Module.dataset_manager import DatasetManager


def _worker_init_fn_(worker_id):
    torch_seed = torch.initial_seed()
    np_seed = torch_seed // 2**32 - 1
    random.seed(torch_seed)
    np.random.seed(np_seed)
    return True


class Trainer(object):

    def __init__(self):
        scannet_dataset_folder_path = "/home/chli/chLi/ScanNet/scans/"
        scannet_object_dataset_folder_path = "/home/chli/chLi/ScanNet/objects/"
        scannet_bbox_dataset_folder_path = "/home/chli/chLi/ScanNet/bboxes/"
        scan2cad_dataset_folder_path = "/home/chli/chLi/Scan2CAD/scan2cad_dataset/"
        scan2cad_object_model_map_dataset_folder_path = "/home/chli/chLi/Scan2CAD/object_model_maps/"
        shapenet_dataset_folder_path = "/home/chli/chLi/ShapeNet/Core/ShapeNetCore.v2/"
        shapenet_udf_dataset_folder_path = "/home/chli/chLi/ShapeNet/udfs/"

        self.dataset_manager = DatasetManager(
            scannet_dataset_folder_path, scannet_object_dataset_folder_path,
            scannet_bbox_dataset_folder_path, scan2cad_dataset_folder_path,
            scan2cad_object_model_map_dataset_folder_path,
            shapenet_dataset_folder_path, shapenet_udf_dataset_folder_path)

        self.scannet_scene_name = None
        self.scannet_scene_name_list = self.dataset_manager.getScanNetSceneNameList(
        )

        self.model = ClassMsg().cuda()

        self.occ_dataset = OccDataset(self.dataset_manager)
        self.occ_dataloader = DataLoader(self.occ_dataset,
                                         batch_size=24,
                                         shuffle=False,
                                         num_workers=0,
                                         worker_init_fn=_worker_init_fn_)

        self.lr = 1e-4
        self.step = 0
        self.loss_min = float('inf')
        self.log_folder_name = getCurrentTime()
        self.save_result_idx = 0

        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        self.summary_writer = None
        return

    def loadSummaryWriter(self):
        self.summary_writer = SummaryWriter("./logs/" + self.log_folder_name +
                                            "/")
        return True

    def loadModel(self, model_file_path):
        if not os.path.exists(model_file_path):
            self.loadSummaryWriter()
            print("[WARN][Trainer::loadModel]")
            print("\t model_file not exist! start training from step 0...")
            return True

        model_dict = torch.load(model_file_path)

        self.model.load_state_dict(model_dict['class_msg'])
        self.optimizer.load_state_dict(model_dict['optimizer'])
        self.step = model_dict['step']
        self.loss_min = model_dict['loss_min']
        self.log_folder_name = model_dict['log_folder_name']
        self.save_result_idx = model_dict['save_result_idx']

        self.loadSummaryWriter()
        print("[INFO][Trainer::loadModel]")
        print("\t load model success! start training from step " +
              str(self.step) + "...")
        return True

    def saveModel(self, save_model_file_path):
        model_dict = {
            'class_msg': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'step': self.step,
            'loss_min': self.loss_min,
            'log_folder_name': self.log_folder_name,
            'save_result_idx': self.save_result_idx,
        }

        createFileFolder(save_model_file_path)

        tmp_save_model_file_path = save_model_file_path.split(
            ".pth")[0] + "_tmp.pth"

        torch.save(model_dict, tmp_save_model_file_path)

        removeFile(save_model_file_path)
        renameFile(tmp_save_model_file_path, save_model_file_path)
        return True

    def loadScene(self, scannet_scene_name, print_progress=False):
        assert scannet_scene_name in self.scannet_scene_name_list

        if self.scannet_scene_name == scannet_scene_name:
            return True

        self.scannet_scene_name = scannet_scene_name
        self.occ_dataset.loadScene(self.scannet_scene_name, print_progress)
        return True

    def loadSceneByIdx(self, scannet_scene_name_idx, print_progress=False):
        assert scannet_scene_name_idx <= len(self.scannet_scene_name_list)

        return self.loadScene(
            self.scannet_scene_name_list[scannet_scene_name_idx],
            print_progress)

    def testTrain(self, print_progress=False):
        data = self.occ_dataset.__getitem__(0, True)
        print(data['inputs']['point_array'].shape)
        toCuda(data)
        data = self.model(data)

        for_data = self.occ_dataloader
        if print_progress:
            print("[INFO][Trainer::testTrain]")
            print("\t start test training...")
            for_data = tqdm(for_data)
        for data in for_data:
            toCuda(data)
            data = self.model(data)
        return True

    def trainStep(self, data):
        toCuda(data)

        self.model.train()
        self.model.zero_grad()
        self.optimizer.zero_grad()

        data = self.model(data)

        losses = data['losses']

        losses_tensor = torch.cat([
            loss if len(loss.shape) > 0 else loss.reshape(1)
            for loss in data['losses'].values()
        ])

        loss_sum = torch.sum(losses_tensor)
        loss_sum_float = loss_sum.detach().cpu().numpy()
        self.summary_writer.add_scalar("Loss/loss_sum", loss_sum_float,
                                       self.step)

        if loss_sum_float < self.loss_min:
            self.loss_min = loss_sum_float
            self.saveModel("./output/" + self.log_folder_name +
                           "/model_best.pth")

        for key, loss in losses.items():
            loss_tensor = loss.detach() if len(
                loss.shape) > 0 else loss.detach().reshape(1)
            loss_mean = torch.mean(loss_tensor)
            self.summary_writer.add_scalar("Loss/" + key, loss_mean, self.step)

        loss_sum.backward()
        self.optimizer.step()
        return True

    def trainScene(self, scannet_scene_name, print_progress=False):
        self.loadScene(scannet_scene_name)

        for_data = self.occ_dataloader
        if print_progress:
            for_data = tqdm(for_data)
        for data in for_data:
            self.trainStep(data)
            self.step += 1

        self.saveModel("./output/" + self.log_folder_name + "/model_last.pth")
        #  self.saveResult(print_progress)
        return True

    def trainEpoch(self, global_epoch_idx, global_epoch, print_progress=False):
        scannet_scene_name_list = self.dataset_manager.getScanNetSceneNameList(
        )
        for scannet_scene_name_idx, scannet_scene_name in enumerate(
                scannet_scene_name_list):

            # FIXME: for test only
            #  scannet_scene_name = "scene0474_02"

            if print_progress:
                print("[INFO][Trainer::trainScene]")
                print("\t start train on scene " + scannet_scene_name +
                      ", epoch: " + str(global_epoch_idx + 1) + "/" +
                      str(global_epoch) + ", scene: " +
                      str(scannet_scene_name_idx + 1) + "/" +
                      str(len(scannet_scene_name_list)) + "...")
            self.trainScene(scannet_scene_name, print_progress)
        return True

    def train(self, print_progress=False):
        global_epoch = 10000000

        for global_epoch_idx in range(global_epoch):
            self.trainEpoch(global_epoch_idx, global_epoch, print_progress)
        return True

    def testScene(self, scannet_scene_name):
        self.model.eval()

        self.occ_dataset.loadScene(scannet_scene_name)

        data_list = []
        for data_idx in range(len(self.occ_dataset)):
            data = self.occ_dataset.__getitem__(data_idx, True)
            data['inputs']['dataset_manager'] = self.dataset_manager
            toCuda(data)
            data = self.model(data)
            toCpu(data)
            toNumpy(data)
            data_list.append(data)

        renderDataList(data_list)
        return True

    def test(self, print_progress=False):
        scannet_scene_name_list = self.dataset_manager.getScanNetSceneNameList(
        )

        for i, scannet_scene_name in enumerate(scannet_scene_name_list):
            if print_progress:
                print("[INFO][Trainer::test]")
                print("\t start test on scene " + scannet_scene_name + ", " +
                      str(i + 1) + "/" + str(len(scannet_scene_name_list)) +
                      "...")
            self.testScene(scannet_scene_name)
        return True

    def saveSceneResult(self,
                        scannet_scene_name,
                        save_json_file_path,
                        print_progress=False):
        self.model.eval()

        self.occ_dataset.loadScene(scannet_scene_name)

        data_list = []

        for_data = range(len(self.occ_dataset))
        if print_progress:
            print("[INFO][Trainer::saveSceneResult]")
            print("\t start save scene result on scene " + scannet_scene_name +
                  " as " + str(self.save_result_idx) + ".json"
                  "...")
            for_data = tqdm(for_data)
        for data_idx in for_data:
            data = self.occ_dataset.__getitem__(data_idx, True)
            data['inputs']['dataset_manager'] = self.dataset_manager
            toCuda(data)
            data = self.model(data)
            toCpu(data)
            toNumpy(data)
            data_list.append(data)

        saveDataList(data_list, save_json_file_path)
        return True

    def saveResult(self, print_progress=False):
        assert self.scannet_scene_name is not None

        save_folder_path = "./output/" + self.log_folder_name + "/result/" + \
            self.scannet_scene_name + "/" + str(self.save_result_idx) + ".json"
        self.saveSceneResult(self.scannet_scene_name, save_folder_path,
                             print_progress)
        self.save_result_idx += 1
        return True
