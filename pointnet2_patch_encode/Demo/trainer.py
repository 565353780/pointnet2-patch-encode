#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

sys.path.append("../mesh-manage/")
sys.path.append("../udf-generate/")
sys.path.append("../scannet-dataset-manage")
sys.path.append("../scan2cad-dataset-manage")
sys.path.append("../shapenet-dataset-manage")
sys.path.append("../auto-cad-recon")

from pointnet2_patch_encode.Module.trainer import Trainer


def demo():
    print_progress = True

    scannet_scene_name = "scene0474_02"

    trainer = Trainer()
    trainer.loadScene(scannet_scene_name)
    trainer.testTrain(print_progress)
    return True
