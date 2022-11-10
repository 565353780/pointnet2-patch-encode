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
    model_file_path = "./output/0/model_last.pth"
    print_progress = True

    trainer = Trainer()
    trainer.loadModel(model_file_path)
    trainer.train(print_progress)
    return True


def demo_test():
    model_file_path = "./output/0/model_last.pth"

    trainer = Trainer()
    trainer.loadModel(model_file_path)
    trainer.testScene("scene0474_02")
    #  trainer.test(print_progress)
    return True
