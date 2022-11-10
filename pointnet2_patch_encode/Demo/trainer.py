#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pointnet2_patch_encode.Module.trainer import Trainer


def demo():
    trainer = Trainer()
    trainer.train()
    return True
