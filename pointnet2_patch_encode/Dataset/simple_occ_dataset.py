#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


class SimpleOccDataset(Dataset):

    def __init__(self, crop_num=32):
        self.crop_num = crop_num
        self.total_crop_num = self.crop_num**3
        self.dataset_len = 100000
        return

    def __getitem__(self, index, test=False):
        assert index < self.dataset_len

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        occ = np.random.randint(2,
                                size=(1, self.crop_num, self.crop_num,
                                      self.crop_num))
        if test:
            data['inputs']['occ'] = torch.tensor(
                occ.reshape(1, 1, self.crop_num, self.crop_num,
                            self.crop_num).astype(np.float32))
        else:
            data['inputs']['occ'] = torch.tensor(occ.astype(np.float32))
        return data

    def __len__(self):
        return self.dataset_len
