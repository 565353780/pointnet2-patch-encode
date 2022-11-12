#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset


class SimpleOccDataset(Dataset):

    def __init__(self, crop_num=32):
        self.crop_num = crop_num
        self.total_crop_num = self.crop_num**3
        return

    def __getitem__(self, index, test=False):
        assert index < 2**self.total_crop_num

        data = {'inputs': {}, 'predictions': {}, 'losses': {}, 'logs': {}}

        bin_str = bin(index)[2:]
        if len(bin_str) < self.total_crop_num:
            append_zero_num = self.total_crop_num - len(bin_str)
            bin_str = '0' * append_zero_num + bin_str

        occ = np.array([int(single_bin_str) for single_bin_str in bin_str],
                       dtype=int)
        if test:
            data['inputs']['occ'] = torch.tensor(
                occ.reshape(1, 1, self.crop_num, self.crop_num,
                            self.crop_num).astype(np.float32))
        else:
            data['inputs']['occ'] = torch.tensor(
                occ.reshape(1, self.crop_num, self.crop_num,
                            self.crop_num).astype(np.float32))
        return data

    def __len__(self):
        # FIXME: dataset is too large to training
        return int(1e10)
        return 2**self.total_crop_num
