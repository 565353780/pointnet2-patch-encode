#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import numpy as np

from pointnet2_patch_encode.Method.path import createFileFolder


def toJson(data):
    json_dict = {}
    for first_key in data.keys():
        json_dict[first_key] = {}
        for key, value in data[first_key].items():
            if isinstance(value, np.ndarray):
                json_dict[first_key][key] = value.tolist()
                continue

            if isinstance(value, str) or isinstance(value, list):
                json_dict[first_key][key] = value
    return json_dict


def saveData(data, save_json_file_path):
    createFileFolder(save_json_file_path)

    json_dict = toJson(data)

    with open(save_json_file_path, "w") as f:
        json.dump(json_dict, f, indent=4)
    return True


def saveDataList(data_list, save_json_file_path):
    save_dict = {}
    for i in range(len(data_list)):
        save_dict[str(i)] = toJson(data_list[i])

    createFileFolder(save_json_file_path)

    with open(save_json_file_path, "w") as f:
        json.dump(save_dict, f)
    return True


def loadData(json_file_path):
    assert os.path.exists(json_file_path)

    with open(json_file_path, "r") as f:
        data = json.load(f)
    return data


def loadDataList(json_file_path):
    assert os.path.exists(json_file_path)

    with open(json_file_path, "r") as f:
        data = json.load(f)
    return list(data.values())
