"""
Process the sequences from the detection/gt file. Moreover, if they are already processed, loads the corresponding
dataframes.
"""

import numpy as np
import os
import os.path as osp
import pandas as pd
import torch

import time
from pytorch3d.ops import box3d_overlap
from models.tracking_utils import load_initial_detections, load_gt
from .splits import get_seq_list, get_seq_list_fixed_val


class MOT3DTrackDataset:
    def __init__(self, dataset_path, gt_path, detection_set, percentage_data, debug):
        self.dataset_path = dataset_path
        self.gt_path = gt_path
        self.detection_set = detection_set
        self.percentage_data = percentage_data
        
        if 'evaluation' in detection_set:
            split = 'val'
        else:
            split = 'train'
        self.split = split
        
        # for debugging
        if debug:
            if split == 'val' and 'Argo' in self.dataset_path:
                self.data = ['04994d08-156c-3018-9717-ba0e29be8153']
            elif split == 'train' and 'Argo' in self.dataset_path:
                self.data = ['00a6ffc1-6ce9-3bc3-a060-6006e9893a1a']
            elif split == 'val':
                self.data = ['10023947602400723454']#['809159138284604331']
            else:
                self.data = ['10023947602400723454'] #['809159138284604331']
        else:
            # self.seqs = get_seq_list(
            self.data = get_seq_list_fixed_val(
                path=os.path.join(dataset_path, split),
                detection_set=detection_set,
                percentage=percentage_data)

    def __getitem__(self, idx):
        seq_name = self.data[idx]
        return seq_name, self.dataset_path, self.gt_path, self.split, self.detection_set, self.percentage_data

    def __len__(self):
        return len(self.data)

