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
 

class MOT3DTrackDataset:
    def __init__(self, dataset_path, gt_path, split):
        self.dataset_path = dataset_path
        self.split = split
        self.gt_path = gt_path
        self.data = os.listdir(os.path.join(dataset_path, split))

    def __getitem__(self, idx):
        seq_name = self.data[idx]
        return seq_name, self.dataset_path, self.gt_path, self.split

    def __len__(self):
        return len(self.data)

