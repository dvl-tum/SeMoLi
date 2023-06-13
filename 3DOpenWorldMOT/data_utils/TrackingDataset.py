"""
Process the sequences from the detection/gt file. Moreover, if they are already processed, loads the corresponding
dataframes.
"""

import numpy as np
import os
import os.path as osp
import pandas as pd
import torch
from lapsolver import solve_dense

import time
from pytorch3d.ops import box3d_overlap
from models.tracking_utils import load_initial_detections, load_gt
 

class MOT3DTrackDataset:
    def __init__(self, dataset_path, gt_path, seq_name, split, tracks=False):
        self.seq_name = seq_name
        self.dataset_path = dataset_path
        self.split = split
        self.gt_path = gt_path
        self._load_detections(tracks, self.dataset_path, self.split, self.seq_name)

    def _add_extra_det_features(self):
        """
        Create additional features for each detection. (e.g bbox centre, area etc.)
        """
        # get timestamps2frame dict that matches timestamps to frames
        timestamps2frame = {t: frame for frame, t in enumerate(
            sorted(os.listdir(osp.join(self.gt_path, 'sensors', 'lidar'))))}

        # get frames from sorted timestamps for detections
        for _, dets in self.dets.items():
            for det in dets:
                det.frame = timestamps2frame[dets.timestamps[0, 0].item()]

        # get frames from sorted timestamps for ground truth
        if not 'test' in self.dataset_path:
            frames = [timestamps2frame[t] for t in self.gt_df['timestamp_ns']]
            self.gt_df['frames'] = frames

    def _load_detections(self, tracks, load_path, split, log_id):
        """
        Load a pd.Dataframe with each entry corresponding to a detection. Same for the ground truth file.
        """
        # Read the dfs
        self.dets = load_initial_detections(load_path, split, log_id, tracks=tracks, every_x_frame=1, overlap=1)
        if not 'test' in self.dataset_path:
            self.gts = load_gt(self.seq_name, self.gt_path)

        # Add extra measurements
        self._add_extra_det_features()



