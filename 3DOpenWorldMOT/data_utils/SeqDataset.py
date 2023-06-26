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
 

class MOT3DSeqDataset:
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
        timestamps2frame = {int(t[:-8]): frame for frame, t in enumerate(
            sorted(os.listdir(osp.join(self.gt_path, self.seq_name, 'sensors', 'lidar'))))}
        
        # get frames from sorted timestamps for detections
        for _, dets in self.dets.items():
            for det in dets:
                det.frame = timestamps2frame[det.timestamps[0, 0].item()]

        # get frames from sorted timestamps for ground truth
        if not 'test' in self.dataset_path:
            frames = [timestamps2frame[t] for t in self.gts['timestamp_ns']]
            self.gts['frames'] = frames

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
    
    def _assign_gt(self):
        """
        Assigns a GT identity to every detection in self.det_df, based on the ground truth boxes in self.gt_df.
        The assignment is done frame by frame via bipartite matching.
        """
        if not 'test' in self.dataset_path:
            print(f"Assigning ground truth identities to detections to sequence {self.seq_name}")
            for frame in self.dets.keys():
                frame_detects = self.dets[frame]
                frame_gt = self.gt_df[self.gt_df.frame == frame]

                # Compute IoU for each pair of detected / GT bounding box
                det_boxes = self._create_box(
                    np.stack([d.translation for d in frame_detects]),
                    np.stack([d.lwh for d in frame_detects]), 
                    np.array([d.alpha for d in frame_detects]))
                
                gt_boxes = self._create_box(
                    frame_gt[['tx_m', 'ty_m', 'tz_m']].values,
                    frame_gt[['length_m', 'width_m', 'height_m']].values, 
                    np.arccos(frame_gt['qw'].values)*2)
                
                _, iou_matrix = box3d_overlap(
                    det_boxes.cuda(),
                    gt_boxes.cuda(),
                    eps=1e-6)

                iou_matrix[iou_matrix < self.config.gt_assign_min_iou] = np.nan
                dist_matrix = 1 - iou_matrix
                assigned_detect_ixs, assigned_detect_ixs_ped_ids = solve_dense(dist_matrix)
                unassigned_detect_ixs = np.array(list(set(range(frame_detects.shape[0])) - set(assigned_detect_ixs)))

                for idx_dt, idx_gt in zip(assigned_detect_ixs, assigned_detect_ixs_ped_ids):
                    self.dets[idx_dt].gt_id_box = frame_gt.iloc[idx_gt]['id'].values
                for idx_dt in unassigned_detect_ixs:
                    self.dets[idx_dt].gt_id_box = -1
    
    @staticmethod
    def _create_box(xyz, lwh, rot):
        '''
        x, y, z = xyz
        l, w, h = lwh

        
        verts = torch.tensor(
            [
                [x - l / 2.0, y - w / 2.0, z - h / 2.0],
                [x + l / 2.0, y - w / 2.0, z - h / 2.0],
                [x + l / 2.0, y + w / 2.0, z - h / 2.0],
                [x - l / 2.0, y + w / 2.0, z - h / 2.0],
                [x - l / 2.0, y - w / 2.0, z + h / 2.0],
                [x + l / 2.0, y - w / 2.0, z + h / 2.0],
                [x + l / 2.0, y + w / 2.0, z + h / 2.0],
                [x - l / 2.0, y + w / 2.0, z + h / 2.0],
            ],
            device=xyz.device,
            dtype=torch.float32,
        )
        '''

        unit_vertices_obj_xyz_m = torch.tensor(
            [
                [- 1, - 1, - 1],
                [+ 1, - 1, - 1],
                [+ 1, + 1, - 1],
                [- 1, + 1, - 1],
                [- 1, - 1, + 1],
                [+ 1, - 1, + 1],
                [+ 1, + 1, + 1],
                [- 1, + 1, + 1],
            ],
            device=xyz.device,
            dtype=torch.float32,
        )

        # Transform unit polygons.
        vertices_obj_xyz_m = (lwh/2.0) * unit_vertices_obj_xyz_m
        vertices_dst_xyz_m = vertices_obj_xyz_m @ rot.T + xyz
        vertices_dst_xyz_m = vertices_dst_xyz_m.type(torch.float32)
        return vertices_dst_xyz_m

    



