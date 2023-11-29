import os
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import logging
import torch
from .tracking_utils import *


logger = logging.getLogger("Model.Tracker")


class Detector3D():
    def __init__(
            self,
            out_path='out',
            split='val', 
            every_x_frame=1, 
            num_interior=10, 
            overlap=5, 
            av2_loader=None, 
            rank=0,
            precomp_dets=False,
            kNN=0,
            threshold=0.5,
            median_flow=False,
            median_center=False,
            min_area=False,
            root_dir='',
            track_data_path='') -> None:
        
        self.detections = dict()
        self.log_id = -1
        self.split = split
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.num_interior = num_interior
        self.av2_loader = av2_loader
        self.precomp_dets = precomp_dets
        self.rank = rank
        self.name = os.path.basename(out_path)
        self.out_path = os.path.join(out_path,  f'rank_{self.rank}')
        self.threshold = threshold
        self.kNN = kNN
        self.median_flow = median_flow
        self.median_center = median_center
        self.min_area = min_area
        self.root_dir = root_dir
        self.track_data_path = track_data_path

    def new_log_id(self, log_id):
        # save tracks to feather and reset variables
        if self.log_id != -1:
            self.active_tracks = list()
            self.inactive_tracks = list()
            found = self.to_feather()
            if not found:
                logger.info(f'No detections found in {log_id}')

        self.log_id = log_id
        # logger.info(f"New log id {log_id}...")
    
    def get_detections(self, points, traj, clusters, timestamps, log_id,
                       gt_instance_ids, gt_instance_cats, last=False, resampled=None):
        # account for padding in from DistributedTestSampler
        if timestamps.cpu()[0, 0].item() in self.detections.keys():
            if last:
                found = self.to_feather()
                if not found:
                    logger.info(f'No detections found in {log_id}')
            return

        # set new log id
        if self.log_id != log_id:
            self.new_log_id(log_id)

        # iterate over clusters that were found and get detections with their 
        # corresponding flows, trajectories and canonical points
        detections = list()
        if type(clusters) == np.ndarray:
            clusters = torch.from_numpy(clusters).to(self.rank)
        
        if str(gt_instance_ids.device) == 'cpu':
            gt_instance_ids = gt_instance_ids.to(self.rank)
            gt_instance_cats = gt_instance_cats.to(self.rank)
        
        if str(points.device) == 'cpu':
            points = points.to(self.rank)
        
        if str(traj.device) == 'cpu':
            traj = traj.to(self.rank)

        for c in torch.unique(clusters):
            # filter if 'junk' cluster
            if c == -1:
                continue
            num_interior = torch.sum(clusters==c).item()
            gt_id = (torch.bincount(gt_instance_ids[clusters==c]).argmax()).item()
            gt_cat = (torch.bincount(gt_instance_cats[clusters==c]).argmax()).item()

            # get points, bounding boxes
            point_cluster = points[clusters==c]
            # generate new detected trajectory
            traj_cluster = traj[clusters==c]

            if self.kNN > 0:
                point_cluster, mask = outlier_removal(point_cluster, threshold=self.threshold, kNN=self.kNN)
                traj_cluster = traj_cluster[mask]
                resampled_cluster = resampled[mask]
                num_interior = torch.sum(clusters==c).item()
                
            # filter if cluster too small
            if num_interior < max(2, self.num_interior):
                continue

            detections.append(Detection(
                traj_cluster.cpu(),
                point_cluster.cpu(),
                log_id=log_id,
                timestamps=timestamps.cpu(),
                num_interior=num_interior,
                overlap=self.overlap,
                gt_id=gt_id,
                gt_cat=gt_cat, 
                median_flow=self.median_flow,
                median_center=self.median_center,
                min_area=self.min_area))
        
        self.detections[timestamps.cpu()[0, 0].item()] = detections

        if last:
            found = self.to_feather()
            if not found:
                logger.info(f'No detections found in {log_id}')
            
        return detections

    def to_feather(self):
        to_feather(self.detections, self.log_id, self.out_path, self.split, self.rank, self.precomp_dets, self.name, self.root_dir, self.track_data_path)
        self.detections = dict()
        return True

