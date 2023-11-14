import copy
import glob
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as PyGDataLoader
import os
import torch
from pathlib import Path
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
import numpy as np
from data_utils import point_cloud_handling
from data_utils import av2_classes
import os.path as osp
from av2.map.map_api import ArgoverseStaticMap, RasterLayerType
import logging
import csv
import glob
import re
from multiprocessing.pool import Pool
from functools import partial
import pytorch3d.ops.points_normals as points_normals
from pyarrow import feather
import av2.utils.io as io_utils
from .splits import get_seq_list, get_seq_list_fixed_val
from torch import multiprocessing as mp



logger = logging.getLogger("Model.Dataset")


ARGOVERSE_CLASSES = {v: k for k, v in av2_classes._class_dict.items()}
WAYMO_CLASSES = {'TYPE_UNKNOWN': 0, 'TYPE_VECHICLE': 1, 'TYPE_PEDESTRIAN': 2, 'TYPE_SIGN': 3, 'TYPE_CYCLIST': 4}


class TrajectoryDataset(PyGDataset):
    def __init__(
            self,
            data_dir,
            split,
            trajectory_dir,
            edge_dir,
            use_all_points,
            num_points,
            remove_static,
            static_thresh,
            debug,
            every_x_frame=1,
            margin=0.6,
            _processed_dir=False,
            do_process=True,
            percentage_data=1,
            detection_set='train_gnn',
            filtered_file_path=None,
            detection_out_path=None,
            get_vels=False, 
            vel_augment=False,
            remove_non_move_thresh=1,
            traj_channels=25,
            pos_channels=3,
            waymo_style=False,
            cd_filter=0.2):
        
        if 'gt' in _processed_dir:
            self.trajectory_dir = Path(os.path.join(trajectory_dir, split))
        else:
            self.trajectory_dir = Path(os.path.join(trajectory_dir))
        self.data_dir = data_dir
        self.remove_static = remove_static
        self.static_thresh = static_thresh
        self._trajectory_dir = trajectory_dir
        self.split = split
        self.use_all_points = use_all_points
        self.num_points = num_points
        if self.split == 'train' or self.split == 'val' or self.split == 'test':
            self.loader = AV2SensorDataLoader(
                data_dir=Path(os.path.join(data_dir, split)),
                labels_dir=Path(os.path.join(data_dir, split)))
        else:
            self.loader = None
        self.every_x_frame = every_x_frame
        self.margin = margin
        self._processed_dir = _processed_dir
        self.do_process =  do_process
        self.percentage_data = percentage_data
        self.filtered_file_path = filtered_file_path
        self.edge_dir = edge_dir
        self.get_vels = get_vels
        self.vel_augment = vel_augment
        self.remove_non_move_thresh = remove_non_move_thresh
        self.traj_channels = traj_channels
        self.pos_channels = pos_channels
        self.waymo_style = waymo_style
        self.cd_filter = cd_filter
        print(f"USING WAYMO STYLE {self.waymo_style}")
        # for debugging
        if debug:
            if split == 'val' and 'Argo' in self.data_dir:
                self.seqs = ['04994d08-156c-3018-9717-ba0e29be8153']
            elif split == 'train' and 'Argo' in self.data_dir:
                self.seqs = ['00a6ffc1-6ce9-3bc3-a060-6006e9893a1a']
            elif split == 'val':
                self.seqs = ['10023947602400723454'] #['16473613811052081539']
            else:
                self.seqs = ['10023947602400723454'] # ['2400780041057579262']
            self.seqs = ['13310437789759009684']
        else:
            # self.seqs = get_seq_list(
            self.seqs = get_seq_list_fixed_val(
                path=os.path.join(data_dir, split),
                detection_set=detection_set,
                percentage=self.percentage_data)
        if 'detector' in detection_set:
            split = 'train' if 'train' in detection_set else 'val'
            self.already_evaluated = list()
            print(f'{detection_out_path}/{detection_set}')
            self.already_evaluated = [os.path.basename(os.path.dirname(p)) for p in glob.glob(f'{detection_out_path}/*/{detection_set}/*/*')]
        else:
            self.already_evaluated = list()
        self.class_dict = ARGOVERSE_CLASSES if 'Argo' in self.data_dir else WAYMO_CLASSES
        self._process()
        super().__init__()
        
    @property
    def raw_file_names(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [flow_file for seq in seqs\
            for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq, 'sensors', 'lidar')))) \
                if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
            
    @property
    def raw_paths(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [os.path.join(self.trajectory_dir, seq, flow_file)\
            for seq in seqs\
                for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq, 'sensors', 'lidar')))) \
                    if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
    
    @property
    def processed_file_names(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [flow_file[:-3] + 'pt' for seq in seqs\
            for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq, 'sensors', 'lidar')))) \
                if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
            
    @property
    def processed_paths(self):
        if not self.do_process:
            seqs = [seq for seq in self.seqs if seq in os.listdir(self.processed_dir) and seq not in self.already_evaluated]
            print(f'{len(seqs)} to be evaluated, {len(self.already_evaluated)} were already there...')
            return [os.path.join(self.processed_dir, seq, 'sensors', 'lidar', flow_file)\
                    for seq in seqs\
                        for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.processed_dir, seq, 'sensors', 'lidar'))))\
                        if i % self.every_x_frame == 0 and i < len(os.listdir(osp.join(self.processed_dir, seq, 'sensors', 'lidar')))-5]#[:64]
        else:
            seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir) and seq not in self.already_evaluated]
            return [os.path.join(self.processed_dir, seq, flow_file[:-3] + 'pt')\
                    for seq in seqs\
                        for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq, 'sensors', 'lidar'))))\
                             if i % self.every_x_frame == 0]

    def __len__(self):
        return len(self._processed_paths)
    
    def len(self):
        return self.__len__()

    @property
    def processed_dir(self) -> str:
        if 'gt' in self._processed_dir:
            return Path(os.path.join(self._processed_dir, self.split))
        return Path(os.path.join(self._processed_dir))

    def add_margin(self, label):
        # Add a margin to the cuboids. Some cuboids are very tight 
        # and might lose some points.
        if self.margin:
            label.length_m += self.margin
            label.width_m += self.margin
            label.height_m += self.margin
        return label
    
    def _process(self):
        self._processed_paths = self.processed_paths
        return

    def get(self, idx):
        path = self._processed_paths[idx]
        data = feather.read_feather(path)
        pos = data[np.logical_and(4*data['cd_dist_4']>0.2, data['non_ground_pts_rc'])]
        pos = pos[['x', 'y', 'z']].values
        traj = torch.ones([pos.shape[0], 24, 3])
        traj[:, 1, 0] = 1
        timestamps = np.array([int(path.split('/')[-1][:-8])])
        point_categories_mov = torch.ones(pos.shape[0], dtype=int)
        point_instances_mov = torch.ones(pos.shape[0], dtype=int)
        point_categories = torch.ones(pos.shape[0], dtype=int)
        point_instances = torch.ones(pos.shape[0], dtype=int)
        seq = path.split('/')[-4]
        data = PyGData(
                pc_list=torch.from_numpy(pos),
                traj=traj,
                timestamps=torch.from_numpy(timestamps),
                point_categories_mov=point_categories_mov,
                point_instances_mov=point_instances_mov,
                point_categories=point_categories,
                point_instances=point_instances,
                log_id=seq,
                # pc_normals=pc_normals
		)

        if self.waymo_style:
            mask = torch.logical_and(
                torch.logical_and(data['pc_list'][:, 0] < 50, data['pc_list'][:, 0] > -50),
                torch.logical_and(data['pc_list'][:, 1] < 20, data['pc_list'][:, 1] > -20))
            data['traj'] = data['traj'][mask]
            data['pc_list'] = data['pc_list'][mask]
            data['point_instances'] = data['point_instances'][mask]
            data['point_categories'] = data['point_categories'][mask]
            if 'point_categories_mov' in data.keys:
                data['point_instances_mov'] = data['point_instances_mov'][mask]
                data['point_categories_mov'] = data['point_categories_mov'][mask]

        # if you always want same number of points (during training), sample/re-sample
        if not self.use_all_points and data['traj'].shape[0] > self.num_points:
            idxs = torch.randint(0, data['traj'].shape[0], size=(self.num_points, ))
            mask = torch.arange(data['traj'].shape[0])
            mask = torch.isin(mask, idxs)

            data['pc_list'] = data['pc_list'][idxs, :]
            data['traj'] = data['traj'][idxs]
            data['point_categories'] = data['point_categories'][idxs]
            data['point_instances'] = data['point_instances'][idxs]
            if 'point_categories_mov' in data.keys:
                data['point_categories_mov'] = data['point_categories_mov'][idxs]
                data['point_instances_mov'] = data['point_instances_mov'][idxs]

        data['batch'] = torch.ones(data['pc_list'].shape[0])*idx
        data['timestamps'] = data['timestamps'].unsqueeze(0)
        data['path'] = path

        return data


def get_TrajectoryDataLoaderOrig(cfg, name=None, train=True, val=True, test=False):
    if 'graph_construction' in cfg.models.hyperparams.keys():
        graph_dir = cfg.data.trajectory_dir.split('/')
        graph_dir[-1] = graph_dir[-1] + f'_{cfg.models.hyperparams.graph_construction}'
        graph_dir[-2] = graph_dir[-2] + f'_{cfg.models.hyperparams.graph_construction}'
        graph_dir = '/'.join(graph_dir)
    else:
        graph_dir = None
    # get datasets
    if train and not cfg.just_eval:
        train_data = TrajectoryDataset(cfg.data.data_dir + f'_train/' + os.path.basename(cfg.data.data_dir) if 'Argo' not in cfg.data.data_dir else cfg.data.data_dir,
            'train',
            cfg.data.data_dir + '_train',
            graph_dir,
            cfg.data.use_all_points,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug,
            do_process=cfg.data.do_process,
            _processed_dir=cfg.data.data_dir + '_train',
            percentage_data=cfg.data.percentage_data_train,
            filtered_file_path=cfg.data.filtered_file_path,
            get_vels=cfg.data.get_vels,
            vel_augment=cfg.data.vels_augment,
            remove_non_move_thresh=cfg.data.remove_static_thresh,
            traj_channels=cfg.data.traj_channels,
            pos_channels=cfg.data.pos_channels,
            waymo_style=cfg.data.waymo_style)
    else:
        train_data = None
    if val:
        if 'evaluation' in cfg.data.detection_set:
            split = 'val'
        else:
            split = 'train'
        val_data = TrajectoryDataset(cfg.data.data_dir + f'_{split}/' + os.path.basename(cfg.data.data_dir) if 'Argo' not in cfg.data.data_dir else cfg.data.data_dir,
                split,
                cfg.data.data_dir + f'_{split}/{split}',
                graph_dir,
                # '/workspace/result/all_egocomp_margin0.6_width25_min_mean_max_vel',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                every_x_frame=cfg.data.every_x_frame,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.data_dir + f'_{split}/{split}', 
                percentage_data=cfg.data.percentage_data_val,
                detection_set=cfg.data.detection_set,
                filtered_file_path=cfg.data.filtered_file_path,
                detection_out_path=name,
                get_vels=cfg.data.get_vels,
                traj_channels=cfg.data.traj_channels,
                pos_channels=cfg.data.pos_channels,
                waymo_style=cfg.data.waymo_style,
                cd_filter=cfg.data.cd_filter)
    else:
        val_data = None
    if test:
        split = 'test'
        test_data = TrajectoryDataset(cfg.data.data_dir + f'_{split}/' + os.path.basename(cfg.data.data_dir) if 'Argo' not in cfg.data.data_dir else cfg.data.data_dir,
                split,
                cfg.data.trajectory_dir + f'_{split}',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                every_x_frame=cfg.data.every_x_frame,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.processed_dir + f'_{split}', 
                percentage_data=1.0,
                filtered_file_path=cfg.data.filtered_file_path)
    else:
        test_data = None
    
    return train_data, val_data, test_data





















