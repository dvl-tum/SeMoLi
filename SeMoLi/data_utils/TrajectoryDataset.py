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
from SeMoLi.data_utils import point_cloud_handling
from SeMoLi.data_utils import av2_classes
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
from .splits import get_seq_list_fixed_val
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
            use_all_points,
            num_points,
            remove_static,
            static_thresh,
            debug,
            margin=0.6,
            _processed_dir=False,
            do_process=True,
            percentage_data=1,
            detection_set='train_gnn',
            filtered_file_path=None,
            vel_augment=False,
            traj_channels=25,
            pos_channels=3,
            roi_clipping=False,
            root_dir=''):
        
        self.trajectory_dir = Path(os.path.join(trajectory_dir))
        self.data_dir = data_dir
        self.remove_static = remove_static
        self.static_thresh = static_thresh
        self.split = split
        self.use_all_points = use_all_points
        self.num_points = num_points
        self.margin = margin

        self._processed_dir = _processed_dir
        self.do_process =  do_process
        if self.do_process:
            self.loader = AV2SensorDataLoader(
                data_dir=Path(os.path.join(data_dir)),
                labels_dir=Path(os.path.join(data_dir)))
        else:
            self.loader = None

        self.percentage_data = percentage_data
        self.filtered_file_path = os.path.join(root_dir, filtered_file_path)
        self.vel_augment = vel_augment
        self.traj_channels = traj_channels
        self.pos_channels = pos_channels
        self.roi_clipping = roi_clipping
        print(f"USING WAYMO STYLE {self.roi_clipping}")

        self.seqs = get_seq_list_fixed_val(
            path=os.path.join(data_dir, split),
            root_dir=root_dir,
            detection_set=detection_set,
            percentage=self.percentage_data)
        # debugging
        if debug:
            self.seqs = [self.seqs[5]]

        self.class_dict = ARGOVERSE_CLASSES if 'Argo' in self.data_dir else WAYMO_CLASSES
        super().__init__()
        
    @property
    def raw_file_names(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [flow_file for seq in seqs\
            for flow_file in sorted(os.listdir(osp.join(self.trajectory_dir, seq)))]
            
    @property
    def raw_paths(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [os.path.join(self.trajectory_dir, seq, flow_file)\
            for seq in seqs\
                for flow_file in sorted(os.listdir(osp.join(self.trajectory_dir, seq)))]
    
    @property
    def processed_file_names(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [flow_file[:-3] + 'pt' for seq in seqs\
            for flow_file in sorted(os.listdir(osp.join(self.trajectory_dir, seq)))]
            
    @property
    def processed_paths(self):
        if not self.do_process:
            seqs = [seq for seq in self.seqs if seq in os.listdir(self.processed_dir)]
            return [os.path.join(self.processed_dir, seq, flow_file)\
                    for seq in seqs\
                        for flow_file in sorted(os.listdir(osp.join(self.processed_dir, seq)))]
        else:
            seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
            return [os.path.join(self.processed_dir, seq, flow_file[:-3] + 'pt')\
                    for seq in seqs\
                        for flow_file in sorted(os.listdir(osp.join(self.trajectory_dir, seq)))]

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
        if self.do_process:
            logger.info('Processing...')
            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()
            logger.info('Done!')
    
    def process(self, multiprocessing=True):
        self._raw_paths = self.raw_paths
        data_loader = enumerate(self._raw_paths)
        self.len_raw_paths = len(self._raw_paths)
        if multiprocessing:
            # mp.set_start_method('forkserver')
            # with mp.Pool() as pool:
            #     pool.map(self.process_sweep, data_loader, chunksize=None)
            with Pool() as pool:
                _eval_sequence = partial(self.process_sweep)
                pool.map(_eval_sequence, data_loader)
        else:
            for data in data_loader:
                self.process_sweep(data)
                
    def process_sweep(self, data):
        j, traj_file = data

        if j % 100 == 0:
            logger.info(f"sweep {j}/{self.len_raw_paths}, {j}-th file")
        
        # Get store path
        processed_path = os.path.join(
            self.processed_dir,
            os.path.basename(os.path.dirname(traj_file)),
            os.path.basename(traj_file)[:-3] + 'pt')
        seq = os.path.basename(os.path.dirname(traj_file))

        # load point clouds
        pred = np.load(traj_file)
        pc_list = pred['pcs'] if 'pcs' in [k for k in pred.keys()] else pred['pc_list']
        timestamps = pred['timestamps']

        if len(pc_list.shape) > 2:
            pc_list = pc_list[0]

        if len(pred['timestamps'].shape) > 1:
            timestamps = timestamps[0]

        # get labels
        if self.split != 'test':
            # get all labels and moving labels
            labels = self.loader.get_labels_at_lidar_timestamp(
                log_id=seq, lidar_timestamp_ns=int(timestamps[0]))

            filtered_file_path = f'{self.filtered_file_path}/{self.split}_1_per_frame_remove_non_move_remove_far_filtered_version_city_w0.feather'
            labels_mov = self.loader.get_labels_at_lidar_timestamp_all(
                filtered_file_path, log_id=seq, lidar_timestamp_ns=int(timestamps[0]), get_moving=True)
            
            # if argoverse bounding boxes increase slightly
            if self.margin and 'Argo' in self.data_dir:
                for label in labels:
                    self.add_margin(label)
            
            # All points: get per point and object masks and bounding boxs and their labels 
            point_categories, point_instances = self.get_point_instance_and_class(labels, pc_list)

            # Moving points: get per point and object masks and bounding boxs and their labels 
            point_categories_mov, point_instances_mov = self.get_point_instance_and_class(labels_mov, pc_list)
            
            # get object velocities
            velocities = self.get_object_velocities(seq, timestamps, pc_list)

            # putting it all together
            data = PyGData(
                point_categories_mov=point_categories_mov,
                point_instances_mov=point_instances_mov,
                point_categories=point_categories,
                point_instances=point_instances,
                log_id=seq,
                velocities=velocities
		)
        else:
            data = PyGData(
                log_id=seq,
                # pc_normals=pc_normals
		)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        torch.save(data, processed_path)

    def get_point_instance_and_class(self, labels, pc_list):
        # get per point and object masks and bounding boxs and their labels 
        masks = list()
        for label in labels:
            interior = point_cloud_handling.compute_interior_points_mask(
                    pc_list, label.vertices_m)
            int_label = self.class_dict[label.category] if 'Argo' in self.data_dir else int(label.category)
            interior = interior.astype(int) * int_label
            masks.append(interior)

        if len(labels) == 0:
            masks.append(np.zeros(pc_list.shape[0]))
        
        masks = np.asarray(masks).T
    
        # assign unique label and instance to each point
        # label 0 and instance 0 is background
        point_categories = list()
        point_instances = list()
        for j in range(masks.shape[0]):
            if np.where(masks[j]>0)[0].shape[0] != 0:
                point_categories.append(masks[j, np.where(masks[j]>0)[0][0]])
                point_instances.append(np.where(masks[j]>0)[0][0]+1)
            else:
                point_categories.append(0)
                point_instances.append(0)

        point_instances = np.asarray(point_instances, dtype=np.int64)
        point_categories = np.asarray(point_categories, dtype=np.int64)

        point_categories=torch.atleast_2d(torch.from_numpy(point_categories).squeeze())
        point_instances=torch.atleast_2d(torch.from_numpy(point_instances).squeeze())

        return point_categories, point_instances
    
    def _remove_static(self, data):
        # remove static points
        mean_traj = data['traj'][:, :, :-1]
        timestamps = data['timestamps']
        # get mean velocity [m/s] along trajectory and check if > thresh
        diff_dist = torch.linalg.norm(
            mean_traj[:, 1, :] - mean_traj[:, 0, :] , axis=1)

        diff_time = timestamps[1] - timestamps[0]
        # bring from nano / mili seconds to seconds
        if 'Argo' in self.data_dir:
            diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
        else:
            diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)

        mean_traj = diff_dist/diff_time
        mean_traj = mean_traj > self.static_thresh

        # if no moving point and not evaluation, sample few random
        if torch.all(~mean_traj):
            idxs = torch.randint(0, mean_traj.shape[0], size=(200, ))
            mean_traj[idxs] = True
        
        # apply mask
        data['pc_list'] = data['pc_list'][mean_traj, :]
        data['traj'] = data['traj'][mean_traj]
        if 'pc_normals' in data.keys:
            data['pc_normals'] = data['pc_normals'][mean_traj]
        
        data['point_instances'] = data['point_instances'].squeeze()[mean_traj]
        data['point_categories'] = data['point_categories'].squeeze()[mean_traj]
        if 'point_categories_mov' in data.keys:
            data['point_instances_mov'] = data['point_instances_mov'].squeeze()[mean_traj]
            data['point_categories_mov'] = data['point_categories_mov'].squeeze()[mean_traj]
        
        if 'edge_index' in data.keys:
            nodes = torch.where(mean_traj)[0]
            data['edge_index'] = data['edge_index'][torch.logical_and(
                data['edge_index'][0, :].isin(nodes),
                data['edge_index'][1, :].isin(nodes))]
        return data
    
    def get_object_velocities(self, log_id, timestamps, pc_list):
        # get labels and SE3 transformation at t
        labels = self.loader.get_labels_at_lidar_timestamp(
            log_id=log_id, lidar_timestamp_ns=timestamps[0].item())
        city_SE3_t1 = self.loader.get_city_SE3_ego(
            log_id, timestamps[0].item())
        
        # labels, SE3 transformation, and ids at t+1
        labels_t2 = self.loader.get_labels_at_lidar_timestamp(
            log_id=log_id, lidar_timestamp_ns=timestamps[1].item())
        city_SE3_t2 = self.loader.get_city_SE3_ego(
            log_id, timestamps[1].item())
        ids_t2 = [label.track_id for label in labels_t2]

        velocities = torch.zeros(pc_list.shape[0])
        ego_traj_SE3_ego_ref = city_SE3_t2.inverse().compose(city_SE3_t1)
        for m, label in enumerate(labels):
            # get object center
            center = label.dst_SE3_object.translation

            # if id of label also t2
            if len(labels_t2) and label.track_id in ids_t2:
                # Pose of the object in the destination reference frame.
                # ego_SE3_object --> from object to ego   
                ego_traj_SE3_obj_traj = labels_t2[ids_t2.index(
                    label.track_id)].dst_SE3_object
                ego_ref_SE3_obj_ref = label.dst_SE3_object

                # transform points belonging to object in t1 into ego t2 coordinate system
                obj_ref_ego_traj = ego_traj_SE3_ego_ref.transform_point_cloud(
                    center)
                
                # transform points belonging to object in t1 into obj and from obj to ego t2 (assmue obj same points)
                obj_traj_ego_traj = ego_traj_SE3_obj_traj.compose(
                    ego_ref_SE3_obj_ref.inverse()).transform_point_cloud(center)

                # get flow
                translation = obj_traj_ego_traj - obj_ref_ego_traj
                dist = np.linalg.norm(translation)
                if 'Argo' in self.data_dir:
                    diff_time = (
                        timestamps[1]-timestamps[0]) / np.power(10, 9)
                else:
                    diff_time = (
                        timestamps[1]-timestamps[0]) / np.power(10, 6)
                vel = dist/diff_time
                interior = torch.from_numpy(point_cloud_handling.compute_interior_points_mask(
                        pc_list, label.vertices_m))
                velocities[interior] = vel

        return velocities

    def velocity_augment(self, data):
        point_instances_mov_new = copy.deepcopy(data['point_instances'])
        for instance in data['point_instances'].unique():
            # don't augment background velocity
            if instance == 0:
                continue

            # should we augment and if yes which scale
            do_augment = torch.randint(2, (1,))
            # scale*10*np.exp(-torch.linalg.norm(data['traj'][instance_mask][:, 1, :-1].mean(dim=0)))
            scale = torch.rand(1)*10
            instance_mask = (data['point_instances'] == instance).squeeze()
            if not do_augment:
                if data['point_instances_mov'][:, instance_mask].sum() == 0:
                    point_instances_mov_new[:, instance_mask] = 0
                continue
            # augment and adapt moving mask
            data['traj'][instance_mask] = data['traj'][instance_mask] * scale.to(data['traj'].device)
            dist = torch.linalg.norm(data['traj'][instance_mask][:, 1, :-1].mean(dim=0))
            if 'Argo' in self.data_dir:
                diff_time = (
                    data['timestamps'][1]-data['timestamps'][0]) / np.power(10, 9)
            else:
                diff_time = (
                    data['timestamps'][1]-data['timestamps'][0]) / np.power(10, 6)
            vel = dist/diff_time
            if vel < self.static_thresh:
                point_instances_mov_new[:, instance_mask] = 0
        data['point_instances_mov'] = point_instances_mov_new
        
        return data

    def filter_waymo(self, data):
        if self.roi_clipping:
            mask = torch.logical_and(
                torch.logical_and(data['pc_list'][:, 0] < 50, data['pc_list'][:, 0] > -50),
                torch.logical_and(data['pc_list'][:, 1] < 20, data['pc_list'][:, 1] > -20))
            data['traj'] = data['traj'][mask]
            data['pc_list'] = data['pc_list'][mask]
            data['point_instances'] = data['point_instances'][:, mask]
            data['point_categories'] = data['point_categories'][:, mask]
            data['velocities'] = data['velocities'][mask]
            if 'point_categories_mov' in data.keys:
                data['point_instances_mov'] = data['point_instances_mov'][:, mask]
                data['point_categories_mov'] = data['point_categories_mov'][:, mask]
        else:
            self.filter_far_high(data)
        return data

    def filter_far_high(self, data):
        mask = torch.logical_and(
            torch.norm(data['pc_list'][:, :-1], p=2, dim=1) <= 80,
            data['pc_list'][:, -1] <= 4)
        data['traj'] = data['traj'][mask]
        data['pc_list'] = data['pc_list'][mask]
        data['point_instances'] = data['point_instances'][:, mask]
        data['point_categories'] = data['point_categories'][:, mask]
        data['velocities'] = data['velocities'][mask]
        if 'point_categories_mov' in data.keys:
            data['point_instances_mov'] = data['point_instances_mov'][:, mask]
            data['point_categories_mov'] = data['point_categories_mov'][:, mask]
        return data

    def load(self, path):
        # load ground truth
        data = torch.load(path)
        
        # load timstamps, trajectory and point cloud
        traj_file = os.path.join(
            self.trajectory_dir,
            os.path.basename(os.path.dirname(path)),
            os.path.basename(path)[:-2] + 'npz')
        pred = np.load(traj_file)
        traj = pred['traj']
        pc_list = pred['pc_list']        
        data['pc_list'] = torch.from_numpy(pc_list)
        data['traj'] = torch.from_numpy(traj)
        data['timestamps'] = torch.from_numpy(pred['timestamps'])

        # only take number of timestamps for trajectoy wanted and position 
        data['traj'] = data['traj'][:, :self.traj_channels, :self.pos_channels]
        data['pc_list'] = data['pc_list'][:, :self.pos_channels]
        data['timestamps'] = data['timestamps'][:self.traj_channels]
        
        # ensure at least 3d
        data['point_instances'] = torch.atleast_2d(data['point_instances'])
        data['point_categories'] = torch.atleast_2d(data['point_categories'])

        return data

    def get(self, idx):
        # load data
        path = self._processed_paths[idx]
        data = self.load(path)

        if data['traj'].shape[0] == 0:
            path = self._processed_paths[0]
            data = self.load(path)
        
        # filter points like in waymo
        data = self.filter_waymo(data)
        
        if data['traj'].shape[0] == 0:
            path = self._processed_paths[0]
            data = self.load(path)
            data = self.filter_waymo(data)

        if self.vel_augment:
            data = self.velocity_augment(data)

        if self.remove_static and self.static_thresh > 0:
            data = self._remove_static(data)
        else:
            data['point_categories'] = torch.atleast_1d(data['point_categories'].squeeze())
            data['point_instances'] = torch.atleast_1d(data['point_instances'].squeeze())
            if 'point_categories_mov' in data.keys:
                data['point_categories_mov'] = data['point_categories_mov'].squeeze()
                data['point_instances_mov'] = data['point_instances_mov'].squeeze()

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


def get_TrajectoryDataLoader(cfg, train=True, val=True):
    # get datasets
    if train and not cfg.training.just_eval:
        train_data = TrajectoryDataset(cfg.data.data_dir + '/train',
            'train',
            cfg.data.trajectory_dir + '/train',
            cfg.data.use_all_points,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug,
            do_process=cfg.data.do_process,
            _processed_dir=cfg.data.processed_dir + '/train',
            percentage_data=cfg.data.percentage_data_train,
            filtered_file_path=cfg.data.filtered_file_path,
            vel_augment=cfg.data.vels_augment,
            traj_channels=cfg.data.traj_channels,
            pos_channels=cfg.data.pos_channels,
            roi_clipping=cfg.data.roi_clipping,
            root_dir=cfg.root_dir)
    else:
        train_data = None
    if val:
        if 'evaluation' in cfg.data.detection_set:
            split = 'val'
        else:
            split = 'train'
        val_data = TrajectoryDataset(cfg.data.data_dir + f'/{split}',
                split,
                cfg.data.trajectory_dir+ f'/{split}',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.processed_dir+ f'/{split}', 
                percentage_data=cfg.data.percentage_data_val,
                detection_set=cfg.data.detection_set,
                filtered_file_path=cfg.data.filtered_file_path,
                traj_channels=cfg.data.traj_channels,
                pos_channels=cfg.data.pos_channels,
                roi_clipping=cfg.data.roi_clipping,
                root_dir=cfg.root_dir)
    else:
        val_data = None
    
    return train_data, val_data




