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
            waymo_style=False):
        
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
        super().__init__()
        
    @property
    def raw_file_names(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [flow_file for seq in seqs\
            for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
            
    @property
    def raw_paths(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [os.path.join(self.trajectory_dir, seq, flow_file)\
            for seq in seqs\
                for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                    if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
    
    @property
    def processed_file_names(self):
        seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir)]
        return [flow_file[:-3] + 'pt' for seq in seqs\
            for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
            
    @property
    def processed_paths(self):
        print('getting processed paths')
        if not self.do_process:
            seqs = [seq for seq in self.seqs if seq in os.listdir(self.processed_dir) and seq not in self.already_evaluated]
            print(f'{len(seqs)} to be evaluated, {len(self.already_evaluated)} were already there...')
            return [os.path.join(self.processed_dir, seq, flow_file)\
                    for seq in seqs\
                        for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.processed_dir, seq))))\
                        if i % self.every_x_frame == 0]#[:64]
        else:
            seqs = [seq for seq in self.seqs if seq in os.listdir(self.trajectory_dir) and seq not in self.already_evaluated]
            return [os.path.join(self.processed_dir, seq, flow_file[:-3] + 'pt')\
                    for seq in seqs\
                        for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq))))\
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
        if self.do_process:
            logger.info('Processing...')
            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()
            logger.info('Done!')
    
    def process(self, multiprocessing=True):
        # only process what is not there yet
        already_processed = glob.glob(str(self.processed_dir)+'/*/*')
        # already_processed = list()
        missing_paths = set(self._processed_paths).difference(already_processed)
        missing_paths = [os.path.join(
            self.trajectory_dir, os.path.basename(os.path.dirname(m)), os.path.basename(m)[:-2] + 'npz')\
                for m in missing_paths]
        
        logger.info(f"Already processed {len(already_processed)},\
                    Missing {len(missing_paths)},\
                    In total {len(self._processed_paths)}")
        self.len_missing = len(missing_paths)

        if self.len_missing and self.loader is None:
            self.loader = AV2SensorDataLoader(
                data_dir=Path(os.path.join(self.data_dir, self.split)),
                labels_dir=Path(os.path.join(self.data_dir, self.split)))

        data_loader = enumerate(missing_paths)            
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
        
    def load_initial_pc(
            self, 
            lidar_fpath: Path,
            index=[0, 1],
            laser_id=[0, 1, 2, 3, 4],
            remove_height=True,
            remove_far=True,
            remove_static=True,
            remove_ground_pts_rc=True,
            remove_non_dirvable=False,
            num_cd_dist_frames=4):
        """Get the lidar sweep from the given sweep_directory.
    ​
        Args:
            sweep_directory: path to middle lidar sweep.
            sweep_index: index of the middle lidar sweep.
            width: +/- lidar scans to grab.
        Returns:
            List of plys with their associated pose if all the sweeps exist.
        """

        # get sweep information
        sweep_df = io_utils.read_feather(lidar_fpath)

        # get pc
        lidar_points_ego = sweep_df[
                list(['x', 'y', 'z'])].to_numpy().astype(np.float64)

        # get mask to filter point cloud
        mask = np.ones(lidar_points_ego.shape[0], dtype=bool)

        if 'Argo' not in self._processed_dir:            
            mask = np.logical_and(mask, sweep_df['laser_id'].isin(laser_id))
            mask = np.logical_and(mask, sweep_df['index'].isin(index))

        if remove_ground_pts_rc:
            mask = np.logical_and(mask, sweep_df['non_ground_pts_rc'])

        # remove non drivable area points (non RoI points)
        if remove_non_dirvable and 'Argo' in self._processed_dir:
            mask = np.logical_and(mask, sweep_df['driveable_area_pts'])
        
        # Remove points above certain height.
        if remove_height:
            mask = np.logical_and(mask, sweep_df['low_pts'] < 4)

        # Remove points beyond certain distance.
        if remove_far:
            mask = np.logical_and(mask, sweep_df['close_pts'] < 80)
        
        if remove_static:
            for i in range(1, num_cd_dist_frames+1):
                mask = np.logical_and(mask, sweep_df[f'cd_dist_{i}'] > 0.2)

        return lidar_points_ego, mask
                
    def process_sweep(self, data):
        j, traj_file = data

        if j % 100 == 0:
            logger.info(f"sweep {j}/{self.len_missing}, {j}-th file")
        
        processed_path = os.path.join(
            self.processed_dir,
            os.path.basename(os.path.dirname(traj_file)),
            os.path.basename(traj_file)[:-3] + 'pt')
        seq = os.path.basename(os.path.dirname(traj_file))
            
        # load original pc
        # orig_path = os.path.join(
        #     self.data_dir,
        #     self.split,
        #     os.path.basename(os.path.dirname(traj_file)),
        #     'sensors',
        #     'lidar',
        #     os.path.basename(traj_file)[:-3] + 'feather')
        # lidar_points_ego, mask = self.load_initial_pc(orig_path)
        # normals = points_normals.estimate_pointcloud_normals(
        #     torch.from_numpy(lidar_points_ego).cuda().unsqueeze(0)).squeeze()
        # pc_normals = normals[mask]

        # load point clouds
        pred = np.load(traj_file)
        traj = pred['traj']
        pc_list = pred['pcs'] if 'pcs' in [k for k in pred.keys()] else pred['pc_list']
        timestamps = pred['timestamps']

        if len(pc_list.shape) > 2:
            pc_list = pc_list[0]

        if len(pred['timestamps'].shape) > 1:
            timestamps = timestamps[0]

        # get labels
        if self.split != 'test':
            labels = self.loader.get_labels_at_lidar_timestamp(
                log_id=seq, lidar_timestamp_ns=int(timestamps[0]))
            if 'Waymo' in self.data_dir:
                filtered_file_path = f'{self.filtered_file_path}/Waymo_Converted_filtered_{self.split}/{self.split}_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather'
            else:
                filtered_file_path = f'{self.filtered_file_path}/Argoverse2_filtered/{self.split}_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather'
            labels_mov = self.loader.get_labels_at_lidar_timestamp_all(
                filtered_file_path, log_id=seq, lidar_timestamp_ns=int(timestamps[0]), get_moving=True)
            
            if self.margin and 'Argo' in self.data_dir:
                for label in labels:
                    self.add_margin(label)

            # remove points of labels that are far away (>80,)
            if 'far' in self._trajectory_dir:
                all_centroids = np.asarray([label.dst_SE3_object.translation for label in labels])
                dists_to_center = np.sqrt(np.sum(all_centroids ** 2, 1))
                ind = np.where(dists_to_center <= 80)[0]
                labels = [labels[i] for i in ind]
            
            # hemove points that hare high
            if 'low' in self._trajectory_dir:
                all_centroids = np.asarray([label.dst_SE3_object.translation for label in labels])[:, -1]
                ind = np.where(all_centroids <= 4)[0]
                labels = [labels[i] for i in ind]
            
            # ALL
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

            # ONLY MOVING
            # get per point and object masks and bounding boxs and their labels 
            masks = list()
            for label in labels_mov:
                interior = point_cloud_handling.compute_interior_points_mask(
                        pc_list, label.vertices_m)
                int_label = int(label.category)
                # int_label = self.class_dict[label.category] if 'Argo' in self.data_dir else int(label.category)
                interior = interior.astype(int) * int_label
                masks.append(interior)

            if len(labels_mov) == 0:
                masks.append(np.zeros(pc_list.shape[0]))
            
            masks = np.asarray(masks).T
        
            # assign unique label and instance to each point
            # label 0 and instance 0 is background
            point_categories_mov = list()
            point_instances_mov = list()
            for j in range(masks.shape[0]):
                if np.where(masks[j]>0)[0].shape[0] != 0:
                    point_categories_mov.append(masks[j, np.where(masks[j]>0)[0][0]])
                    point_instances_mov.append(np.where(masks[j]>0)[0][0]+1)
                else:
                    point_categories_mov.append(0)
                    point_instances_mov.append(0)

            point_instances_mov = np.asarray(point_instances_mov, dtype=np.int64)
            point_categories_mov = np.asarray(point_categories_mov, dtype=np.int64)

            point_categories_mov=torch.atleast_2d(torch.from_numpy(point_categories_mov).squeeze())
            point_instances_mov=torch.atleast_2d(torch.from_numpy(point_instances_mov).squeeze())

            # putting it all together
            data = PyGData(
                pc_list=torch.from_numpy(pc_list),
                traj=torch.from_numpy(traj),
                timestamps=torch.from_numpy(timestamps),
                point_categories_mov=point_categories_mov,
                point_instances_mov=point_instances_mov,
                point_categories=point_categories,
                point_instances=point_instances,
                log_id=seq,
                # pc_normals=pc_normals
		)
        else:
            data = PyGData(
                pc_list=torch.from_numpy(pc_list),
                traj=torch.from_numpy(traj),
                timestamps=torch.from_numpy(timestamps),
                log_id=seq,
                # pc_normals=pc_normals
		)
        os.makedirs(os.path.dirname(processed_path), exist_ok=True)
        torch.save(data, osp.join(processed_path))
        
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
    
    def get_object_velocities(self, data, path):
        path = '/'.join(['/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/data'] + path.split('/')[2:])
        if os.path.isfile(path):
            return torch.load(path)
        labels = self.loader.get_labels_at_lidar_timestamp(
            log_id=data['log_id'], lidar_timestamp_ns=data['timestamps'][0].item())
        city_SE3_t1 = self.loader.get_city_SE3_ego(
            data['log_id'], data['timestamps'][0].item())
        # labels at t+1
        labels_t2 = self.loader.get_labels_at_lidar_timestamp(
            log_id=data['log_id'], lidar_timestamp_ns=data['timestamps'][1].item())
        city_SE3_t2 = self.loader.get_city_SE3_ego(
            data['log_id'], data['timestamps'][1].item())
        ids_t2 = [label.track_id for label in labels_t2]

        velocities = torch.zeros(data['pc_list'].shape[0])
        velocities_city = torch.zeros(data['pc_list'].shape[0])
        ego_traj_SE3_ego_ref = city_SE3_t2.inverse().compose(city_SE3_t1)
        for m, label in enumerate(labels):
            center_city = city_SE3_t1.transform_point_cloud(label.dst_SE3_object.translation)
            center = label.dst_SE3_object.translation
            if len(labels_t2) and label.track_id in ids_t2:
                # Pose of the object in the destination reference frame.
                # ego_SE3_object --> from object to ego   
                center_lab_city = city_SE3_t2.transform_point_cloud(labels_t2[ids_t2.index(
                    label.track_id)].dst_SE3_object.translation)
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
                translation_city = center_lab_city - center_city
                dist = np.linalg.norm(translation)
                dist_city = np.linalg.norm(translation_city)
                if 'Argo' in self.data_dir:
                    diff_time = (
                        data['timestamps'][1]-data['timestamps'][0]) / np.power(10, 9)
                else:
                    diff_time = (
                        data['timestamps'][1]-data['timestamps'][0]) / np.power(10, 6)
                vel = dist/diff_time
                vel_city = dist_city/diff_time
                interior = torch.from_numpy(point_cloud_handling.compute_interior_points_mask(
                        data['pc_list'].numpy(), label.vertices_m))
                velocities[interior] = vel
                velocities_city[interior] = vel_city
        
        data['velocities'] = velocities
        data['velocities_city'] = velocities_city
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(data, osp.join(path))
        return data

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
            if vel < self.remove_non_move_thresh:
                point_instances_mov_new[:, instance_mask] = 0
        data['point_instances_mov'] = point_instances_mov_new
        
        return data

    def filter_waymo(self, data):
        if self.waymo_style:
            mask = torch.logical_and(
                torch.logical_and(data['pc_list'][:, 0] < 50, data['pc_list'][:, 0] > -50),
                torch.logical_and(data['pc_list'][:, 1] < 20, data['pc_list'][:, 1] > -20))
            data['traj'] = data['traj'][mask]
            data['pc_list'] = data['pc_list'][mask]
            data['point_instances'] = data['point_instances'][:, mask]
            data['point_categories'] = data['point_categories'][:, mask]
            if 'point_categories_mov' in data.keys:
                data['point_instances_mov'] = data['point_instances_mov'][:, mask]
                data['point_categories_mov'] = data['point_categories_mov'][:, mask]
        return data

    def get(self, idx):
        path = self._processed_paths[idx]
        data = torch.load(path)
        if data['traj'].shape[0] == 0:
            path = self._processed_paths[0]
            data = torch.load(path)

        data['traj'] = data['traj'][:, :self.traj_channels, :self.pos_channels]
        data['pc_list'] = data['pc_list'][:, :self.pos_channels]
        data['timestamps'] = data['timestamps'][:self.traj_channels]
        
        data['point_instances'] = torch.atleast_2d(data['point_instances'])
        data['point_categories'] = torch.atleast_2d(data['point_categories'])
        
        data = self.filter_waymo(data)
        
        if data['traj'].shape[0] == 0:
            path = self._processed_paths[0]
            data = torch.load(path)
            data['point_instances'] = torch.atleast_2d(data['point_instances'])
            data['point_categories'] = torch.atleast_2d(data['point_categories'])
            data = self.filter_waymo(data)

        if 'velocities' not in data.keys and self.get_vels:
            data = self.get_object_velocities(data, path)
        if self.vel_augment:
            data = self.velocity_augment(data)
        ''' 
        name = path.split('_')[-1]
        if os.path.isfile(f'{self.edge_dir}/{name}'):
            edge_idx = torch.load(f'{self.edge_dir}/{name}', map_location='cpu')['x']
            # try:
            has_len = len(edge_idx)
            data['edge_index'] = edge_idx
            if data['edge_index'].min() != 0:
                data['edge_index'] = data['edge_index'] - data['edge_index'].min()
                # d = PyGData(data['edge_index'])
                # torch.save(d, f'{self.edge_dir}/{name}')
            # except:
            #     print('recompute... here??')
        '''
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


def get_TrajectoryDataLoader(cfg, name=None, train=True, val=True, test=False):
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
            cfg.data.trajectory_dir + '_train',
            graph_dir,
            cfg.data.use_all_points,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug,
            do_process=cfg.data.do_process,
            _processed_dir=cfg.data.processed_dir + '_train',
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
                cfg.data.trajectory_dir + f'_{split}',
                graph_dir,
                # '/workspace/result/all_egocomp_margin0.6_width25_min_mean_max_vel',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                every_x_frame=cfg.data.every_x_frame,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.processed_dir + f'_{split}', 
                percentage_data=cfg.data.percentage_data_val,
                detection_set=cfg.data.detection_set,
                filtered_file_path=cfg.data.filtered_file_path,
                detection_out_path=name,
                get_vels=cfg.data.get_vels,
                traj_channels=cfg.data.traj_channels,
                pos_channels=cfg.data.pos_channels,
                waymo_style=cfg.data.waymo_style)
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




