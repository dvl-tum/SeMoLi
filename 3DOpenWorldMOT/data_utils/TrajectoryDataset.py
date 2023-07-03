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


logger = logging.getLogger("Model.Dataset")


ARGOVERSE_CLASSES = {v: k for k, v in av2_classes._class_dict.items()}
WAYMO_CLASSES = {'TYPE_UNKNOWN': 0, 'TYPE_VECHICLE': 1, 'TYPE_PEDESTRIAN': 2, 'TYPE_SIGN': 3, 'TYPE_CYCLIST': 4}


class TrajectoryDataset(PyGDataset):
    def __init__(self, data_dir, split, trajectory_dir, use_all_points, num_points, remove_static, static_thresh, debug, _eval=False, every_x_frame=1, margin=0.6, split_val=False, _processed_dir=False, do_process=True, seq=None, name=None, percentage_data=1):
        self.split_dir = Path(os.path.join(data_dir, split))
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
        self.split_val = split_val
        if self.split == 'train' or self.split == 'val' or self.split == 'test':
            self.loader = AV2SensorDataLoader(data_dir=self.split_dir, labels_dir=self.split_dir)
        else:
            self.loader = None
        self._eval = _eval
        self.every_x_frame = every_x_frame
        self.margin = margin
        self._processed_dir = _processed_dir
        self.do_process =  do_process
        self.percentage_data = percentage_data
        
        self.seq = None
        # for validation multi-gpu processing
        if seq is not None:
            self.seq = seq
        # for debugging
        elif debug:
            if split == 'val' and 'argo' in self.data_dir:
                self.seq = '04994d08-156c-3018-9717-ba0e29be8153'
            elif split == 'train' and 'argo' in self.data_dir:
                self.seq = '00a6ffc1-6ce9-3bc3-a060-6006e9893a1a'
            elif split == 'val':
                self.seq = '16473613811052081539'
            else:
                self.seq = '2400780041057579262'

        self.class_dict = ARGOVERSE_CLASSES if 'argo' in self.data_dir else WAYMO_CLASSES
        if name is not None:
            self.already_evaluated = [f'{os.sep}'.join(f'{os.sep}'.split(f)[-2:]) for f in glob.glob('/workspace/result/out/' + name + '/val/*/*')]
        else:
            self.already_evaluated = list()
        print(self.already_evaluated)
        super().__init__()
        
        # import glob
        # a = len(self._processed_paths)
        # self._processed_paths = glob.glob(str(self.processed_dir)+'/*/*')
        # logger.info(f'Missing files {a-len(self._processed_paths)}')
        self._processed_paths_0 = self._processed_paths[0]
        # self._processed_paths = ['/storage/user/seidensc/datasets/trajectories_waymo/processed_remove_static/normal/gt_all_egocomp_margin0.6_width25/val/16473613811052081539/1543280278723549.pt']
    
    @property
    def raw_file_names(self):
        if self.seq is not None:
            raw_file_names = list()
            for seq in os.listdir(self.trajectory_dir):
                # to get results just for specific seq
                if seq != self.seq:
                    continue
                # don't take last file in directory cos some how only 24 not 25 long
                for i, flow_file in enumerate(sorted(os.listdir(os.path.join(self.trajectory_dir, seq)))):
                    if i >= len(os.listdir(os.path.join(self.processed_dir, seq)))-1:
                        continue
                    if i % self.every_x_frame != 0:
                        continue
                    raw_file_names.append(flow_file)
                    
            return raw_file_names
        
        seqs = os.listdir(self.trajectory_dir)
        seqs = seqs[:int(self.percentage_data*len(seqs))]
        raw_file_names = [flow_file for seq in seqs\
            for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
        
        if self.split_val and self.split == 'val':
            return raw_file_names[:int(len(raw_file_names)/2)]
        elif self.split_val and self.split == 'val':
            return raw_file_names[int(len(raw_file_names)/2):]
        else:
            return raw_file_names
    
    @property
    def raw_paths(self):
        if self.seq is not None:
            raw_paths = list()
            for seq in os.listdir(self.trajectory_dir):
                # to get results just for specific seq
                if seq != self.seq:
                    continue
                # don't take last file in directory cos some how only 24 not 25 long
                for i, flow_file in enumerate(sorted(os.listdir(os.path.join(self.trajectory_dir, seq)))):
                    if i >= len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1 and\
                            len(os.listdir(os.path.join(self.trajectory_dir, seq))) != 1:
                        continue
                    if i % self.every_x_frame != 0:
                        continue
                    raw_paths.append(os.path.join(self.trajectory_dir, seq, flow_file))

            return raw_paths

        seqs = os.listdir(self.trajectory_dir)
        seqs = seqs[:int(self.percentage_data*len(seqs))]
        raw_paths = [os.path.join(self.trajectory_dir, seq, flow_file)\
            for seq in seqs\
                for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                    if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
    
        if self.split_val and self.split == 'val':
            return raw_paths[:int(len(raw_paths)/2)]
        elif self.split_val and self.split == 'val':
            return raw_paths[int(len(raw_paths)/2):]
        else:
            return raw_paths

    @property
    def processed_file_names(self):
        if self.seq is not None:
            processed_file_names = list()
            for seq in os.listdir(self.processed_dir):
                # ignore pre_transfor files
                if seq[-3:] == '.pt':
                    continue
                # to get results just for specific seq
                if seq != self.seq:
                    continue
                # don't take last file in directory cos some how only 24 not 25 long
                for i, flow_file in enumerate(sorted(os.listdir(os.path.join(self.processed_dir, seq)))):
                    if i >= len(os.listdir(os.path.join(self.processed_dir, seq)))-1 and\
                            len(os.listdir(os.path.join(self.processed_dir, seq))) != 1:
                        continue
                    if i % self.every_x_frame != 0:
                        continue
                    processed_file_names.append(flow_file)

            return processed_file_names

        seqs = os.listdir(self.trajectory_dir)
        seqs = seqs[:int(self.percentage_data*len(seqs))]
        processed_file_names =  [flow_file[:-3] + 'pt' for seq in seqs\
            for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq)))) \
                if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
        
        if self.split_val and self.split == 'val':
            return processed_file_names[:int(len(processed_file_names)/2)]
        elif self.split_val and self.split == 'val':
            return processed_file_names[int(len(processed_file_names)/2):]
        else:
            return processed_file_names
    
    @property
    def processed_paths(self):
        if not self.do_process:
            self.processed_once = True
            if self.seq is not None:
                processed_paths = list()
                for i, flow_file in enumerate(sorted(os.listdir(os.path.join(self.processed_dir, self.seq)))):
                    if i % self.every_x_frame != 0:
                        continue
                    '''if '1543280278723549' not in flow_file:
                        continue'''
                    '''if 'gt' in str(self.processed_dir) and i > 20:
                        break'''
                    processed_paths.append(os.path.join(self.processed_dir, self.seq, flow_file))
            else:
                seqs = os.listdir(self.processed_dir)
                seqs = seqs[:int(self.percentage_data*len(seqs))]
                processed_paths = [os.path.join(self.processed_dir, seq, flow_file)\
                    for seq in seqs\
                        for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.processed_dir, seq))))\
                            if i % self.every_x_frame == 0]
        else:
            self.processed_once = False
            if self.seq is not None:
                processed_paths = list()
                for seq in os.listdir(self.trajectory_dir):
                    # ignore pre_transfor files
                    if seq[-3:] == '.pt':
                        continue
                    # to get results just for specific seq
                    if seq != self.seq:
                        continue
                    # don't take last file in directory cos some how only 24 not 25 long
                    for i, flow_file in enumerate(sorted(os.listdir(os.path.join(self.trajectory_dir, seq)))):
                        if i % self.every_x_frame != 0:
                            continue
                        if i >= len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1 \
                                and len(os.listdir(os.path.join(self.trajectory_dir, seq))) != 1:
                            continue
                        processed_paths.append(os.path.join(self.processed_dir, seq, flow_file[:-3] + 'pt'))

                return processed_paths
            else:
                seqs = os.listdir(self.trajectory_dir)
                seqs = seqs[:int(self.percentage_data*len(seqs))]
                print(len(seqs))
                processed_paths = [os.path.join(self.processed_dir, seq, flow_file[:-3] + 'pt')\
                    for seq in seqs\
                        for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq))))\
                             if i % self.every_x_frame == 0] # and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]

        if self.split_val and self.split == 'val':
            return processed_paths[:int(len(processed_paths)/2)]
        elif self.split_val and self.split == 'train':
            return processed_paths[int(len(processed_paths)/2):]
        else:
            return processed_paths

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
        self._processed_paths = [f for f in self._processed_paths if f'{os.sep}'.join(f'{os.sep}'.split(f)[-2:]) not in self.already_evaluated]
        if self.do_process:
            logger.info('Processing...')
            os.makedirs(self.processed_dir, exist_ok=True)
            self.process()
            logger.info('Done!')
        else:
            logger.info('Not Processing this time :) ')
    
    def process(self):
        self.process_once()
        return

    def process_once(self, multiprocessing=False):
        already_processed = glob.glob(str(self.processed_dir)+'/*/*')
        # already_processed = list()
        missing_paths = set(self._processed_paths).difference(already_processed)
        missing_paths = [os.path.join(
            self.trajectory_dir, os.path.basename(os.path.dirname(m)), os.path.basename(m)[:-2] + 'npz')\
                for m in missing_paths]
        print(len(already_processed), len(missing_paths), len(self._processed_paths))
        if len(missing_paths) and self.loader is None:
            self.loader = AV2SensorDataLoader(data_dir=self.split_dir, labels_dir=self.split_dir)

        data_loader = enumerate(missing_paths)            
        from torch import multiprocessing as mp
        self.len_missing = len(missing_paths)
        if multiprocessing:
            mp.set_start_method('forkserver')
            with mp.Pool() as pool:
                pool.map(self.process_sweep, data_loader, chunksize=None)
        else:
            for data in data_loader:
                self.process_sweep(data)
        
        self.processed_once = True

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

        if 'argo' not in self._processed_dir:            
            mask = np.logical_and(mask, sweep_df['laser_id'].isin(laser_id))
            mask = np.logical_and(mask, sweep_df['index'].isin(index))

        if remove_ground_pts_rc:
            mask = np.logical_and(mask, sweep_df['non_ground_pts_rc'])

        # remove non drivable area points (non RoI points)
        if remove_non_dirvable and 'argo' in self._processed_dir:
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
        print(j)
        if j % 1 == 0:
            logger.info(f"sweep {j}/{self.len_missing}, {j}-th file")
        
        processed_path = os.path.join(
            self.processed_dir,
            os.path.basename(os.path.dirname(traj_file)),
            os.path.basename(traj_file)[:-3] + 'pt')

        orig_path = os.path.join(
            self.data_dir,
            self.split,
            os.path.basename(os.path.dirname(traj_file)),
            'sensors',
            'lidar',
            os.path.basename(traj_file)[:-3] + 'feather')
        
        # If processed does not exist
        seq = os.path.basename(os.path.dirname(traj_file))
        if 'non_drive' in self._trajectory_dir and not 'waymo' in self._trajectory_dir:
            if seq not in map_dict.keys():
                log_map_dirpath = self.split_dir / seq / "map"
                map_dict[seq] = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)
            
        # load original pc
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
        all_instances = list()
        all_categories = list()
        if self.split != 'test':
            labels = self.loader.get_labels_at_lidar_timestamp(
                log_id=seq, lidar_timestamp_ns=int(timestamps[0]))
            filtered_file_path = f'/dvlresearch/jenny/Waymo_Converted_filtered_{self.split}/{self.split}_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather'
            filtered_file_path = f'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Waymo_Converted_filtered_{self.split}/{self.split}_1.0_per_frame_remove_non_move_remove_far_filtered_version.feather'
            labels_mov = self.loader.get_labels_at_lidar_timestamp_all(
                filtered_file_path, log_id=seq, lidar_timestamp_ns=int(timestamps[0]), get_moving=True)
            
            if self.margin and 'argo' in self.data_dir:
                for label in labels:
                    self.add_margin(label)

            # remove labels that are non in dirvable area
            if 'non_drive' in self._trajectory_dir and not 'waymo' in self._trajectory_dir:
                centroids_ego = np.asarray([label.dst_SE3_object.translation for label in labels])
                city_SE3_ego = self.loader.get_city_SE3_ego(seq, int(timestamps[0]))
                centroids_city = city_SE3_ego.transform_point_cloud(
                        centroids_ego)
                bool_labels = map_dict[seq].get_raster_layer_points_boolean(
                    centroids_city, layer_name=RasterLayerType.DRIVABLE_AREA)
                labels = [l for i, l in enumerate(labels) if bool_labels[i]]

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
                int_label = self.class_dict[label.category] if 'argo' in self.data_dir else int(label.category)
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
                int_label = self.class_dict[label.category] if 'argo' in self.data_dir else int(label.category)
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
        if 'argo' in self.data_dir:
            diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
        else:
            diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)

        mean_traj = diff_dist/diff_time
        mean_traj = mean_traj > self.static_thresh

        empty = False
        # if no moving point and not evaluation, sample few random
        if torch.all(~mean_traj):
            idxs = torch.randint(0, mean_traj.shape[0], size=(200, ))
            mean_traj[idxs] = True
            empty = True                    
        
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
        data['empty'] = empty
        
        return data

    def get(self, idx): 
        path = self._processed_paths[idx]
        try:
            data = torch.load(path)
        except:
            print(f"Not able to load {self._processed_paths_0}")
            data = torch.load(self._processed_paths_0)

        if self.remove_static and self.static_thresh > 0:
            data = self._remove_static(data)
        else:
            # print("HOOOHO", data, data['point_categories'], type(data['point_categories']), data['point_categories'].shape)
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

        if 'empty' not in data.keys:
            if data['pc_list'].shape[0] == 0:
                data['empty'] = True
            else:
                data['empty'] = False

        if data['empty']:
            data = torch.load(self._processed_paths_0)
        data['batch'] = torch.ones(data['pc_list'].shape[0])*idx
        data['timestamps'] = data['timestamps'].unsqueeze(0)
        
        return data

    def visualize(self, data, idx):
        import matplotlib.pyplot as plt
        idxs = torch.randint(0, data['flow'].shape[0], size=(200, ))
        idxs = torch.arange(0, data['flow'].shape[0])
        pc_list = data['pc_list'][idxs, :]
        traj = data['traj'][idxs]
        flow = data['flow'][idxs]

        here = list(range(pc_list.shape[0]))[-3]
        for i, (p, t, f) in enumerate(zip(pc_list, traj, flow)):
            future = p.repeat((t.shape[0], 1)) + t
            if idx == 0:
                t = t[10:15]
                future = future[10:15]
            else:
                t = t[:5]
                future = future[:5]
            plt.scatter(future[:, 0], future[:, 1])
            # future2 = future[:-1, :] + flow
            # plt.scatter(future2[:, 0], future2[:, 1])
        plt.savefig(f'../../../vis_{idx}.jpg')
        plt.close()


def get_TrajectoryDataLoader(cfg, name=None, train=True, val=True, test=False):
    # get datasets
    if train and not cfg.just_eval:
        logger.info('TRAIN')
        train_data = TrajectoryDataset(
            cfg.data.data_dir + '_train',
            'train',
            cfg.data.trajectory_dir + '_train',
            cfg.data.use_all_points,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug,
            do_process=cfg.data.do_process,
            _processed_dir=cfg.data.processed_dir + '_train',
            percentage_data=cfg.data.percentage_data)
    else:
        train_data = None
    if val:
        logger.info("VAL")
        val_data = TrajectoryDataset(cfg.data.data_dir + '_val',
                'val',
                cfg.data.trajectory_dir + '_val',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                _eval=True, 
                every_x_frame=cfg.data.every_x_frame,
                split_val=cfg.data.split_val,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.processed_dir + '_val', 
                name=name,
                percentage_data=cfg.data.percentage_data)

        '''val_data = list()
        seq_list = os.listdir(f'{cfg.data.trajectory_dir}_val/val')[:4]
        for i, seq in enumerate(seq_list):
            logger.info(f"Seq {i}/{len(seq_list)}")
            val_data.append(TrajectoryDataset(cfg.data.data_dir,
                'val',
                cfg.data.trajectory_dir + '_val',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                _eval=True, 
                every_x_frame=cfg.data.every_x_frame,
                split_val=cfg.data.split_val,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.processed_dir + '_val',
                seq=seq))'''
    else:
        val_data = None
    if test:
        logger.info("TEST")
        test_data = list()
        seq_list = os.listdir(f'{cfg.data.trajectory_dir}_test/test')
        for i, seq in enumerate(seq_list):
            logger.info(f"Seq {i}/{len(seq_list)}")
            test_data.append(TrajectoryDataset(cfg.data.data_dir,
                'test',
                cfg.data.trajectory_dir + '_test',
                cfg.data.use_all_points_eval,
                cfg.data.num_points_eval,
                cfg.data.remove_static,
                cfg.data.static_thresh,
                cfg.data.debug,
                _eval=True, 
                every_x_frame=cfg.data.every_x_frame,
                split_val=cfg.data.split_val,
                do_process=cfg.data.do_process,
                _processed_dir=cfg.data.processed_dir + '_test',
                seq=seq))
    else:
        test_data = None
    
    return train_data, val_data, test_data




