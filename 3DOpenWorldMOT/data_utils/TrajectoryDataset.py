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


logger = logging.getLogger("Model.Dataset")


ARGOVERSE_CLASSES = {v: k for k, v in av2_classes._class_dict.items()}
WAYMO_CLASSES = {'TYPE_UNKNOWN': 0, 'TYPE_VECHICLE': 1, 'TYPE_PEDESTRIAN': 2, 'TYPE_SIGN': 3, 'TYPE_CYCLIST': 4}


class TrajectoryDataset(PyGDataset):
    def __init__(self, data_dir, split, trajectory_dir, use_all_points, num_points, remove_static, static_thresh, debug, _eval=False, every_x_frame=1, margin=0.6, split_val=False, short_train=False):
        self.split_dir = Path(os.path.join(data_dir, split))
        self.trajectory_dir = Path(os.path.join(trajectory_dir, split))
        self.data_dir = data_dir
        self.remove_static = remove_static
        self.static_thresh = static_thresh
        self._trajectory_dir = trajectory_dir
        self.split = split
        self.use_all_points = use_all_points
        self.num_points = num_points
        self.debug = debug
        self.split_val = split_val
        if self.split == 'val' or self.split == 'test':
            self.loader = AV2SensorDataLoader(data_dir=self.split_dir, labels_dir=self.split_dir)
        else:
            self.loader = None
        self._eval = _eval
        self.every_x_frame = every_x_frame
        self.margin = margin

        # use subset of seqs for training
        if short_train:
            with open('../../../debug_seqs.csv', newline='') as f:
                reader = csv.reader(f)
                train_seqs = list(reader)
            self.short_train = train_seqs[0]
        else:
            self.short_train = None
        
        if split == 'val' and 'argo' in self.data_dir:
            self.seq = '04994d08-156c-3018-9717-ba0e29be8153'
        elif split == 'train' and 'argo' in self.data_dir:
            self.seq = '00a6ffc1-6ce9-3bc3-a060-6006e9893a1a'
        elif split == 'val':
            self.seq = '16473613811052081539'
        else:
            self.seq = '2400780041057579262'

        self.class_dict = ARGOVERSE_CLASSES if 'argo' in self.data_dir else WAYMO_CLASSES
        super().__init__()
        self._processed_paths = self.processed_paths
        import glob
        a = len(self._processed_paths)
        self._processed_paths = glob.glob(self.processed_dir)
        logger.info(f'Missing files {a-len(self._processed_paths)}')
    
    @property
    def raw_file_names(self):
        if self.debug:
            raw_file_names = list()
            for seq in os.listdir(self.trajectory_dir):
                # to get results just for specific seq
                if self.seq is not None:
                    if seq != self.seq:
                        continue
                # don't take last file in directory cos some how only 24 not 25 long
                for i, flow_file in enumerate(sorted(os.listdir(os.path.join(self.trajectory_dir, seq)))):
                    if i >= len(os.listdir(os.path.join(self.processed_dir, seq)))-1 and\
                            len(os.listdir(os.path.join(self.processed_dir, seq))) != 1:
                        continue
                    if i % self.every_x_frame != 0:
                        continue
                    raw_file_names.append(flow_file)
                    
            return raw_file_names
        
        seqs = self.short_train if self.short_train is not None else os.listdir(self.trajectory_dir)

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
        if self.debug:
            raw_paths = list()
            for seq in os.listdir(self.trajectory_dir):
                # to get results just for specific seq
                if self.seq is not None:
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

        seqs = self.short_train if self.short_train is not None else os.listdir(self.trajectory_dir)

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
        if self.debug:
            processed_file_names = list()
            for seq in os.listdir(self.processed_dir):
                # ignore pre_transfor files
                if seq[-3:] == '.pt':
                    continue
                # to get results just for specific seq
                if self.seq is not None:
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

        seqs = self.short_train if self.short_train is not None else os.listdir(self.trajectory_dir)

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
        if self.debug:
            processed_paths = list()
            for seq in os.listdir(self.trajectory_dir):
                # ignore pre_transfor files
                if seq[-3:] == '.pt':
                    continue
                # to get results just for specific seq
                if self.seq is not None:
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

        seqs = self.short_train if self.short_train is not None else os.listdir(self.trajectory_dir)

        processed_paths = [os.path.join(self.processed_dir, seq, flow_file[:-3] + 'pt')\
            for seq in seqs\
                for i, flow_file in enumerate(sorted(os.listdir(osp.join(self.trajectory_dir, seq))))\
                     if i % self.every_x_frame == 0 and i < len(os.listdir(os.path.join(self.trajectory_dir, seq)))-1]
        
        if self.split_val and self.split == 'val':
            return processed_paths[:int(len(processed_paths)/2)]
        elif self.split_val and self.split == 'val':
            return processed_paths[int(len(processed_paths)/2):]
        else:
            return processed_paths

    def __len__(self):
        return len(self._processed_paths)
    
    def len(self):
        return self.__len__()

    @property
    def processed_dir(self) -> str:
        
        data = '_waymo' if 'waymo' in str(self.trajectory_dir) else ''
        if 'rubbish' in str(self.trajectory_dir):
            add_on = 'rubbish'
        elif 'local' in str(self.trajectory_dir):
            add_on = 'local'
        else:
            add_on = 'normal'

        processed_dir = os.path.join('/storage/user/seidensc/datasets/trajectories' + data,\
            'processed', add_on , os.path.basename(os.path.dirname(self.trajectory_dir)), self.split)

        if self.remove_static:
            processed_dir = re.sub('processed', 'processed_remove_static', processed_dir)

        return Path(processed_dir)
        
    def add_margin(self, label):
        # Add a margin to the cuboids. Some cuboids are very tight 
        # and might lose some points.
        if self.margin:
            label.length_m += self.margin
            label.width_m += self.margin
            label.height_m += self.margin
        return label
    
    def _process(self):
        logger.info('Processing...')
        os.makedirs(self.processed_dir, exist_ok=True)
        self.process()
        logger.info('Done!')

    def process(self, multiprocessing=True):

        already_processed = glob.glob(str(self.processed_dir)+'/*/*')
        missing_paths = set(self.processed_paths).difference(already_processed)

        missing_paths = [os.path.join(
            self.trajectory_dir, os.path.basename(os.path.dirname(m)), os.path.basename(m)[:-2] + 'npz')\
                for m in missing_paths]

        if len(missing_paths) and self.loader is None:
            self.loader = AV2SensorDataLoader(data_dir=self.split_dir, labels_dir=self.split_dir)
        import random
        random.shuffle(missing_paths)
        data_loader = enumerate(missing_paths)
            
        if multiprocessing:
            with Pool() as pool:
                _eval_sequence = partial(self.process_sweep, len(missing_paths))
                pool.map(_eval_sequence, data_loader)
        else:
            for data in data_loader:
                self.process_sweep(len(missing_paths), data)
                
    def process_sweep(self, num_data, data):
        j, traj_file = data
        if j % 10000 == 0:
            logger.info(f"sweep {j}/{num_data}, {j}-th file")
        
        processed_path = os.path.join(
            self.processed_dir,
            os.path.basename(os.path.dirname(traj_file)),
            os.path.basename(traj_file)[:-3] + 'pt')

        if self.remove_static:
            processed_path2 = processed_path
            processed_path = re.sub('processed_remove_static', 'processed', processed_path)

        # If processed file exists and not static gt removal
        if os.path.isfile(processed_path) and not self.remove_static:
            j += 1
            return
        
        # If processed file exists and static gt removal and static removed exists
        elif self.remove_static and os.path.isfile(processed_path) and os.path.isfile(processed_path2):
            j += 1
            return
        
        # If processed does not exist
        elif not os.path.isfile(processed_path):
            seq = os.path.basename(os.path.dirname(traj_file))
            if 'non_drive' in self._trajectory_dir and not 'waymo' in self._trajectory_dir:
                if seq not in map_dict.keys():
                    log_map_dirpath = self.split_dir / seq / "map"
                    map_dict[seq] = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

            # load point clouds
            try:
                pred = np.load(traj_file, allow_pickle=True)
            except:
                logger.info(f"could not load processed {traj_file}")
                return
                
            
            try:
                traj = pred['traj']
            except:
                logger.info(f"no traj111 in file {traj_file}, {[k for k in pred.keys()]}")
                return

            pc_list = pred['pcs'] if 'pcs' in [k for k in pred.keys()] else pred['pc_list']
            if len(pc_list.shape) > 2:
                pc_list = pc_list[0]

            timestamps = pred['timestamps']
            if len(pred['timestamps'].shape) > 1:
                timestamps = timestamps[0]

            # get labels
            all_instances = list()
            all_categories = list()
            if self.split != 'test':
                labels = self.loader.get_labels_at_lidar_timestamp(
                    log_id=seq, lidar_timestamp_ns=int(timestamps[0]))
                
                if self.margin:
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

                all_categories.append(point_categories)
                all_instances.append(point_instances)

                # putting it all together
                data = PyGData(
                    pc_list=torch.from_numpy(pc_list),
                    traj=torch.from_numpy(traj),
                    timestamps=torch.from_numpy(timestamps),
                    point_categories=torch.atleast_2d(torch.from_numpy(np.asarray(all_categories)).squeeze()),
                    point_instances=torch.atleast_2d(torch.from_numpy(np.asarray(all_instances)).squeeze()),
                    log_id=seq)
            else:
                data = PyGData(
                    pc_list=torch.from_numpy(pc_list),
                    traj=torch.from_numpy(traj),
                    timestamps=torch.from_numpy(timestamps),
                    log_id=seq)

            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            torch.save(data, osp.join(processed_path))
            # print(f"Save {osp.join(processed_path)}...")
        
        else:
            try:
                data = torch.load(osp.join(processed_path))
            except:
                logger.info(f'Failed to load {processed_path}...')
                return

        ### If remove static file does not exist
        if not os.path.isfile(processed_path2):
            try:
                if len(data['pc_list'].shape) > 2:
                    data['pc_list'] = data['pc_list'][0]
            except:
                logger.info(f"len oc list thing {data['pc_list'].shape}")
                return
            if len(data['point_instances'].shape) > 1:
                data['point_instances'] = data['point_instances'][0]
            if len(data['point_categories']) > 1:
                data['point_categories'] = data['point_categories'][0]

            # remove static points
            if self.remove_static:
                mean_traj = data['traj'][:, :, :-1]
                timestamps = data['timestamps']
                # get mean velocity [m/s] along trajectory and check if > thresh
                diff_dist = torch.linalg.norm(
                    mean_traj[:, :-1, :] - mean_traj[:, 1:, :] , axis=2)

                diff_time = timestamps[1:diff_dist.shape[1]+1] - timestamps[:diff_dist.shape[1]]
                # bring from nano / mili seconds to seconds
                if 'argo' in self.data_dir:
                    diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
                else:
                    diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)

                mean_traj = torch.mean(diff_dist/diff_time, axis=1)
                mean_traj = mean_traj > self.static_thresh
                
                # if no moving point and not evaluation, sample few random
                if torch.all(~mean_traj) and not self._eval:
                    idxs = torch.randint(0, mean_traj.shape[0], size=(200, ))
                    mean_traj[idxs] = True
                
                # apply mask
                data['pc_list'] = data['pc_list'][mean_traj, :]
                data['traj'] = data['traj'][mean_traj]

                data['point_instances'] = data['point_instances'].squeeze()[mean_traj]
                data['point_categories'] = data['point_categories'].squeeze()[mean_traj]

                os.makedirs(os.path.dirname(processed_path2), exist_ok=True)
                torch.save(data, processed_path2)

    def get(self, idx):
        path = self._processed_paths[idx]

        '''if self.remove_static:
            if 'argo' in self.data_dir:
                path = re.sub('argoverse2', 'argoverse2_remove_static', path)
            else:
                path = re.sub('waymo', 'waymo_remove_static', path)'''
        try:
            data = torch.load(path)
        except:
            logger.info(f"could not load file {path} of index {idx}/{len(self.processed_paths)}")
            return

        # if you always want same number of points (during training), sample/re-sample
        if not self.use_all_points:
            idxs = torch.randint(0, data['traj'].shape[0], size=(self.num_points, ))
            mask = torch.arange(data['traj'].shape[0])
            mask = torch.isin(mask, idxs)

            data['pc_list'] = data['pc_list'][idxs, :]
            data['traj'] = data['traj'][idxs]
            data['point_categories'] = data['point_categories'][idxs]
            data['point_instances'] = data['point_instances'][idxs]

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


def get_TrajectoryDataLoader(cfg, train=True, val=True, test=False):
    # get datasets
    if train and not cfg.just_eval:
        logger.info('TRAIN')
        train_data = TrajectoryDataset(
            cfg.data.data_dir,
            'train',
            cfg.data.trajectory_dir,
            cfg.data.use_all_points,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug,
            short_train=cfg.data.short_train)
        if len(train_data):
            train_loader = PyGDataLoader(
                train_data,
                batch_size=cfg.training.batch_size,
                drop_last=True,
                shuffle=True)
        else:
            train_loader = None
    else:
        train_loader = None
    if val:
        logger.info("VAL")
        val_data = TrajectoryDataset(cfg.data.data_dir,
            'val',
            cfg.data.trajectory_dir,
            cfg.data.use_all_points_eval,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug,
            _eval=True, 
            every_x_frame=cfg.data.every_x_frame,
            split_val=cfg.data.split_val)
        if len(val_data):
            val_loader = PyGDataLoader(val_data, batch_size=cfg.training.batch_size_val)
        else:
            val_loader = None
    else:
        val_loader = None
    if test:
        logger.info("TEST")
        test_data = TrajectoryDataset(cfg.data.data_dir,
            'test',
            cfg.data.trajectory_dir,
            cfg.data.use_all_points_eval,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug,
            _eval=True, 
            every_x_frame=cfg.data.every_x_frame,
            split_val=cfg.data.split_val)
        if len(test_data):
            test_loader = PyGDataLoader(test_data, batch_size=cfg.training.batch_size_val)
        else:
            test_loader = None
    else:
        test_loader = None

    return train_loader, val_loader, test_loader
