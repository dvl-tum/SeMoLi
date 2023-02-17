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
        self.loader = AV2SensorDataLoader(data_dir=self.split_dir, labels_dir=self.split_dir)
        self.class_dict = ARGOVERSE_CLASSES if 'argo' in self.data_dir else WAYMO_CLASSES
        super().__init__()
        self._processed_paths = self.processed_paths
    
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
        rubbish = '_rubbish' if 'rubbish' in str(self.trajectory_dir) else ''

        return Path(os.path.join('/storage/user/seidensc/datasets/trajectories' + data,\
            'processed' + rubbish, os.path.basename(os.path.dirname(self.trajectory_dir)), self.split))
        
    def add_margin(self, label):
        # Add a margin to the cuboids. Some cuboids are very tight 
        # and might lose some points.
        if self.margin:
            label.length_m += self.margin
            label.width_m += self.margin
            label.height_m += self.margin
        return label

    def process(self):
        self.loader = AV2SensorDataLoader(data_dir=self.split_dir, labels_dir=self.split_dir)
        idx = 0
        map_dict = dict()

        already_processed = glob.glob(str(self.processed_dir)+'/*/*')

        missing_paths = set(self.processed_paths).difference(already_processed)
        missing_paths = [os.path.join(
            self.trajectory_dir, os.path.basename(os.path.dirname(m)), os.path.basename(m)[:-2] + 'npz')\
                for m in missing_paths]
                
        for j, traj_file in enumerate(missing_paths): #self.raw_paths[start:]
            if j % 50 == 0:
                logger.info(f"sweep {idx}/{len(missing_paths)}, {j}-th file")
            
            processed_path = os.path.join(
                self.processed_dir,
                os.path.basename(os.path.dirname(traj_file)),
                os.path.basename(traj_file)[:-3] + 'pt')

            if os.path.isfile(processed_path):
                print(f'Exists...{processed_path}, {idx}')
                idx += 1
                continue

            seq = os.path.basename(os.path.dirname(traj_file))
            if 'non_drive' in self._trajectory_dir and not 'waymo' in self._trajectory_dir:
                if seq not in map_dict.keys():
                    log_map_dirpath = self.split_dir / seq / "map"
                    map_dict[seq] = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

            # load point clouds
            try:
                pred = np.load(traj_file, allow_pickle=True)
            except:
                print(traj_file)
                pred = np.load(traj_file, allow_pickle=True)
            traj = pred['traj']
            flow = pred['flows'] if 'flows' in [k for k in pred.keys()] else pred['flow']
            pc_list = pred['pcs'] if 'pcs' in [k for k in pred.keys()] else pred['pc_list']
            timestamps = pred['timestamps']
            if len(pred['timestamps'].shape) > 1:
                timestamps = timestamps[0]

            # When using GT flow not all pcs have the same length
            if len(np.unique([p.shape[0] for p in pc_list])) > 1:
                num_points = min([p.shape[0] for p in pc_list])
                pcs_sampled = list()
                flows_sampled = list()
                for i in range(len(pc_list)):
                    mask1_flow = np.arange(len(pc_list[i]))
                    
                    if len(pc_list[i]) >= num_points:
                        sample_idx1 = np.random.choice(mask1_flow, num_points, replace=False)
                        pcs_sampled.append(pc_list[i][sample_idx1, :].astype('float32'))
                        if i < len(flow):
                            flows_sampled.append(flow[i][sample_idx1, :].astype('float32'))
                    else:
                        pcs_sampled.append(pc_list[i])
                        flows_sampled.append(flow[i])

                    if i == 0:
                        traj = traj[sample_idx1, :, :].astype('float32')

                pc_list = np.stack(pcs_sampled)
                flow = np.stack(flows_sampled)

            # get labels
            all_instances = list()
            all_categories = list()
            if self.split != 'test':
                for i in range(len(timestamps)):
                    if i != 0:
                        continue
                    labels = self.loader.get_labels_at_lidar_timestamp(
                        log_id=seq, lidar_timestamp_ns=int(timestamps[i]))
                    
                    if self.margin:
                        for label in labels:
                            self.add_margin(label)

                    # remove labels that are non in dirvable area
                    if 'non_drive' in self._trajectory_dir and not 'waymo' in self._trajectory_dir:
                        centroids_ego = np.asarray([label.dst_SE3_object.translation for label in labels])
                        city_SE3_ego = self.loader.get_city_SE3_ego(seq, int(timestamps[i]))
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
                            pc_list[i], label.vertices_m)
                        int_label = self.class_dict[label.category] if 'argo' in self.data_dir else int(label.category)
                        interior = interior.astype(int) * int_label
                        masks.append(interior)

                    if len(labels) == 0:
                        masks.append(np.zeros(pc_list[i].shape[0]))
                    
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
                    flow=torch.from_numpy(flow),
                    pc_list=torch.from_numpy(pc_list),
                    traj=torch.from_numpy(traj),
                    timestamps=torch.from_numpy(timestamps),
                    point_categories=torch.atleast_2d(torch.from_numpy(np.asarray(all_categories)).squeeze()),
                    point_instances=torch.atleast_2d(torch.from_numpy(np.asarray(all_instances)).squeeze()),
                    log_id=seq)
            else:
                data = PyGData(
                    flow=torch.from_numpy(flow),
                    pc_list=torch.from_numpy(pc_list),
                    traj=torch.from_numpy(traj),
                    timestamps=torch.from_numpy(timestamps),
                    log_id=seq)

            os.makedirs(os.path.dirname(processed_path), exist_ok=True)
            torch.save(data, osp.join(processed_path))
            print(f"Save {osp.join(processed_path)}...")
            idx += 1

    def get(self, idx):
        path = self._processed_paths[idx]
        exists = False

        if self.remove_static:
            if 'argo' in self.data_dir:
                path2 = re.sub('argoverse2', 'argoverse2_remove_static', path)
            else:
                path2 = re.sub('waymo', 'waymo_remove_static', path)

            exists = os.path.isfile(path2)
            path = path2 if exists else path

        try:
            data = torch.load(path)
        except:
            print(f"could not load file {path} of index {idx}/{len(self.processed_paths)}")
            quit()

        if not exists:
            if len(data['pc_list'].shape) > 2:
                data['pc_list'] = data['pc_list'][0]
            if len(data['point_instances'].shape) > 1:
                data['point_instances'] = data['point_instances'][0]
            if len(data['point_categories']) > 1:
                data['point_categories'] = data['point_categories'][0]
            if len(data['flow'].shape) > 2:
                data['flow'] = data['flow'][0]

            # remove static points
            if self.remove_static:
                mean_traj = data['traj'][:, :, :-1]
                timestamps = data['timestamps']
                # get mean velocity [m/s] along trajectory and check if > thresh
                diff_cent = torch.linalg.norm(
                    mean_traj[:, :-1, :] - mean_traj[:, 1:, :] , axis=2)
                diff_time = timestamps[1:diff_cent.shape[1]+1] - timestamps[:diff_cent.shape[1]]
                if 'argo' in self.data_dir:
                    diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
                else:
                    diff_time = diff_time / torch.pow(torch.tensor(10), 6.0) 
                mean_traj = torch.mean(diff_cent/diff_time, axis=1)
                mean_traj = mean_traj > self.static_thresh

                # if no moving point and not evaluation, sample few random
                if not torch.all(~mean_traj) and not self._eval:
                    idxs = torch.randint(0, mean_traj.shape[0], size=(200, ))
                    mean_traj[idxs] = True
                
                # apply mask
                data['flow'] = data['flow'][mean_traj]
                data['pc_list'] = data['pc_list'][mean_traj, :]
                data['traj'] = data['traj'][mean_traj]

                data['point_instances'] = data['point_instances'].squeeze()[mean_traj]
                data['point_categories'] = data['point_categories'].squeeze()[mean_traj]

                os.makedirs(os.path.dirname(path2), exist_ok=True)
                torch.save(data, path2)

        # if you always want same number of points (during training), sample/re-sample
        if not self.use_all_points:
            idxs = torch.randint(0, data['flow'].shape[0], size=(self.num_points, ))
            mask = torch.arange(data['flow'].shape[0])
            mask = torch.isin(mask, idxs)

            data['flow'] = data['flow'][idxs]
            data['pc_list'] = data['pc_list'][idxs, :]
            data['traj'] = data['traj'][idxs]
            # data['point_categories'] = data['point_categories'][idxs]
            data['point_instances'] = data['point_instances'][idxs]

        if data['flow'].shape[1] == 24:
            data['flow'] = data['flow'][:, :-1, :]

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
        print('TRAIN')
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
                shuffle=True,
                num_workers=8)
        else:
            train_loader = None
    else:
        train_loader = None
    if val:
        print("VAL")
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
        print("TEST")
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
