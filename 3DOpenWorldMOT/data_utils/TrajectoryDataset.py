from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Dataset as PyGDataset
from torch_geometric.data import Data as PyGData
from torch_geometric.data import DataLoader as PyGDataLoader
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

logger = logging.getLogger("Model.Dataset")


class_dict = {v: k for k, v in av2_classes._class_dict.items()}


class TrajectoryDataset(PyGDataset):
    def __init__(self, data_dir, split, trajectory_dir, use_all_points, num_points, remove_static, static_thresh, debug, _eval=False, every_x_frame=1):
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
        self._eval = _eval
        self.every_x_frame = every_x_frame
        self.seq = None # '04994d08-156c-3018-9717-ba0e29be8153'
        self.loader = AV2SensorDataLoader(data_dir=self.split_dir, labels_dir=self.split_dir)
        super().__init__()
    
    @property
    def raw_file_names(self):
        return [flow_file for seq in os.listdir(self.trajectory_dir)\
            for flow_file in sorted(os.listdir(osp.join(self.trajectory_dir, seq)))] #if len(os.listdir(osp.join(self.trajectory_dir, seq))) > 130]
    
    @property
    def raw_paths(self):
        return [os.path.join(self.trajectory_dir, seq, flow_file)\
            for seq in os.listdir(self.trajectory_dir)\
                for flow_file in sorted(os.listdir(osp.join(self.trajectory_dir, seq)))] #if len(os.listdir(osp.join(self.trajectory_dir, seq))) > 130]
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
                    if i >= len(os.listdir(os.path.join(self.processed_dir, seq)))-1:
                        continue
                    processed_file_names.append(flow_file)
            return processed_file_names

        return [flow_file[:-3] + 'pt' for seq in os.listdir(self.trajectory_dir)\
            for flow_file in sorted(os.listdir(osp.join(self.trajectory_dir, seq)))] #if len(os.listdir(osp.join(self.trajectory_dir, seq))) > 130]
    
    @property
    def processed_paths(self):
        if self.debug:
            processed_paths = list()
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
                    if i % self.every_x_frame != 0:
                        continue
                    if i >= len(os.listdir(os.path.join(self.processed_dir, seq)))-1:
                        continue
                    processed_paths.append(os.path.join(self.processed_dir, seq, flow_file))
            return processed_paths

        return [os.path.join(self.processed_dir, seq, flow_file[:-3] + 'pt')\
            for seq in os.listdir(self.trajectory_dir)\
                for flow_file in sorted(os.listdir(osp.join(self.trajectory_dir, seq)))] #if len(os.listdir(osp.join(self.trajectory_dir, seq))) > 130] #  

    def __len__(self):
        return len(self.processed_paths)
    
    def len(self):
        return self.__len__()

    @property
    def processed_dir(self) -> str:
        return Path(os.path.join('/storage/user/seidensc/datasets/trajectories',\
            'processed', f"{os.sep}".join(self.data_dir.split(os.sep)[-1:] + [self.split])))

    def process(self):
        self.loader = AV2SensorDataLoader(data_dir=self.split_dir, labels_dir=self.split_dir)
        idx = 0
        map_dict = dict()
        for j, traj_file in enumerate(self.raw_paths):
            if j % 50 == 0:
                print(f"sweep {j}/{len(self.raw_paths)}")
            
            if os.path.isfile(self.processed_paths[idx]):
                continue

            seq = os.path.basename(os.path.dirname(traj_file))
            if 'non_drive' in self._trajectory_dir:
                if seq not in map_dict.keys():
                    log_map_dirpath = self.split_dir / seq / "map"
                    map_dict[seq] = ArgoverseStaticMap.from_map_dir(log_map_dirpath, build_raster=True)

            # load point clouds
            pred = np.load(traj_file, allow_pickle=True)
            traj = pred['traj']
            flow = pred['flow']
            pc_list = pred['pc_list']
            timestamps = pred['timestamps'][0]

            all_instances = list()
            # get labels
            indicator = list()
            categories = list()
            all_categories = list()
            rots, trans, lwh = list(), list(), list()
            if self.split != 'test':
                for i in range(len(timestamps)):
                    labels = self.loader.get_labels_at_lidar_timestamp(
                        log_id=seq, lidar_timestamp_ns=int(timestamps[i]))            
                    # remove labels that are non in dirvable area
                    if 'non_drive' in self._trajectory_dir:
                        centroids_ego = np.asarray([label.dst_SE3_object.translation for label in labels])
                        city_SE3_ego = self.loader.get_city_SE3_ego(seq, int(timestamps[i]))
                        centroids_city = city_SE3_ego.transform_point_cloud(
                                centroids_ego)
                        bool_labels = map_dict[seq].get_raster_layer_points_boolean(
                            centroids_city, layer_name=RasterLayerType.DRIVABLE_AREA)
                        labels = [l for i, l in enumerate(labels) if bool_labels[i]]

                    # remove points that are far away (>80,)
                    all_centroids = np.asarray([label.dst_SE3_object.translation for label in labels])
                    dists_to_center = np.sqrt(np.sum(all_centroids ** 2, 1))
                    ind = np.where(dists_to_center <= 80)[0]
                    labels = [labels[i] for i in ind]

                    # get per point and object masks and bounding boxs and their labels 
                    masks = list()
                    for label in labels:
                        indicator.append(i)
                        interior = point_cloud_handling.compute_interior_points_mask(
                            pc_list[i], label.vertices_m)
                        interior = interior.astype(int) * class_dict[label.category]
                        masks.append(interior)
                        categories.append(class_dict[label.category])
                        rots.append(label.dst_SE3_object.rotation)
                        trans.append(label.dst_SE3_object.translation)
                        lwh.append(np.asarray([label.length_m, label.width_m, label.height_m]))
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

                    point_instances = np.asarray(point_instances)
                    point_categories = np.asarray(point_categories)

                    all_categories.append(point_categories)
                    all_instances.append(point_instances)

                # putting it all together
                data = PyGData(
                    flow=torch.from_numpy(flow),
                    pc_list=torch.from_numpy(pc_list),
                    traj=torch.from_numpy(traj),
                    timestamps=torch.from_numpy(timestamps),
                    rots=torch.from_numpy(np.asarray(rots)),
                    trans=torch.from_numpy(np.asarray(trans)),
                    lwh=torch.from_numpy(np.asarray(lwh)),
                    categories=torch.from_numpy(np.asarray(categories)),
                    point_categories=torch.from_numpy(np.asarray(all_categories)),
                    point_instances=torch.from_numpy(np.asarray(all_instances)),
                    indicator=torch.tensor(indicator),
                    log_id=seq)
            else:
                data = PyGData(
                    flow=torch.from_numpy(flow),
                    pc_list=torch.from_numpy(pc_list),
                    traj=torch.from_numpy(traj),
                    timestamps=torch.from_numpy(timestamps),
                    log_id=seq)
            os.makedirs(os.path.dirname(self.processed_paths[idx]), exist_ok=True)
            torch.save(data, osp.join(self.processed_paths[idx]))
            idx += 1

    def get(self, idx):
        data = torch.load(self.processed_paths[idx])

        '''city_SE3_ego = self.loader.get_city_SE3_ego(
                        data.log_id, data.timestamps[0].item())'''

        data['pc_list'] = data['pc_list'][0]
        '''data['pc_list'] = city_SE3_ego.transform_point_cloud(
                                data['pc_list'])

        data['traj'] = city_SE3_ego.transform_point_cloud(
                                data['traj'].view(-1, 3))
        data['traj'] = data['traj'].view(data['pc_list'].shape[0], -1, 3)'''

        data['point_categories'] = data['point_categories'][0]
        data['point_instances'] = data['point_instances'][0]
        if self.remove_static:
            mean_traj = data['traj'][:, 1, :-1]
            # mean_traj = mean_traj.reshape(mean_traj.shape[0], -1)
            # mean_traj = torch.abs(torch.mean(mean_traj, dim=1))
            mean_traj = torch.linalg.norm(mean_traj, axis=1)
            mean_traj = mean_traj > self.static_thresh

            if not torch.all(~mean_traj) and not self._eval:
                idxs = torch.randint(0, mean_traj.shape[0], size=(200, ))
                mean_traj[idxs] = True

            data['flow'] = data['flow'][mean_traj]
            data['pc_list'] = data['pc_list'][mean_traj, :]
            data['traj'] = data['traj'][mean_traj]
            data['point_categories'] = data['point_categories'][mean_traj]
            data['point_instances'] = data['point_instances'][mean_traj]

        if not self.use_all_points:
            idxs = torch.randint(0, data['flow'].shape[0], size=(self.num_points, ))
            mask = torch.arange(data['flow'].shape[0])
            mask = torch.isin(mask, idxs)

            data['flow'] = data['flow'][idxs]
            data['pc_list'] = data['pc_list'][idxs, :]
            data['traj'] = data['traj'][idxs]
            data['point_categories'] = data['point_categories'][idxs]
            data['point_instances'] = data['point_instances'][idxs]

        self.visualize(data, idx)

        return data

    def visualize(self, data, idx):
        import matplotlib.pyplot as plt
        idxs = torch.randint(0, data['flow'].shape[0], size=(200, ))
        pc_list = data['pc_list'][idxs, :]
        traj = data['traj'][idxs]
        flow = data['flow'][idxs]

        print(pc_list.shape)
        print(traj.shape)
        print(flow.shape)

        for p, t, f in zip(pc_list, traj, flow):
            future = p.repeat((t.shape[0], 1)) + t
            if idx == 0:
                future = future[10:15]
            else:
                future = future[:5]

            plt.scatter(future[:, 0], future[:, 1])
            # future2 = future[:-1, :] + flow
            # plt.scatter(future2[:, 0], future2[:, 1])
        plt.savefig(f'../../../vis_{idx}.jpg')
        plt.close()


def get_TrajectoryDataLoader(cfg, train=False, val=True, test=False):
    # get datasets
    if train:
        print('TRAIN')
        train_data = TrajectoryDataset(
            cfg.data.data_dir,
            'train',
            cfg.data.trajectory_dir,
            cfg.data.use_all_points,
            cfg.data.num_points,
            cfg.data.remove_static,
            cfg.data.static_thresh,
            cfg.data.debug)
        if len(train_data):
            train_loader = PyGDataLoader(train_data, batch_size=cfg.training.batch_size, drop_last=True, shuffle=True)
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
            every_x_frame=cfg.data.every_x_frame)
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
            every_x_frame=cfg.data.every_x_frame)
        if len(test_data):
            test_loader = PyGDataLoader(test_data, batch_size=cfg.training.batch_size_val)
        else:
            test_loader = None
    else:
        test_loader = None

    return train_loader, val_loader, test_loader