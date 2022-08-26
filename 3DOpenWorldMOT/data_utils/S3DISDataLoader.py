import os
import os.path as osp
import numpy as np

from .base_dataset import BaseDataset
from torch_geometric.data.dataset import Dataset as PyGDataset
from torch_geometric.data import Data
import torch


class S3DIS(PyGDataset):
    def __init__(self,
      root,
      split="trainval",
      transform=None,
      process_workers=1,
      pre_transform=None,
      test_area=5):
        self.test_area = test_area
        self.root = root
        self.pre_transform = pre_transform
        self._processed_paths = {split: list() for split in self.AVAILABLE_SPLITS}
        self.split = split
        super().__init__(self)
        
    @property
    def SPLIT(self) -> dict:
        splits = dict()
        rooms = sorted(os.listdir(self.dataroot))
        rooms = [room for room in rooms if 'Area_' in room]
        splits['train'] = [room for room in rooms if not 'Area_{}'.format(self.test_area) in room]
        splits['val'] = [room for room in rooms if 'Area_{}'.format(self.test_area) in room]
        splits['test'] = [room for room in rooms if 'Area_{}'.format(self.test_area) in room]
        return splits
    
    @property
    def AVAILABLE_SPLITS(self) -> list:
        return ["train", "val", "test"]
    
    def load_paths(self, rooms):
        room_paths = list()
        for room in rooms:
            room_paths.extend(osp.join(self.root, room))
        return room_paths

    def process(self):
        room_points, room_labels = [], []
        room_coord_min, room_coord_max = [], []
        num_point_all = []
        labelweights = np.zeros(13)

        for i, split in enumerate(self.AVAILABLE_SPLITS):
            if osp.exists(self.processed_paths[i]):
                continue
            os.makedirs(self.processed_paths[i])

            rooms = self.SPLIT[split]
            room_paths = self._load_paths(rooms)
            for room_path in room_paths:
                room_data = np.load(room_path)
                points, labels = room_data[:, 0:6], room_data[:, 6]
                tmp, _ = np.histogram(labels, range(14))
                labelweights += tmp
                coord_min, coord_max = np.amin(points, axis=0)[:3], np.amax(points, axis=0)[:3]
                room_points.append(points)
                room_labels.append(labels)
                room_coord_min.append(coord_min)
                room_coord_max.append(coord_max)
                num_point_all.append(labels.size)

            room_idxs, labelweights = self.get_final_rooms(
                room_points, room_labels, labelweights, num_point_all)
            for idx in range(len(room_idxs)):
                self.sample_and_save(
                    idx, room_idxs, room_points, room_labels, room_coord_min, room_coord_max, split)

        return 
    
    def get_final_rooms(self, room_points, labelweights, num_point_all):

        # WEIGHT LABELS BY PERCENTAGE
        labelweights = labelweights.astype(np.float32)
        labelweights = labelweights / np.sum(labelweights)
        labelweights = np.power(np.amax(labelweights) / labelweights, 1 / 3.0)

        # COMPUTE ROOM SAMPLE PROBABILITY AND NUMB
        sample_prob = num_point_all / np.sum(num_point_all)
        num_iter = int(np.sum(num_point_all) *\
            self.pre_transform.sample_rate /\
                self.pre_transform.npoints)
        # how often do we need to sample room to get (sample_rate * # all points) 
        # if we take npoints per batch
        room_idxs = []
        for index in range(len(room_points)):
            room_idxs.extend([index] * int(round(sample_prob[index] * num_iter)))
        room_idxs = np.array(room_idxs)

        return room_idxs, labelweights
    
    def sample_and_save(
      self, idx, room_idxs, room_points, room_labels, room_coord_min, room_coord_max, split):
        room_idx = room_idxs[idx]
        points = room_points[room_idx]   # N * 6
        labels = room_labels[room_idx]   # N
        N_points = points.shape[0]

        point_idxs, center = self.get_points(points, N_points)
        current_points, current_labels = self.select_and_center(point_idxs, points, center, room_idx, labels, room_coord_min, room_coord_max)

        data = Data(pos=torch.from_numpy(current_points[:, :3]), x=torch.from_numpy(current_points[:, 3:]))
        data.y = torch.from_numpy(current_labels)
        out_file = osp.join(self.processed_room_dirs[room_idx], "{}.pt".format(idx)) 
        torch.save(data, out_file)
        split_list = self.processed_paths[split]
        self.processed_paths[split] = split_list.append(out_file)

    def get_points(self, points, N_points):
        while (True):
            center = points[np.random.choice(N_points)][:3]
            block_min = center - [
                self.pre_transform.block_size / 2.0, self.pre_transform.block_size / 2.0, 0]
            block_max = center + [
                self.pre_transform.block_size / 2.0, self.pre_transform.block_size / 2.0, 0]
            point_idxs = np.where((
                points[:, 0] >= block_min[0]) & (
                    points[:, 0] <= block_max[0]) & (
                        points[:, 1] >= block_min[1]) & (
                            points[:, 1] <= block_max[1]))[0]
            if point_idxs.size > 1024:
                break
        return point_idxs, center
    
    def select_and_center(self, point_idxs, points, center, room_idx, labels, room_coord_min, room_coord_max):
        if point_idxs.size >= self.pre_transform.num_point:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=False)
        else:
            selected_point_idxs = np.random.choice(point_idxs, self.num_point, replace=True)

        # normalize
        selected_points = points[selected_point_idxs, :]  # num_point * 6
        current_points = np.zeros((self.num_point, 9))  # num_point * 9
        current_points[:, 6] = selected_points[:, 0] / room_coord_max[room_idx][0]
        current_points[:, 7] = selected_points[:, 1] / room_coord_max[room_idx][1]
        current_points[:, 8] = selected_points[:, 2] / room_coord_max[room_idx][2]
        selected_points[:, 0] = selected_points[:, 0] - center[0]
        selected_points[:, 1] = selected_points[:, 1] - center[1]
        selected_points[:, 3:6] /= 255.0
        current_points[:, 0:6] = selected_points
        current_labels = labels[selected_point_idxs]

        return current_points, current_labels

    @property
    def raw_file_names(self):
        return ["sequences"]

    @property
    def processed_room_dirs(self):
        return [s for s in self.AVAILABLE_SPLITS[:-1]]

    @property
    def processed_paths(self) -> List[str]:
        r"""The absolute filepaths that must be present in order to skip
        processing."""
        return self._processed_paths

    def get(self, idx):
        data = torch.load(self.processed_paths[self.split][idx])
        return data

    def __getitem__(self, idx):
        data = self.get(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def __len__(self):
        return len(self.processed_paths[self.split])


class S3DISDataset(BaseDataset):
    """ Wrapper around Semantic Kitti that creates train and test datasets.
    Parameters
    ----------
    dataset_opt: omegaconf.DictConfig
        Config dictionary that should contain
            - root,
            - split,
            - transform,
            - pre_transform
            - process_workers
    """

    def __init__(self, dataset_opt):
        super().__init__(dataset_opt)
        
        process_workers: int = dataset_opt.process_workers if dataset_opt.process_workers else 0
        self.train_dataset = S3DIS(
            self._data_path,
            split="train",
            transform=self.train_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

        self.val_dataset = S3DIS(
            self._data_path,
            split="val",
            transform=self.val_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )

        self.test_dataset = S3DIS(
            self._data_path,
            split="test",
            transform=self.test_transform,
            pre_transform=self.pre_transform,
            process_workers=process_workers,
        )