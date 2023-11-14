from scipy.spatial.transform import Rotation
from pytorch3d.ops import knn_points
import shutil
import torch
import os
from collections import defaultdict
import numpy as np
import matplotlib.colors as mcolors
from pytorch3d.ops import box3d_overlap
from lapsolver import solve_dense
from pyarrow import feather
import pandas as pd
import copy
from av2.geometry.se3 import SE3
from av2.structures.cuboid import Cuboid

_class_dict = {
    1: 'REGULAR_VEHICLE',
    2: 'PEDESTRIAN',
    3: 'BOLLARD',
    4: 'CONSTRUCTION_CONE',
    5: 'CONSTRUCTION_BARREL',
    6: 'STOP_SIGN',
    7: 'BICYCLE',
    8: 'LARGE_VEHICLE',
    9: 'WHEELED_DEVICE',
    10: 'BUS',
    11: 'BOX_TRUCK',
    12: 'SIGN',
    13: 'TRUCK',
    14: 'MOTORCYCLE',
    15: 'BICYCLIST',
    16: 'VEHICULAR_TRAILER',
    17: 'TRUCK_CAB',
    18: 'MOTORCYCLIST',
    19: 'DOG',
    20: 'SCHOOL_BUS',
    21: 'WHEELED_RIDER',
    22: 'STROLLER',
    23: 'ARTICULATED_BUS',
    24: 'MESSAGE_BOARD_TRAILER',
    25: 'MOBILE_PEDESTRIAN_SIGN',
    26: 'WHEELCHAIR',
    27: 'RAILED_VEHICLE',
    28: 'OFFICIAL_SIGNALER',
    29: 'TRAFFIC_LIGHT_TRAILER',
    30: 'ANIMAL',
    31: 'MOBILE_PEDESTRIAN_CROSSING_SIGN'}


cols = list(mcolors.CSS4_COLORS.keys())[15:]
cols = [c for i, c in enumerate(cols) if i % 10 ==0]

cols = [
    'brown',
    'red',
    'teal',
    'blue',
    'midnightblue',
    'fuchsia',
    'crimson',
    'mediumvioletred',
    'darkgreen',
    'dodgerblue',
    'lime',
    'darkgoldenrod',
    'orange',
    'deeppink',
    'darkslategray',
    'pink',
    'gold',
    'darkblue',
    'limegreen',
    'green',
    'yellow',
    'darkorange',
    'purple',
    'magenta']

cols = cols + cols + cols + cols + cols + cols + cols + cols

column_names = [
    'timestamp_ns',
    'track_uuid',
    'category',
    'length_m',
    'width_m',
    'height_m',
    'qw',
    'qx',
    'qy',
    'qz',
    'tx_m',
    'ty_m',
    'tz_m',
    'num_interior_pts']

column_names_dets_wo_traj = [
    'tx_m',
    'ty_m',
    'tz_m',
    'length_m',
    'width_m',
    'height_m',
    'qw',
    'qx',
    'qy',
    'qz',
    'timestamp_ns',
    'category',
    'gt_id',
    'num_interior_pts',
    'pts_density',
    'log_id',
    'rot',
    'gt_cat']

column_names_dets = column_names_dets_wo_traj # + [f'{i}_{j}' for i in range(25) for j in ['x', 'y', 'z']]

column_dtypes = {
    'timestamp_ns': 'int64',
    'track_uuid': 'int32',
    'length_m': 'float32',
    'width_m': 'float32',
    'height_m': 'float32',
    'qw': 'float32',
    'qx': 'float32',
    'qy': 'float32',
    'qz': 'float32',
    'tx_m': 'float32',
    'ty_m': 'float32',
    'tz_m': 'float32'}

column_dtypes_dets_wo_traj = {
    'timestamp_ns': 'int64',
    'length_m': 'float32',
    'width_m': 'float32',
    'height_m': 'float32',
    'qw': 'float32',
    'qx': 'float32',
    'qy': 'float32',
    'qz': 'float32',
    'tx_m': 'float32',
    'ty_m': 'float32',
    'tz_m': 'float32',
    'gt_id': 'str',
    'num_interior_pts': 'int64'}

column_dtypes_dets = dict() # {f'{i}_{j}': 'float32' for i in range(25) for j in ['x', 'y', 'z']}
column_dtypes_dets.update(column_dtypes_dets_wo_traj)



class Track():
    def __init__(self, detection, track_id, every_x_frame=-1, overlap=-1) -> None:
        self.detections = [detection]
        detection.track_id = track_id
        self.inactive_count = 0
        self.track_id = track_id
        self.log_id = detection.log_id
        self.dead = False
        self.final = list()
        self.max_num_interior = detection.num_interior
        self.min_num_interior = detection.num_interior

    def add_detection(self, detection):
        self.detections.append(detection)
        detection.track_id = self.track_id
        if detection.num_interior > self.max_num_interior:
            self.max_num_interior = detection.num_interior
        if detection.num_interior < self.min_num_interior:
            self.min_num_interior = detection.num_interior
    
    def fill_detections(self, av2_loader, ordered_timestamps, max_time):
        filled_detections = list()
        for i, det in enumerate(self.detections):
            filled_detections.append(det)
            if i != len(self.detections)-1:
                t0 = self.detections[i].timestamps[0, 0].item()
                t1 = self.detections[i+1].timestamps[0, 0].item()
                _range = ordered_timestamps.index(t1) - ordered_timestamps.index(t0)
            else:
                _range = max_time if max_time != -1 else det.trajectory.shape[1]-1 

            city_SE3_ego0 = av2_loader.get_city_SE3_ego(self.log_id, det.timestamps[0, 0].item())
            for time in range(1, _range):
                points_c_time = det.canonical_points + det.trajectory[:, time, :]
                city_SE3_ego = av2_loader.get_city_SE3_ego(self.log_id, det.timestamps[0, time].item())
                ego_SE3_ego0 = city_SE3_ego.inverse().compose(city_SE3_ego0)
                points_c_time = ego_SE3_ego0.transform_point_cloud(points_c_time)

                if time < det.trajectory.shape[1] - 1:
                    traj = det.trajectory[:, time:, :]
                else:
                    traj = det.trajectory[:, time-1:, :]
                
                filled_detections.append(Detection(
                    trajectory=traj,
                    canonical_points=points_c_time,
                    timestamps=self.detections[i].timestamps[0, time:],
                    log_id=det.log_id,
                    num_interior=det.num_interior,
                    gt_id=det.gt_id,
                    gt_cat=det.gt_cat,
                    lwh=det.lwh))

        self.detections = filled_detections
    
    def _sort_dets(self):
        by_time = {d.timestamps[0, 0]: d for d in self.detections}
        self.detections = [by_time[k] for k in sorted(list(by_time.keys()))]

    def _get_traj(self, i=-1):
        return self.detections[i].trajectory
    
    def _get_canonical_points(self, i=-1):
        return self.detections[i].canonical_points
    
    def _get_propagated_bbs(self, propagated_pos_tracks):
        return get_propagated_bbs(propagated_pos_tracks)
    
    def _get_whole_traj_and_convert_time(self, t1, av2_loader, i=-1, city=False, max_time=25):
        index = torch.where(self.detections[-1].timestamps[0]==t1)[0][0].item()
        traj = copy.deepcopy(self._get_traj(i))[:, index:]
        cano = self._get_canonical_points()
        traj = torch.tile(cano.unsqueeze(1), (1, traj.shape[1], 1)) + traj
        
        t0 = self.detections[i].timestamps[0, 0].item()
        if not city:
            traj = self._convert_time(t0, t1, av2_loader, traj)
        else:
            traj = self._convert_city(t0, av2_loader, traj)

        return traj[:, :max_time, :]
    
    def _get_canonical_points_and_convert_time(self, t1, av2_loader, i=-1, city=False):
        canonical_points = copy.deepcopy(self._get_canonical_points(i=i))
        t0 = self.detections[i].timestamps[0, 0].item()
        if not city:
            canonical_points = self._convert_time(t0, t1, av2_loader, canonical_points)
        else:
            canonical_points = self._convert_city(t0, av2_loader, canonical_points)
        # print(self.log_id, self.track_id, t1, t0, canonical_points[0])
        return canonical_points
        
    def t1_SE3_t0(self,t0, t1, av2_loader):
        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)
        city_SE3_t1 = av2_loader.get_city_SE3_ego(self.log_id, t1)
        t1_SE3_t0 = city_SE3_t1.inverse().compose(city_SE3_t0)
        return t1_SE3_t0

    def _convert_time(self, t0, t1, av2_loader, points):
        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)
        city_SE3_t1 = av2_loader.get_city_SE3_ego(self.log_id, t1)
        t1_SE3_t0 = city_SE3_t1.inverse().compose(city_SE3_t0)

        return t1_SE3_t0.transform_point_cloud(points)
    
    def _convert_city(self, t0, av2_loader, points):
        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)

        return city_SE3_t0.transform_point_cloud(points)

    def __len__(self):
        if len(self.final):
            return len(self.final)
        return len(self.detections)
    
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.final):
            result = self.final[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration
        
    def resample(self, gt_path='/workspace/Waymo_Converted_train/train'):
        for d in self.detections:
            p = os.path.join(gt_path, self.log_id, 'sensors/lidar', f'{d.timestamp.item()}.feather')
            pc = feather.read_feather(p)
            pc = pc[pc['non_ground_pts_rc']]
            pc = pc[['x', 'y', 'z']].values
            ego_SE3_object = SE3(rotation=d.rot.numpy(), translation=d.translation.numpy())
            cuboid = Cuboid(
                dst_SE3_object=ego_SE3_object,
                length_m=d.lwh[0],
                width_m=d.lwh[1],
                height_m=d.lwh[2],
                category='REGULAR_VEHICLE',
                timestamp_ns=d.timestamp.item(),
                        track_id='-1'
                )
            pts, mask = cuboid.compute_interior_points(pc)
            d.resampled_pc = torch.from_numpy(pts)
            # print(d.canonical_points.shape, d.resampled_pc.shape)


class Detection():
    def __init__(self, trajectory, canonical_points, timestamps, log_id, num_interior, overlap=None, gt_id=None, gt_id_box=None, rot=None, alpha=None, gt_cat=-10, lwh=None, translation=None, pts_density=None, median_flow=False, min_area=False, median_center=False, resampled=None) -> None:
        self.trajectory = trajectory
        self.canonical_points = canonical_points
        self.timestamps = torch.atleast_2d(timestamps)
        self.log_id = log_id
        self.num_interior = num_interior
        # self.overlap = overlap
        self.gt_id = gt_id
        self.gt_cat = gt_cat
        self.gt_id_box = gt_id
        self.length = trajectory.shape[0] if len(trajectory.shape) < 3 else trajectory.shape[1]
        self.track_id = 0
        self.pts_density = pts_density
        self.median_flow = median_flow
        self.min_area = min_area  
        self.median_center = median_center
        self.resampled = resampled
        
        if resampled is not None:
            trajectory = trajectory[~(resampled.bool())]
        else:
            trajectory = trajectory
        
        if trajectory.shape[0] == 0 and rot is None:
            self.rot, self.alpha, _lwh, _translation = get_min_area(canonical_points)
        else:
            if rot is not None:
                self.rot = rot
                self.alpha = alpha
            else:
                self.rot, self.alpha = self.get_alpha_rot_t0_to_t1(0, 1, trajectory)
        
            if lwh is None or translation is None:
                if not self.min_area:
                    _lwh, _translation = self.get_rotated_center_and_lwh(canonical_points, self.rot)
                else:
                    self.rot, self.alpha, _lwh, _translation = get_min_area(canonical_points, self.alpha)
        
        self.lwh = lwh if lwh is not None else _lwh
        self.translation = translation if translation is not None else _translation
        if pts_density is None:
            self.pts_density = ((self.lwh[0] * self.lwh[1] * self.lwh[2]) / self.num_interior).item()
        else:
            self.pts_density = pts_density
        
        if len(self.trajectory.shape) > 2:
            self.mean_trajectory = torch.mean(self.trajectory, dim=0)
        else:
            self.mean_trajectory = self.trajectory

    @property
    def timestamp(self):
        return self.timestamps[0, 0]
    
    @property
    def traj(self):
        return self.trajectory

    @property
    def rotation(self):
        return self.rot

    def update_after_registration(self, canonical_points_registered, translation, rotation, max_lwh):
        # new_SE3_old = SE3(rotation=rotation, translation=translation)
        # ego_SE3_old = SE3(rotation=self.rot.cpu().numpy(), translation=self.translation.cpu().numpy())
        # chained = ego_SE3_old.compose(new_SE3_old.inverse())
        # self.rot = torch.from_numpy(chained.rotation).to(self.rot.device)
        # self.lwh, self.translation = self.get_rotated_center_and_lwh(canonical_points_registered, self.rot, self.median_center) #, translation=torch.from_numpy(chained.translation).to(self.rot.device))
        self.lwh = max_lwh
        self.translation = self.translation + translation
        self.num_interior = canonical_points_registered.shape[0]
        self.pts_density = ((self.lwh[0] * self.lwh[1] * self.lwh[2]) / self.num_interior).item()

    def get_alpha_rot_t0_to_t1(self, t0=None, t1=None, trajectory=None, traj_t0=None, traj_t1=None):
        rot, alpha = get_alpha_rot_t0_to_t1(t0, t1, trajectory, traj_t0, traj_t1, median=self.median_flow)
        return rot, alpha
    
    def get_rotated_center_and_lwh(self, canonical_points, rot=None, trajectory=None, traj_t0=None, traj_t1=None, translation=None):
        if rot is None:
            rot, _ = self.get_alpha_rot_t0_to_t1(0, 1, trajectory, traj_t0, traj_t1, median=self.median_flow)
        lwh, translation = get_rotated_center_and_lwh(canonical_points, rot, translation, self.median_center)
        return lwh, translation
    
    def _get_propagated_bbs(self, propagated_pos=None):
        if propagated_pos is None:
            propagated_pos = self.trajectory + torch.tile(
                self._get_canonical_points.unsqueeze(1),
                (1, self.trajectory.shape[1], 1))
        return get_propagated_bbs(propagated_pos)

    def _get_traj_city(self, av2_loader, t0):
        traj = self.trajectory
        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)
        return city_SE3_t0.transform_point_cloud(traj)
    
    def _get_canonical_points(self):
        canonical_points = self.canonical_points
        return canonical_points

    def get_canonical_points_city(self, av2_loader, t0):
        canonical_points = self._get_canonical_points()
        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)
        # if  t0 == 1507677181475568 or t0 == 1507677182775411 or t0 == 1507677182975295:
        #     print(city_SE3_t0.translation)
        return city_SE3_t0.transform_point_cloud(canonical_points)


def get_rotated_center_and_lwh(pc, rot, translation=None, median=False):
    # translation = get_center(pc)
    # translation = translation.cpu()
    translation = get_center(pc, median)
    if type(translation) == np.ndarray:
        translation = torch.from_numpy(translation).to(pc.device)
    ego_SE3_object = SE3(rotation=rot.cpu().numpy(), translation=translation.cpu().numpy())
    if type(pc) == torch.Tensor:
        pc_obj = ego_SE3_object.inverse().transform_point_cloud(pc.cpu().numpy())
        pc_obj = torch.from_numpy(pc_obj).to(pc.device)
    else:
        pc_obj = ego_SE3_object.inverse().transform_point_cloud(pc)
    lwh = get_lwh(pc_obj)
    ''' 
    pc = pc @ rot.T # + (-translation.double() @ rot.T)
    translation = get_center(pc)
    translation = translation.to(pc.device)
    # rot.T @ translation not needed since taken from already rotated pc
    pc = pc + (-translation.double())
    lwh = get_lwh(pc)
    # but translatoin needs to be rotated to get correct translation
    translation = rot.T @ translation.double()
    '''
    return lwh, translation


def get_alpha_rot_t0_to_t1(t0=None, t1=None, trajectory=None, traj_t0=None, traj_t1=None, median=False):
    if trajectory is not None:
        if median:
            mean_flow = (trajectory[:, t1, :] - trajectory[:, t0, :]).median(dim=0).values
        else:
            mean_flow = (trajectory[:, t1, :] - trajectory[:, t0, :]).mean(dim=0) # (trajectory[:, t1, :] - trajectory[:, t0, :]).median(dim=0)
    else:
        if median:
            mean_flow = (traj_t1 - traj_t0).median(dim=0).values
        else:
            mean_flow = (traj_t1 - traj_t0).mean(dim=0) # (traj_t1 - traj_t0).median(dim=0)
    alpha = torch.atan2(mean_flow[1], mean_flow[0])
    rot = torch.tensor([
        [torch.cos(alpha), -torch.sin(alpha), 0],
        [torch.sin(alpha), torch.cos(alpha), 0],
        [0, 0, 1]]).double()
    return rot, alpha


def get_propagated_bbs(propagated_pos_tracks, get_trans=True, get_lwh=True, _2d=False, get_corners=True, get_alphas=True, get_xylwa=True):
    lwhs = list()
    translations = list()
    corners = list()
    alphas = list()
    xylwa = list()
    for t in range(propagated_pos_tracks.shape[1]):
        t_0 = t if t != propagated_pos_tracks.shape[1]-1 else t-1
        t_1 = t+1 if t != propagated_pos_tracks.shape[1]-1 else t
        rot, alpha = get_alpha_rot_t0_to_t1(t_0, t_1, propagated_pos_tracks)
        rot = rot.to(propagated_pos_tracks.device)
        lwh, translation = get_rotated_center_and_lwh(propagated_pos_tracks[:, t, :].double(), rot)
        # if type(translation) == torch.Tensor:
        #     translation = translation.cpu()
        # if type(alpha)  == torch.Tensor:
        #     alpha = alpha.cpu()
        if _2d:
            lwh = lwh[:-1]
            translation = translation[:-1]
        lwhs.append(lwh)
        translations.append(translation)
        corners.append(torch.cat([translation - lwh/2, translation + lwh/2]))
        alphas.append(alpha)
        xylwa.append(torch.cat([translation, lwh, alpha.unsqueeze(0)]))
    translations = torch.stack(translations)
    lwhs = torch.stack(lwhs)
    corners = torch.stack(corners)
    alphas = torch.stack(alphas)
    xylwa = torch.stack(xylwa)
    return_dict = dict()
    if get_trans:
        return_dict['translation'] = translations
    if get_lwh:
        return_dict['lwh'] = lwhs
    if get_corners:
        return_dict['corners'] = corners
    if get_alphas:
        return_dict['alphas'] = alphas
    if get_xylwa:
        return_dict['xylwa'] = xylwa
    return return_dict


def get_center(canonical_points, median):
    if not median:
        points_c_time = canonical_points
        mins, maxs = points_c_time.min(dim=0), points_c_time.max(dim=0)
        translation = (maxs.values + mins.values)/2
    else:
        # translation = torch.mean(canonical_points, dim=0) # np.median(canonical_points.cpu().numpy(), axis=0)
        translation = np.median(canonical_points.cpu().numpy(), axis=0)
    return translation


def get_lwh(object_points):
    points_c_time = object_points
    mins, maxs = points_c_time.min(dim=0), points_c_time.max(dim=0)
    lwh = maxs.values - mins.values

    return lwh


def store_initial_detections(detections, seq, out_path, split, tracks=False, gt_path=None):
    p = f'{out_path}/{split}/{seq}'
    #if os.path.isdir(p):
    #    shutil.rmtree(p)
    os.makedirs(p, exist_ok=True)
    
    if tracks:
        extracted_detections = dict()
        for k, t in detections.items():
            extracted_detections[k] = t.detections
        detections =  extracted_detections
    elif gt_path is not None:
        if type(list(detections.keys())[0]) == int:
            detections = {k: v for k, v in detections.items()}
        else:
            detections = {k.item(): v for k, v in detections.items()}
        gt = load_gt(seq, gt_path)
        detections = assign_gt(detections, gt)
    for _, t in enumerate(detections):
        for j, d in enumerate(detections[t]):
            np.savez(
                os.path.join(p, str(t) + '_' + str(j) + '.npz'),
                trajectory=d.trajectory.numpy() if d.trajectory is not None else d.trajectory,
                canonical_points=d.canonical_points.numpy() if d.canonical_points is not None else d.canonical_points,
                timestamps=np.atleast_2d(d.timestamps.numpy()),
                log_id=d.log_id,
                num_interior=d.num_interior,
                # overlap=d.overlap,
                gt_id=d.gt_id,
                gt_id_box=d.gt_id_box if gt_path is not None else d.gt_id,
                track_id=d.track_id,
                rot=d.rot.numpy(),
                alpha=d.alpha.numpy(),
                gt_cat=d.gt_cat,
                lwh=d.lwh.numpy(),
                translation=d.translation.numpy()
            )


def load_initial_detections(out_path, split, seq=None, tracks=False, every_x_frame=1, overlap=1):
    p = f'{out_path}/{split}/{seq}'
    detections = defaultdict(list)
    if tracks:
        detections = defaultdict(dict)
    if not os.path.isdir(p):
        return detections
    print('loading...', len(os.listdir(p)))
    for i, d in enumerate(os.listdir(p)):
        dict_key = int(d.split('_')[0])

        if tracks:
            sorter = int(d.split('_')[1].split('.')[0])
        try:
            d = np.load(os.path.join(p, d), allow_pickle=True)
        except:
            print(os.path.join(p, d))
            quit()
        d = Detection(
            torch.from_numpy(d['trajectory']), 
            torch.from_numpy(d['canonical_points']) if d['canonical_points'] is not None else d['canonical_points'], 
            torch.atleast_2d(torch.from_numpy(d['timestamps'])),
            d['log_id'].item(),
            d['num_interior'].item(),
            # d['overlap'].item(),
            d['gt_id'].item(),
            d['gt_id_box'].item() if 'gt_id_box' in d.keys() else d['gt_id'].item(),
            rot=torch.from_numpy(d['rot']) if 'rot' in d.keys() else None,
            alpha=torch.from_numpy(d['alpha']) if 'alpha' in d.keys() else None,
            gt_cat=d['gt_cat'].item() if 'gt_cat' in d.keys() else -10,
            lwh=torch.from_numpy(d['lwh']) if 'lwh' in d.keys() else None,
            translation=torch.from_numpy(d['translation']) if 'translation' in d.keys() else None,
        )
        # if d.lwh[0] < 0.1 or d.lwh[1] < 0.1 or d.lwh[2] < 0.1:
        #     continue
        if not tracks:
            if not len(d.timestamps.shape):
                detections[d.timestamps.item()].append(d)
            else:
                detections[d.timestamps[0, 0].item()].append(d)
        else:
            detections[dict_key][d.timestamps[0, 0].item()] = d

    if tracks:
        tracks = dict()
        for track_id, dets in detections.items():
            for i, k in enumerate(sorted(list(dets.keys()))):
                d = dets[k]
                if i == 0:
                    t = Track(d, track_id, every_x_frame, overlap)
                else:
                    t.add_detection(d)
            tracks[track_id] = t
        detections = tracks
    return detections


def load_gt(seq_name, data_root_path):
    """
    Load MOT ground truth file
    """
    seq_path = os.path.join(str(data_root_path), seq_name)  # Sequence path
    gt_file_path = os.path.join(seq_path, "annotations.feather")   # Ground truth file

    gt_df = feather.read_feather(gt_file_path)

    return gt_df


def assign_gt(dets, gt, gt_assign_min_iou=0.25):
    """
    Assigns a GT identity to every detection in self.det_df, based on the ground truth boxes in self.gt_df.
    The assignment is done frame by frame via bipartite matching.
    """
    gt = gt[gt['category'] != '3']
    for time, time_dets in dets.items():
        frame_gt = gt[gt.timestamp_ns == time]

        # Compute IoU for each pair of detected / GT bounding box
        dets_trans = torch.stack([d.translation for d in time_dets])
        dets_lwh = torch.stack([d.lwh for d in time_dets])
        dets_rot = torch.stack([d.rot for d in time_dets])
        det_boxes = list()
        for trans, lwh, rot in zip(dets_trans, dets_lwh, dets_rot):
            det_boxes.append(_create_box(trans, lwh, rot))
        gt_boxes = list()
        for trans, lwh, alpha in zip(
                torch.from_numpy(frame_gt[['tx_m', 'ty_m', 'tz_m']].values),
                torch.from_numpy(frame_gt[['length_m', 'width_m', 'height_m']].values),
                torch.from_numpy(np.arccos(frame_gt['qw'].values)*2)):
            rot = torch.tensor([
                    [torch.cos(alpha), torch.sin(alpha), 0],
                    [torch.sin(alpha), torch.cos(alpha), 0],
                    [0, 0, 1]])
            gt_boxes.append(_create_box(trans, lwh, rot))
        
        det_boxes = torch.stack(det_boxes)
        gt_boxes = torch.stack(gt_boxes)
        ''' 
        for d in det_boxes:
            _, iou_matrix = box3d_overlap(
                d.cuda().unsqueeze(0),
                d.cuda().unsqueeze(0),
                eps=1e-6)
        '''
        _, iou_matrix = box3d_overlap(
            det_boxes.cuda(),
            gt_boxes.cuda(),
            eps=1e-6)
        
        iou_matrix[iou_matrix < gt_assign_min_iou] = np.nan
        dist_matrix = 1 - iou_matrix
        assigned_detect_ixs, assigned_detect_ixs_ped_ids = solve_dense(dist_matrix.cpu().numpy())
        unassigned_detect_ixs = np.array(list(set(range(len(time_dets))) - set(assigned_detect_ixs)))
        assigned_detect_ixs_ped_ids = frame_gt.iloc[assigned_detect_ixs_ped_ids]['track_uuid'].values

        for i, idx in enumerate(assigned_detect_ixs):
            time_dets[idx].gt_id_box = assigned_detect_ixs_ped_ids[i]
        for idx in unassigned_detect_ixs:
            time_dets[idx].gt_id_box = -1
    return dets
    

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

def _from_box(vertices):
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


def to_feather(detections, log_id, out_path, split, rank, precomp_dets=False, name=''):
    track_vals = list()
    if precomp_dets:
        store_initial_detections(detections, seq=log_id, out_path=f'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/tracks/tracks/initial_dets/{name}', split=split)
    # per timestamp detections
    for i, timestamp in enumerate(sorted(detections.keys())):
        dets = detections[timestamp]
        for det in dets:

            # only keep bounding boxes with lwh > 0 
            # necessay also for 3DIoU
            if det.lwh[0] < 0.1 or det.lwh[1] < 0.1 or det.lwh[2] < 0.1:
                continue

            # quaternion rotation around z axis
            # quat = torch.tensor([torch.cos(det.alpha/2), 0, 0, torch.sin(det.alpha/2)]).numpy()
            quat = Rotation.from_euler('z', det.alpha).as_quat()
            # REGULAR_VEHICLE = only dummy class
            values = [
                det.translation[0].item(),
                det.translation[1].item(),
                det.translation[2].item(),
                det.lwh[0].item(),
                det.lwh[1].item(),
                det.lwh[2].item(),
                quat[3],
                quat[0],
                quat[1],
                quat[2],
                int(det.timestamp.item()) if type(det.timestamp) is not int else det.timestamp,
                'REGULAR_VEHICLE',
                det.gt_id,
                det.num_interior,
                det.pts_density,
                det.log_id,
                det.alpha.item(),
                det.gt_cat] # + det.mean_trajectory.flatten().numpy().tolist()
            track_vals.append(values)
    track_vals = np.asarray(track_vals)

    if track_vals.shape[0] == 0:
        return False

    df = pd.DataFrame(
        data=track_vals,
        columns=column_names_dets)
    df = df.astype(column_dtypes_dets)
    detections = dict()

    os.makedirs(os.path.join(out_path, split, log_id), exist_ok=True)
    # os.makedirs(os.path.join(out_path, split, 'feathers'), exist_ok=True)
    write_path = os.path.join(out_path, split, log_id, 'annotations.feather') # os.path.join(out_path, split, 'feathers', f'all_{rank}.feather') 

    # if os.path.isfile(write_path):
    #     df_all = feather.read_feather(write_path)
    #     df = df_all.append(df)
    # else:
    #     df = df

    feather.write_feather(df, write_path)
    return True


def outlier_removal(pc, threshold=0.1, kNN=10):
    nn_dists, nn_idx, nn = knn_points(
        pc.unsqueeze(0).float(),
        pc.unsqueeze(0).float(),
        K=kNN)
    return pc[nn_dists[0,:,1:].mean(1) < threshold], nn_dists[0,:,1:].mean(1) < threshold

def get_min_area(pc, init_alpha=None, median=False):
    rot = None
    lwh = None
    translation = None
    min_area = 10000
    _alpha = init_alpha

    if _alpha is None:
        alpha_list =  [torch.pi*0, torch.pi/4, torch.pi/2, torch.pi*3/4]
        for _alpha in alpha_list:
            _alpha = torch.tensor([_alpha])
            _rot = torch.tensor([
                [torch.cos(_alpha), -torch.sin(_alpha), 0],
                [torch.sin(_alpha), torch.cos(_alpha), 0],
                [0, 0, 1]]).double()
            _lwh, _translation = get_rotated_center_and_lwh(pc, _rot, median)
            if _lwh[0] * _lwh[1] < min_area:
                alpha = _alpha
                rot = _rot
                lwh = _lwh
                translation = _translation
                min_area = _lwh[0] * _lwh[1]
        _alpha = alpha
    
    for i in [4, 8, 16, 32, 64]:
        alpha_list = [_alpha-torch.pi/i, _alpha, _alpha+torch.pi/i]
        for _alpha in alpha_list:
            _rot = torch.tensor([
                [torch.cos(_alpha), -torch.sin(_alpha), 0],
                [torch.sin(_alpha), torch.cos(_alpha), 0],
                [0, 0, 1]]).double()
            _lwh, _translation = get_rotated_center_and_lwh(pc, _rot, median)
            if _lwh[0] * _lwh[1] < min_area:
                alpha = _alpha
                rot = _rot
                lwh = _lwh
                translation = _translation
                min_area = _lwh[0] * _lwh[1]
        _alpha = alpha
    return rot, alpha.squeeze(), lwh, translation
