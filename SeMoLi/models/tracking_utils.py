from scipy.spatial.transform import Rotation
from pytorch3d.ops import knn_points
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


class Detection():
    def __init__(self, trajectory, canonical_points, timestamps, log_id, num_interior, gt_id=None, rot=None, alpha=None, gt_cat=-10, lwh=None, translation=None, pts_density=None, median_flow=False, min_area=False, median_center=False, resampled=None) -> None:
        self.trajectory = trajectory
        self.canonical_points = canonical_points
        self.timestamps = torch.atleast_2d(timestamps)
        self.log_id = log_id
        self.num_interior = num_interior
        self.gt_id = gt_id
        self.gt_cat = gt_cat
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
        translation = np.median(canonical_points.cpu().numpy(), axis=0)
    return translation


def get_lwh(object_points):
    points_c_time = object_points
    mins, maxs = points_c_time.min(dim=0), points_c_time.max(dim=0)
    lwh = maxs.values - mins.values

    return lwh
    

def to_feather(detections, log_id, out_path, split):
    track_vals = list()
    
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
    write_path = os.path.join(out_path, split, log_id, 'annotations.feather') 

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
