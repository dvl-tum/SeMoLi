import torch
import os
from collections import defaultdict
import numpy as np
import matplotlib.colors as mcolors


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
    'num_interior_pts',
    'pts_density',
    'log_id']

column_names_dets = column_names_dets_wo_traj + [f'{i}_{j}' for i in range(25) for j in ['x', 'y', 'z']]

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
    'num_interior_pts': 'int64'}

column_dtypes_dets = {f'{i}_{j}': 'float32' for i in range(25) for j in ['x', 'y', 'z']}.update(column_dtypes_dets_wo_traj)



class Track():
    def __init__(self, detection, track_id, every_x_frame, overlap) -> None:
        self.detections = [detection]
        detection.track_id = track_id
        self.inactive_count = 0
        self.track_id = track_id
        self.log_id = detection.log_id
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.final = list()
        self.dead = False
    
    def add_detection(self, detection):
        self.detections.append(detection)
        detection.track_id = self.track_id
    
    def final_detections(self, av2_loader):
        for i, det in enumerate(self.detections):
            if i != len(self.detection) or self.inactive_count != 0:
                _range = self.every_x_frame
            else:
                _range = det.trajectory.shape[1]

            city_SE3_ego0 = av2_loader.get_city_SE3_ego(self.log_id, det.timestamps[0, 0].item())

            for time in range(_range):
                points_c_time = det.canonical_points + det.trajectory[:, time, :]

                city_SE3_ego = av2_loader.get_city_SE3_ego(self.log_id, det.timestamps[0, time].item())
                ego_SE3_ego0 = city_SE3_ego.inverse().compose(city_SE3_ego0)
                points_c_time = ego_SE3_ego0.transform_point_cloud(points_c_time)

                mins, maxs = points_c_time.min(axis=0), points_c_time.max(axis=0)
                lwh = maxs - mins

                translation = (maxs + mins)/2
                if time < det.trajectory.shape[1] - 1:
                    mean_flow = (det.trajectory[:, time+1, :] - det.trajectory[:, time, :]).mean(axis=0)
                else:
                    mean_flow = (det.trajectory[:, time, :] - det.trajectory[:, time-1, :]).mean(axis=0)
                alpha = torch.arctan(mean_flow[1]/mean_flow[0])
                rot = torch.array([
                    [torch.cos(alpha), torch.sin(alpha), 0],
                    [torch.sin(alpha), torch.cos(alpha), 0],
                    [0, 0, 1]])

                # rotate bounding box to get lwh in object coordinate system
                pc = torch.stack([
                    self.translation + torch.tensor([0.5, 0, 0]) * self.lwh,
                    self.translation + torch.tensor([-0.5, 0, 0]) * self.lwh,
                    self.translation + torch.tensor([0, 0.5, 0]) * self.lwh,
                    self.translation + torch.tensor([0, -0.5, 0]) * self.lwh,
                    self.translation + torch.tensor([0, 0, 0.5]) * self.lwh,
                    self.translation + torch.tensor([0, 0, -0.5]) * self.lwh]).double()
                pc = pc @ rot.T + (-self.translation @ rot.T)
                lwh, _ = get_center_and_lwh(pc)

                pts_density = (lwh[0] * lwh[1] * lwh[2]) / det.num_interior
                self.final.append(Detection(rot, translation, lwh, det.timestamps[0, time], det.log_id, det.num_interior, pts_density=pts_density))
    
    def _get_traj(self, i=-1):
        return self.detections[i].trajectory
    
    def _get_traj_and_convert_time(self, t1, av2_loader, i=-1, overlap=False, city=False):
        index = torch.where(self.detections[-1].timestamps[0]==t1)[0][0].item()
        traj = copy.deepcopy(self._get_traj(i))
        if overlap:
            start = self.inactive_count * (self.every_x_frame)
            end = start + self.overlap

            assert end < self.detections[i].length, 'inactive track too short'
            # +1 cos of indexing
            traj = traj[:, start+index:end+index]
        
        if overlap:
            traj_from_overlap = traj - torch.tile(
                traj[:, 0, :].unsqueeze(1), (1, traj.shape[1], 1))

        cano = self._get_canonical_points()
        traj = torch.tile(
                cano.unsqueeze(1), (1, traj.shape[1], 1)) + traj

        t0 = self.detections[i].timestamps[0, 0].item()
        if not city:
            traj = self._convert_time(t0, t1, av2_loader, traj)
        else:
            traj = self._convert_city(t0, av2_loader, traj)

        if overlap:
            return traj_from_overlap, traj
        else:
            return traj
    
    def _get_canonical_points(self, i=-1):
        return self.detections[i].canonical_points
    
    def _get_canonical_points_and_convert_time(self, t1, av2_loader, i=-1, overlap=False, city=False):
        canonical_points = copy.deepcopy(self._get_canonical_points(i=i))
        if overlap:
            start = (1 + self.inactive_count) * (self.every_x_frame)
            canonical_points += copy.deepcopy(self.detections[i].trajectory[:, start])
        t0 = self.detections[i].timestamps[0, 0].item()
        if not city:
            canonical_points = self._convert_time(t0, t1, av2_loader, canonical_points)
        else:
            canonical_points = self._convert_city(t0, av2_loader, canonical_points)
        # print(self.log_id, self.track_id, t1, t0, canonical_points[0])
        return canonical_points

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


class InitialDetection():
    def __init__(self, trajectory, canonical_points, timestamps, log_id, num_interior, overlap, gt_id) -> None:
        self.trajectory = trajectory
        self.canonical_points = canonical_points
        self.timestamps = timestamps
        self.log_id = log_id
        self.num_interior = num_interior
        self.overlap = overlap
        self.gt_id = gt_id
        self.length = trajectory.shape[1]
        self.track_id = 0
        
        self.rot, self.alpha = self.get_alpha_rot_t0_to_t1(0, 1, self.trajectory)
        self.lwh, self.translation = get_rotated_center_and_lwh(canonical_points, self.rot)
        self.mean_trajectory = torch.mean(self.trajectory, dim=0)
    
    def get_alpha_rot_t0_to_t1(self, t0=None, t1=None, trajectory=None, traj_t0=None, traj_t1=None):
        if t0 is not None:
            mean_flow = (trajectory[:, t1, :] - trajectory[:, t0, :]).mean(dim=0)
        else:
            mean_flow = (traj_t1 - traj_t0).mean(dim=0)
        alpha = torch.arctan(mean_flow[1]/mean_flow[0])

        rot = torch.tensor([
            [torch.cos(alpha), -torch.sin(alpha), 0],
            [torch.sin(alpha), torch.cos(alpha), 0],
            [0, 0, 1]]).double()
        return rot, alpha

    def _get_traj(self):
        traj = self.trajectory[:, 1:self.overlap+1]
        return traj

    def _get_traj_city(self, av2_loader, t0):
        traj = self._get_traj()
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

    def final_detection(self):
        pts_density = (self.lwh[0] * self.lwh[1] * self.lwh[2]) / self.num_interior
        self.final = Detection(self.rot, self.alpha, self.translation, self.lwh, self.timestamps[0, 0], self.log_id, self.num_interior, pts_density=pts_density, trajectory=self.mean_trajectory)
        return self.final


def get_rotated_center_and_lwh(pc, rot):
    lwh, translation = get_center_and_lwh(pc)
    translation = translation.cpu()
    lwh = lwh.cpu()
    pc = torch.stack([
        translation + torch.tensor([0.5, 0, 0]) * lwh,
        translation + torch.tensor([-0.5, 0, 0]) * lwh,
        translation + torch.tensor([0, 0.5, 0]) * lwh,
        translation + torch.tensor([0, -0.5, 0]) * lwh,
        translation + torch.tensor([0, 0, 0.5]) * lwh,
        translation + torch.tensor([0, 0, -0.5]) * lwh]).double()
    pc = pc @ rot.T + (-translation.double() @ rot.T)
    lwh, _ = get_center_and_lwh(pc)
    return lwh, translation


def get_center_and_lwh(canonical_points, lwh=None, translation=None):
    points_c_time = canonical_points
    mins, maxs = points_c_time.min(dim=0), points_c_time.max(dim=0)
    if lwh is None:
        lwh = maxs.values - mins.values
        translation = (maxs.values + mins.values)/2
    else: 
        lwh = lwh
        translation = translation

    return lwh, translation


class Detection():
    def __init__(self, rotation, heading, translation, lwh, timestamp, log_id, num_interior, pts_density, trajectory) -> None:
        self.rotation = rotation
        self.heading = heading
        self.translation = translation
        self.lwh = lwh
        self.timestamp = timestamp
        self.log_id = log_id
        self.num_interior = num_interior
        self.pts_density = pts_density
        self.trajectory = trajectory.reshape(trajectory.shape[0]*trajectory.shape[1])


class CollapsedDetection():
    def __init__(self, rotation, heading, translation, lwh, traj, canonical_points, timestamp, log_id, num_interior, overlap, pts_density) -> None:
        self.rotation = rotation
        self.heading = heading
        self.translation = translation
        self.lwh = lwh
        self.traj = traj
        self.canonical_points = canonical_points
        self.timestamp = timestamp
        self.log_id = log_id
        self.num_interior = num_interior
        self.overlap = overlap
        self.track_id = 0
        self.pts_density = pts_density
            

def store_initial_detections(detections, seq, tracks=False):
    if not tracks:
        p = f'{os.getcwd()}/../../../initial_dets/{seq}'
    else:
        p = f'{os.getcwd()}/../../../initial_tracks/{seq}'
    os.makedirs(p, exist_ok=True)
    
    if tracks:
        extracted_detections = dict()
        for t in detections:
            extracted_detections[t.track_id] = t.detections
        detections =  extracted_detections
    else:
        detections = {k.item(): v for k, v in detections.items()}
        
    for _, t in enumerate(detections):
        for j, d in enumerate(detections[t]):
            np.savez(
                os.path.join(p, str(t) + '_' + str(j) + '.npz'),
                trajectory=d.trajectory.numpy(),
                canonical_points=d.canonical_points.numpy(),
                timestamps=d.timestamps.numpy(),
                log_id=d.log_id,
                num_interior=d.num_interior,
                overlap=d.overlap,
                gt_id=d.gt_id,
                length=d.length,
                track_id=d.track_id,
            )
    print(f'Stored {tracks}.....')

def load_initial_detections(tracks=False, seq=None, every_x_frame=1, overlap=1, hydra_add=True):
    if hydra_add:
        hydra_add = '../../../'
    else:
        hydra_add = ''

    if not tracks:
        p = f'{os.getcwd()}/{hydra_add}initial_dets/{seq}'
    else:
        p = f'{os.getcwd()}/{hydra_add}initial_tracks/{seq}'

    detections = defaultdict(list)
    for d in os.listdir(p):
        dict_key = int(d.split('_')[0])
        try:
            d = np.load(os.path.join(p, d))
        except:
            print(os.path.join(p, d))
            quit()

        d = InitialDetection(
            torch.from_numpy(d['trajectory']), 
            torch.from_numpy(d['canonical_points']), 
            torch.from_numpy(d['timestamps']),
            d['log_id'].item(),
            d['num_interior'].item(),
            d['overlap'].item(),
            d['gt_id'].item()
        )
        if not tracks:
            detections[d.timestamps[0, 0].item()].append(d)
        else:
            detections[dict_key].append(d)

    if tracks:
        tracks = list()
        for track_id, dets in detections.items():
            for i, d in enumerate(dets):
                if i == 0:
                    t = Track(d, track_id, every_x_frame, overlap)
                else:
                    t.add_detection(d)
            tracks.append(t)
        detections = tracks

    return detections
