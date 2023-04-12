from gettext import translation
import os
import sklearn.metrics
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from pyarrow import feather
import pandas as pd
import logging
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
import pytorch3d.loss
import torch
from collections import defaultdict
import copy


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

column_names_dets = [
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
    'num_interior_pts']

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

column_dtypes_dets = {
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

logger = logging.getLogger("Model.Tracker")


class Tracker3D():
    def __init__(
            self,
            out_path='out', 
            a_threshold=0.8, 
            i_threshold=0.8, 
            split='val', 
            every_x_frame=1, 
            num_interior=10, 
            overlap=5, 
            av2_loader=None, 
            do_associate=None) -> None:
        
        self.active_tracks = list()
        self.inactive_tracks = list()
        self.detections = dict()
        self.track_id = 0
        self.log_id = -1
        self.split = split
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.num_interior = num_interior
        self.av2_loader = av2_loader
        self.do_associate = do_associate

        self.a_threshold = a_threshold
        self.i_threshold = i_threshold
        self.out_path = os.path.join(out_path)
        
        self.filtered_gt = '../../../data/argoverse2/val_0.833_per_frame_remove_non_move_remove_far_remove_non_drive_filtered_version.feather'
    
    def new_log_id(self, log_id):
        # save tracks to feather and reset variables
        if self.log_id != -1:
            self.active_tracks = list()
            self.inactive_tracks = list()
            found = self.to_feather()
            if not found:
                logger.info(f'No detections found in {log_id}')

        self.log_id = log_id
        logger.info(f"New log id {log_id}...")
    
    def get_detections(self, points, traj, clusters, timestamps, log_id,
                       gt_instance_ids, last=False):
        # set new log id
        if self.log_id != log_id:
            self.new_log_id(log_id)

        # iterate over clusters that were found and get detections with their 
        # corresponding flows, trajectories and canonical points
        detections = list()

        if type(clusters) == np.ndarray:
                clusters = torch.from_numpy(clusters).cuda()

        for c in torch.unique(clusters):
            num_interior = torch.sum(clusters==c).item()
            gt_id = None #(np.bincount(gt_instance_ids[clusters==c]).argmax())

            # filter if cluster too small
            if num_interior < self.num_interior:
                continue
            # filter if 'junk' cluster
            if c == -1:
                continue
            # get points, bounding boxes
            point_cluster = points[clusters==c]

            # generate new detected trajectory
            traj_cluster = traj[clusters==c]
            detections.append(InitialDetection(
                traj_cluster.cpu(),
                point_cluster.cpu(),
                log_id=log_id,
                timestamps=timestamps.cpu(),
                num_interior=num_interior,
                overlap=self.overlap,
                gt_id=gt_id))
        
        self.detections[timestamps.cpu()[0, 0]] = detections

        if last:
            found = self.to_feather()
            if not found:
                logger.info(f'No detections found in {log_id}')
            
        return detections

    def associate(
            self,
            detections,
            matching='greedy',
            timestamp=None,
            alpha=0.0,
            make_cost_mat_dist=False,
            last=False):
        
        # add tracks if no tracks yet, initialize tracks
        if not len(self.active_tracks):
            self.active_tracks = list()
            for d in detections:
                self.active_tracks.append(Track(d, self.track_id, self.every_x_frame, self.overlap))
                self.track_id += 1
            return
        
        if not len(detections):
            # move all tracks to inactive track
            for tr in self.active_tracks:
                self.inactive_tracks.append(tr)

            # increase inactive count
            for idx in range(len(self.inactive_tracks)):
                self.inactive_tracks[idx].inactive_count += 1
            
            # reset active tracks
            self.active_tracks = list()
            return

        # calculate matching costs between detections and tracks
        cost_mat, num_act, num_inact, inactive_tracks_to_use = \
            self._calculate_traj_dist(detections, timestamp, matching=matching, alpha=alpha)

        if make_cost_mat_dist:
            if matching != 'majority':
                cost_mat = torch.nn.functional.softmax(-cost_mat, dim=1)
            else:
                tr, de, ti = cost_mat.shape
                cost_mat = cost_mat.view(tr, -1)
                cost_mat = torch.nn.functional.softmax(-cost_mat, dim=1)
                cost_mat = cost_mat.view(tr, de, ti)
            cost_mat = 1 - cost_mat
        
        # match detections and tracks
        if matching == 'greedy':
            re_activate, matched_tracks, matched_dets = \
                self.greedy(cost_mat, num_act, detections, inactive_tracks_to_use)

        elif matching == 'majority':
            re_activate, matched_tracks, matched_dets = \
                self.majority_vote(cost_mat, num_act, detections, inactive_tracks_to_use)

        self.inactive_tracks = [t for r, t in enumerate(self.inactive_tracks)\
            if r not in re_activate]

        # move tracks with no matching to inactive track
        deactivate = list()
        for idx in range(num_act):
            if idx not in matched_tracks:
                deactivate.append(idx)
                self.inactive_tracks.append(self.active_tracks[idx])

        self.active_tracks = [t for r, t in enumerate(self.active_tracks)\
            if r not in deactivate]

        # increase inactive count
        for idx in range(len(self.inactive_tracks)):
            self.inactive_tracks[idx].inactive_count += 1
        
        # start new tracks
        for idx in range(len(detections)):
            if idx not in matched_dets:
                self.active_tracks.append(Track(
                    detections[idx],
                    self.track_id,
                    overlap=self.overlap,
                    every_x_frame=self.every_x_frame))
                self.track_id += 1
        if last:
            self.active_tracks += self.inactive_tracks
            self.register()
    
    def greedy(self, cost_mat, num_act, detections, inactive_tracks_to_use):
        re_activate = list()
        matched_tracks = list()
        matched_dets = list()
        # greedy matching: first match the one with lowest cost
        min_idx = torch.argsort(torch.min(cost_mat, dim=1))

        for idx in min_idx:
            # check if active or inactive track and get threshold
            act = idx < num_act
            thresh = self.a_threshold if act else self.i_threshold
            det_traj = torch.argmin(cost_mat[idx])
            # match only of cost smaller than threshold
            if cost_mat[idx, det_traj] < thresh:
                matched_dets.append(det_traj)
                # if matched to inactive track, reactivate
                if act: 
                    self.active_tracks[idx].add_detection(detections[det_traj])
                    matched_tracks.append(idx)
                else:
                    inactive_idx = inactive_tracks_to_use[idx-num_act]
                    self.inactive_tracks[inactive_idx].add_detection(detections[det_traj])
                    self.inactive_tracks[inactive_idx].inactive_count = 0
                    self.active_tracks.append(self.inactive_tracks[inactive_idx])
                    re_activate.append(inactive_idx)
                # set costs of all other tracks to det to 1000 to not get matches again
                cost_mat[:, det_traj] = 10000
        
        return re_activate, matched_tracks, matched_dets

    def majority_vote(self, cost_mat, num_act, detections, inactive_tracks_to_use):
        re_activate = list()
        matched_tracks = list()
        matched_dets = list()
        # majority voting over tracks within overlap
        # cost_mat shape: trajs, dats, time
        # get closest vote (trajs, time) for all trajectories over overlap
        votes = torch.zeros((cost_mat.shape[0], cost_mat.shape[2]))
        for t in range(cost_mat.shape[2]):
            cost = cost_mat[:, :, t]
            vote = torch.argsort(cost, dim=1)
            votes[:, t] = vote[:, 0]

        # get majority counts
        count = list()
        max_vote = list()
        for track_votes in votes.numpy():
            c = np.bincount(track_votes.astype(np.int64))
            count.append(c)
            max_vote.append(np.max(c))

        # assign the one with the 
        for idx in torch.argsort(torch.tensor(max_vote))[::-1]:
            if torch.max(torch.tensor(count[idx])) == 0:
                continue
            det_traj = torch.argmax(torch.tensor(count[idx]))
            act = idx < num_act
            thresh = self.a_threshold if act else self.i_threshold
            if cost_mat[idx, det_traj, :].mean() < thresh:
                matched_dets.append(det_traj)
                if act: 
                    self.active_tracks[idx].add_detection(detections[det_traj])
                    matched_tracks.append(idx)
                else:
                    inactive_idx = inactive_tracks_to_use[idx-num_act]
                    self.inactive_tracks[inactive_idx].add_detection(detections[det_traj])
                    self.inactive_tracks[inactive_idx].inactive_count = 0
                    self.active_tracks.append(self.inactive_tracks[inactive_idx])
                    re_activate.append(inactive_idx)

                for i, c in enumerate(count):
                    if i == idx:
                        continue
                    if c.shape[0] - 1 >= det_traj:
                        c[det_traj] = 0
        
        return re_activate, matched_tracks, matched_dets

    def _calculate_traj_dist(self, detections, timestamp, alpha=0.0, \
                             matching='greedy'):
        # get detections and canonical points of last x frames 
        # and convert to current time for all active tracks
        trajs = [t._get_traj_and_convert_time(
            timestamp, self.av2_loader, overlap=True) for t in self.active_tracks]
        cano_points = [t._get_canonical_points_and_convert_time(
            timestamp, self.av2_loader, overlap=True) for t in self.active_tracks]

        # get detections and canonical points of inactive track if 
        # overlap still predicted in previous added detection
        inactive_tracks_to_use = list()
        if len(self.inactive_tracks):
            for i, t in enumerate(self.inactive_tracks):
                if not t.dead:
                    # < cos for example if len_traj = 2, inactive_count=1, overlap=1
                    # then <= will be true but should not cos index starts at 0 not 1
                    if (t.inactive_count) * self.every_x_frame + self.overlap \
                            < t.detections[-1].length:
                        trajs.extend([
                            t._get_traj_and_convert_time(timestamp, self.av2_loader, overlap=True)])
                        cano_points.extend([
                            t._get_canonical_points_and_convert_time(timestamp, self.av2_loader, overlap=True)])
                        inactive_tracks_to_use.append(i)
                    else:
                        t.dead = True

        # trajectories from canonical point of last added detections
        trajs_from_cano = [t[1] for t in trajs]
        # tracejtories from current frame as canonical frame
        trajs_from_overlap = [t[0] for t in trajs]
        
        # trajectories and canonical points of new detections
        det_trajs = [t._get_traj() for t in detections]
        det_cano_points = [t._get_canonical_points() for t in detections]

        # initialize position distances and trajectory distances
        num_time = trajs_from_overlap[0].shape[1]
        if matching != 'majority':
            dists = torch.zeros((len(trajs_from_overlap), len(det_trajs)))
            dists_p = torch.zeros((len(trajs_from_overlap), len(det_trajs)))
        else:
            dists = torch.zeros((len(trajs_from_overlap), len(det_trajs), num_time))
            dists_p = torch.zeros((len(trajs_from_overlap), len(det_trajs), num_time))

        # get trajectory and position distances
        # iterate over trajectories
        for i, (track_traj, track_p) in enumerate(zip(trajs_from_overlap, cano_points)):

            # iterate over detections
            for j, (det_traj, det_p) in enumerate(zip(det_trajs, det_cano_points)):
                # initialize distances for overlap time
                dists_time = torch.zeros(num_time)
                dists_time_p = torch.zeros(num_time)

                # iterate over time, compute minimum distance from each point in track point
                # cloud to points in detection point cloud and get mean over track points
                for time in range(num_time):
                    d = det_traj[:, time, :].unsqueeze(0).float().cuda()
                    t = track_traj[:, time, :].unsqueeze(0).float().cuda()
                    dists_time[time] = pytorch3d.loss.chamfer_distance(t, d)[0].cpu()
                    # dist = sklearn.metrics.pairwise_distances(t, d)
                    # dists_time[time] = dist.min(axis=1).mean()

                    # from get positions using previous canonical point at time
                    _traj_p = track_p + trajs_from_cano[i][:, time, :].unsqueeze(0).float().cuda()

                    # from get positions using current canonical point at time
                    _det_p = det_p + det_trajs[j][:, time, :].unsqueeze(0).float().cuda()
                    dists_time_p[time] = pytorch3d.loss.chamfer_distance(_traj_p, _det_p)[0].cpu()
                    # dist = sklearn.metrics.pairwise_distances(_traj_p, _det_p)

                # depending on matching strategy get mean over time or not
                if matching != 'majority':
                    dists[i, j] = dists_time.mean()
                    dists_p[i, j] = dists_time_p.mean()
                else:
                    dists[i, j, :] = dists_time
                    dists_p[i, j, :] = dists_time_p

        dists = dists * alpha + dists_p * (1-alpha)

        return dists, \
            len(self.active_tracks), \
                len(trajs) - len(self.active_tracks), \
                    inactive_tracks_to_use

    def mat_to_quat(self, mat):
        """Convert a 3D rotation matrix to a scalar _first_ quaternion.
        NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
            we use the scalar FIRST convention.
        Args:
            mat: (...,3,3) 3D rotation matrices.
        Returns:
            (...,4) Array of scalar first quaternions.
        """
        # Convert quaternion from scalar first to scalar last.
        quat_xyzw = Rotation.from_matrix(mat).as_quat()
        quat_wxyz = quat_xyzw[..., [3, 0, 1, 2]]
        return quat_wxyz
    
    def quat_to_mat(self, quat_wxyz):
        """Convert a quaternion to a 3D rotation matrix.

        NOTE: SciPy uses the scalar last quaternion notation. Throughout this repository,
            we use the scalar FIRST convention.

        Args:
            quat_wxyz: (...,4) array of quaternions in scalar first order.

        Returns:
            (...,3,3) 3D rotation matrix.
        """
        # Convert quaternion from scalar first to scalar last.
        quat_xyzw = quat_wxyz[..., [1, 2, 3, 0]]
        mat = Rotation.from_quat(quat_xyzw).as_matrix()
        return mat

    def to_feather(self):
        track_vals = list()
        if self.do_associate:
            # per timestampdetections
            for i, timestamp in enumerate(sorted(self.detections.keys())):
                dets = self.detections[timestamp]
                self.associate(
                    dets,
                    matching='greedy',
                    timestamp=timestamp,
                    alpha=0.0,
                    make_cost_mat_dist=False,
                    last=len(self.detections)==i+1)
                                    
            track_vals = np.asarray(track_vals)
            if track_vals.shape[0] == 0:
                return False

            df = pd.DataFrame(
                data=track_vals,
                columns=column_names_dets)
            df = df.astype(column_dtypes_dets)
            self.active_tracks = list()
            self.inactive_tracks = list()
        
        else:
            # per timestamp detections
            for i, timestamp in enumerate(sorted(self.detections.keys())):
                dets = self.detections[timestamp]
                for det in dets:
                    det = det.final_detection()
                    # quaternion rotation around z axis
                    quat = torch.tensor([torch.cos(det.heading/2), 0, 0, torch.sin(det.heading/2)]).numpy()
                    # REGULAR_VEHICLE = only dummy class
                    values = [
                        det.translation[0].item(),
                        det.translation[1].item(),
                        det.translation[2].item(),
                        det.lwh[0].item(),
                        det.lwh[1].item(),
                        det.lwh[2].item(),
                        quat[0],
                        quat[1],
                        quat[2],
                        quat[3],
                        int(det.timestamp.item()),
                        'REGULAR_VEHICLE',
                        det.num_interior]
                    track_vals.append(values)
            track_vals = np.asarray(track_vals)
            if track_vals.shape[0] == 0:
                return False

            df = pd.DataFrame(
                data=track_vals,
                columns=column_names_dets)
            df = df.astype(column_dtypes_dets)
            self.detections = list()

        os.makedirs(os.path.join(self.out_path, self.split, self.log_id), exist_ok=True)
        write_path = os.path.join(self.out_path, self.split, self.log_id, 'annotations.feather')
        logger.info(f'Stored tracks for sequence {self.log_id} at {os.getcwd()}/{write_path}')
        feather.write_feather(df, write_path)

        return True

    def register(self):
        detections = list()
        init_R = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float().cuda()
        init_s = torch.tensor([1]).float().cuda()
        
        for track in self.active_tracks:
            # we start from timestep with most points and then go
            # from max -> end -> start -> max
            max_interior = torch.argmax(track.detections.num_interior)
            start_in_t0 = track._get_canonical_points(i=max_interior)
            max_to_end = range(max_interior, len(track)-1)
            end_to_start = range(len(track), 0, -1)
            start_to_max = range(0, max_interior)
            iterators = [max_to_end, end_to_start, start_to_max]
            increment = [+1, -1, +1]

            SimilarityTransforms = [None] * len(track)
            for iterator, increment in zip(iterators, increment):
                for i in iterator:
                    # convert pc from ego frame t=i to ego frame t=i+increment
                    t0 = track.detections[i].timestamps[0].item()
                    t1 = track.detections[i+increment].timestamps[0].item()
                    start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)
                    trans0_in_t1 = (start_in_t1.min(axis=0) + start_in_t1.max(axis=0))/2
                    
                    # get canonical poitns at t+increment
                    cano1_in_t1 = track._get_canonical_points(i=i+increment)
                    trans1_in_t1 = (cano1_in_t1.min(axis=0) + cano1_in_t1.max(axis=0))/2
                    
                    # init translation by distance between center points, R unit, s=1
                    init_T = (
                        trans1_in_t1 - trans0_in_t1).float().cuda().unsqueeze(0)
                    init = pytorch3d.ops.points_alignment.SimilarityTransform(
                        R=init_R, T=init_T, s=init_s)

                    # ICP
                    ICPSolution = pytorch3d.ops.points_alignment.iterative_closest_point(
                        start_in_t1.float().cuda().unsqueeze(0),
                        cano1_in_t1.float().cuda().unsqueeze(0),
                        init_transform=init)
                    start_registered = ICPSolution.Xt.cpu().squeeze().numpy()
                    if SimilarityTransforms[i+increment] is not None:
                        SimilarityTransforms.append(ICPSolution.SimilarityTransform)
                    
                    # concatenate cano points at t+increment and registered points as
                    # point cloud for next timestamp / final point cloud
                    start_in_t0 = torch.cat([start_registered, cano1_in_t1])

            max_interior_pc = start_in_t0
            mins, maxs = max_interior_pc.min(axis=0), max_interior_pc.max(axis=0)
            lwh = maxs - mins
            translation = (maxs + mins)/2
            
            # setting last detection
            track.detections[max_interior].lwh = lwh
            track.detections[max_interior].translation = translation
            detections.append(track.detections[-1])
            
            # Iterating from max -> end and max -> start
            max_to_end = range(max_interior, len(track)-1)
            max_to_start = range(max_interior, 0, -1)
            iterators = [max_to_end, max_to_start]
            increment = [+1, -1]
            for iterator, increment in zip(iterators, increment):
                start_in_t0 = max_interior_pc
                for i in iterator:
                    # from t0 (i+1) --> t1 (i)
                    cano1_in_t1 = track._get_canonical_points(i=i)
                    t0 = track.detections[i+increment].timestamps[0].item()
                    t1 = track.detections[i].timestamps[0].item()
                    start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)

                    # apply stored similarity transform
                    start_in_t0 = pytorch3d.ops.points_alignment._apply_similarity_transform(
                        start_in_t1,
                        SimilarityTransforms[i+increment].R,
                        SimilarityTransforms[i+increment].T,
                        SimilarityTransforms[i+increment].s)

                    # get bounding box from registered
                    mins, maxs = start_in_t0.min(axis=0), start_in_t0.max(axis=0)
                    track.detections[i+increment].translation = (maxs + mins)/2
                    track.detections[i+increment].lwh = maxs - mins
                    
                    detections.append(track.detections[i+increment])


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

            city_SE3_ego0 = av2_loader.get_city_SE3_ego(self.log_id, det.timestamps[0].item())

            for time in range(_range):
                points_c_time = det.canonical_points + det.trajectory[:, time, :]

                city_SE3_ego = av2_loader.get_city_SE3_ego(self.log_id, det.timestamps[time].item())
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
                self.final.append(Detection(rot, translation, lwh, det.timestamps[time], det.log_id, det.num_interior))
    
    def _get_traj(self, i=-1):
        return self.detections[i].trajectory
    
    def _get_traj_and_convert_time(self, t1, av2_loader, i=-1, overlap=False):
        traj = copy.deepcopy(self._get_traj(i))
        if overlap:
            start = self.inactive_count * (self.every_x_frame)
            end = start + self.overlap

            assert end < self.detections[i].length, 'inactive track too short'
            # +1 cos of indexing
            traj = traj[:, start+1:end+1]

        t0 = self.detections[i].timestamps[0].item()
        traj = self._convert_time(t0, t1, av2_loader, traj)
        if overlap:
            traj_from_overlap = traj - torch.tile(traj[:, 0, :].unsqueeze(1) , (1, traj.shape[1], 1))
            return traj_from_overlap, traj
        else:
            return traj
    
    def _get_canonical_points(self, i=-1):
        return self.detections[i].canonical_points
    
    def _get_canonical_points_and_convert_time(self, t1, av2_loader, i=-1, overlap=False):
        canonical_points = copy.deepcopy(self._get_canonical_points(i=i))
        if overlap:
            start = (1 + self.inactive_count) * (self.every_x_frame)
            canonical_points += copy.deepcopy(self.detections[i].trajectory[:, start])
        t0 = self.detections[i].timestamps[0].item()
        return self._convert_time(t0, t1, av2_loader, canonical_points)

    def _convert_time(self, t0, t1, av2_loader, points):
        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)
        city_SE3_t1 = av2_loader.get_city_SE3_ego(self.log_id, t1)
        t1_SE3_t0 = city_SE3_t1.inverse().compose(city_SE3_t0)

        return t1_SE3_t0.transform_point_cloud(points)

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
        self.lwh = None
        self.translation = None
    
    def _get_traj(self):
        traj = self.trajectory[:, 1:self.overlap+1]
        return traj
    
    def _get_canonical_points(self):
        canonical_points = self.canonical_points
        return canonical_points

    def final_detection(self):
        points_c_time = self.canonical_points
        mins, maxs = points_c_time.min(dim=0), points_c_time.max(dim=0)
        if self.lwh is None:
            lwh = maxs.values - mins.values
            translation = (maxs.values + mins.values)/2
        else: 
            lwh = self.lwh
            translation = self.translation

        mean_flow = (self.trajectory[:, 1, :] - self.trajectory[:, 0, :]).mean(dim=0)
        alpha = torch.arctan(mean_flow[1]/mean_flow[0])
        rot = torch.tensor([
            [torch.cos(alpha), torch.sin(alpha), 0],
            [torch.sin(alpha), torch.cos(alpha), 0],
            [0, 0, 1]])
        self.final = Detection(rot, alpha, translation, lwh, self.timestamps[0, 0], self.log_id, self.num_interior)

        return self.final


class Detection():
    def __init__(self, rotation, heading, translation, lwh, timestamp, log_id, num_interior) -> None:
        self.rotation = rotation
        self.heading = heading
        self.translation = translation
        self.lwh = lwh
        self.timestamp = timestamp
        self.log_id = log_id
        self.num_interior = num_interior

