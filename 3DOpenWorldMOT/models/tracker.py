from gettext import translation
import os
import sklearn.metrics
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from pyarrow import feather
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
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
    'num_interior_pts',
    'pts_density',
    'log_id'] + [f'{i}_{j}' for i in range(25) for j in ['x', 'y', 'z']]

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

column_dtypes_dets.update({f'{i}_{j}': 'float32' for i in range(25) for j in ['x', 'y', 'z']})

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
            rank=0,
            do_associate=None,
            precomp_tracks=False) -> None:
        
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
        self.precomp_tracks = precomp_tracks

        self.a_threshold = a_threshold
        self.i_threshold = i_threshold
        self.out_path = os.path.join(out_path)
        
        self.filtered_gt = '../../../data/argoverse2/val_0.833_per_frame_remove_non_move_remove_far_remove_non_drive_filtered_version.feather'
        self.rank = rank
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.tps = 0
        self.fns = 0
        self.fps = 0

    def new_log_id(self, log_id):
        # save tracks to feather and reset variables
        if self.log_id != -1:
            self.active_tracks = list()
            self.inactive_tracks = list()
            found = self.to_feather()
            if not found:
                logger.info(f'No detections found in {log_id}')

        self.log_id = log_id
        self.ordered_timestamps = torch.tensor(
            self.av2_loader.get_ordered_log_lidar_timestamps(log_id))
        logger.info(f"New log id {log_id}...")
    
    def get_detections(self, points, traj, clusters, timestamps, log_id,
                       gt_instance_ids, last=False):
        
        # account for padding in from DistributedTestSampler
        if timestamps.cpu()[0, 0] in self.detections.keys():
            if last:
                found = self.to_feather()
                if not found:
                    logger.info(f'No detections found in {log_id}')
            return

        # set new log id
        if self.log_id != log_id:
            self.new_log_id(log_id)

        if self.precomp_tracks and self.do_associate:
            found = self.to_feather()
            if not found:
                logger.info(f'No detections found in {log_id}')

        # iterate over clusters that were found and get detections with their 
        # corresponding flows, trajectories and canonical points
        detections = list()
        if type(clusters) == np.ndarray:
            clusters = torch.from_numpy(clusters).to(self.rank)

        for c in torch.unique(clusters):
            num_interior = torch.sum(clusters==c).item()
            gt_id = (torch.bincount(gt_instance_ids[clusters==c]).argmax()).item()

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

    def vis(self, det_cano_points, active_cano_points, inactive_cano_points, timestamp, log_id):
        fig, ax = plt.subplots()

        det_cano_points = copy.deepcopy(det_cano_points)
        active_cano_points = copy.deepcopy(active_cano_points)
        inactive_cano_points = copy.deepcopy(inactive_cano_points)

        transform = self.av2_loader.get_city_SE3_ego(log_id, timestamp)
        for det_cano_point in det_cano_points:
            det_cano_point = transform.transform_point_cloud(det_cano_point.cpu())
            lwh, translation = get_center_and_lwh(det_cano_point)
            lwh, translation = lwh.cpu(), translation.cpu()
            plt.scatter(translation[0], translation[1],
                            color='blue')
            x_0 = translation[0]-0.5*lwh[0]
            y_0 = translation[1]-0.5*lwh[1]
            rect = patches.Rectangle(
                (x_0, y_0),
                lwh[0],
                lwh[1],
                linewidth=1,
                edgecolor='blue',
                facecolor='none')
            ax.add_patch(rect)
        
        for active_cano_point in active_cano_points:
            active_cano_point = transform.transform_point_cloud(active_cano_point.cpu())
            lwh, translation = get_center_and_lwh(active_cano_point)
            lwh, translation = lwh.cpu(), translation.cpu()
            plt.scatter(translation[0], translation[1],
                            color='green', marker='*')
            x_0 = translation[0]-0.5*lwh[0]
            y_0 = translation[1]-0.5*lwh[1]
            rect = patches.Rectangle(
                (x_0, y_0),
                lwh[0],
                lwh[1],
                linewidth=1,
                edgecolor='green',
                facecolor='none')
            ax.add_patch(rect)
        
        for inactive_cano_point in inactive_cano_points:
            inactive_cano_point = transform.transform_point_cloud(inactive_cano_point.cpu())
            lwh, translation = get_center_and_lwh(inactive_cano_point)
            lwh, translation = lwh.cpu(), translation.cpu()
            plt.scatter(translation[0], translation[1],
                            color='red', marker='*')
            x_0 = translation[0]-0.5*lwh[0]
            y_0 = translation[1]-0.5*lwh[1]
            rect = patches.Rectangle(
                (x_0, y_0),
                lwh[0],
                lwh[1],
                linewidth=1,
                edgecolor='red',
                facecolor='none')
            ax.add_patch(rect)
        plt.ylim([2500, 2700])
        plt.xlim([-350, -250])
        os.makedirs(f'../../../vis_tracking/{log_id}', exist_ok=True)
        plt.savefig(f'../../../vis_tracking/{log_id}/{str(timestamp)}.jpg', dpi=1000)
        plt.close()

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
                t0 = torch.where(
                    self.ordered_timestamps==self.inactive_tracks[
                        idx].detections[-1].timestamps[0, 0])[0][0].item()
                t1 = torch.where(
                    self.ordered_timestamps==timestamp)[0][0].item()
                self.inactive_tracks[idx].inactive_count += t1-t0
            
            # reset active tracks
            self.active_tracks = list()
            return
        
        # calculate matching costs between detections and tracks
        cost_mat, num_act, num_inact, inactive_tracks_to_use = \
            self._calculate_traj_dist(detections, timestamp, matching=matching, alpha=alpha)

        gt_id_detections = [d.gt_id for d in detections]
        gt_id_tracks = [t.detections[-1].gt_id for t in self.active_tracks] + \
            [t.detections[-1].gt_id for t in self.inactive_tracks] 
        gt_matches = sum([1 for gt_id_d in gt_id_detections if gt_id_d in gt_id_tracks])

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
            re_activate, matched_tracks, matched_dets, tps, fps = self.greedy(
                cost_mat,
                num_act,
                detections,
                inactive_tracks_to_use,
                active_first=True,
                timestamp=timestamp,
                gt_id_tracks=gt_id_tracks,
                gt_id_detections=gt_id_detections)
            
            fns = gt_matches - tps
            self.tps += tps
            self.fps += fps
            self.fns += fns

        elif matching == 'majority':
            re_activate, matched_tracks, matched_dets = \
                self.majority_vote(cost_mat, num_act, detections, inactive_tracks_to_use)
        
        # reactivated tracks
        reactivated_tracks = list()
        for r, t in enumerate(self.inactive_tracks):
            if r in re_activate:
                t.inactive_count = 0
                reactivated_tracks.append(t)

        # still inactive
        self.inactive_tracks = [t for r, t in enumerate(self.inactive_tracks)\
            if r not in re_activate]

        # move tracks with no matching to inactive track
        deactivate = list()
        for idx in range(num_act):
            if idx not in matched_tracks:
                deactivate.append(idx)
                self.inactive_tracks.append(self.active_tracks[idx])
        
        # still active
        self.active_tracks = [t for r, t in enumerate(self.active_tracks)\
            if r not in deactivate]

        # add reactivated
        self.active_tracks = self.active_tracks + reactivated_tracks

        # increase inactive count
        for idx in range(len(self.inactive_tracks)):
            # not always 1 cos there can me timestamps wo moving objects
            t0 = torch.where(
                self.ordered_timestamps==self.inactive_tracks[
                    idx].detections[-1].timestamps[0, 0])[0][0].item()
            t1 = torch.where(
                self.ordered_timestamps==timestamp)[0][0].item()
            self.inactive_tracks[idx].inactive_count += t1-t0
        
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
            check = defaultdict(list)
            for track in self.active_tracks:
                for d in track.detections:
                    check[d.timestamps[0, 0].item()].append(d)
            lens = list()
            for k in sorted(check.keys()):
                lens.append(len(check[k]))
            logger.info(f'TPA: {self.tps}, FNA: {self.fns}, FPS: {self.fps}')
            self.tps = 0
            self.fns = 0
            self.fps = 0
            store_initial_detections(self.active_tracks, tracks=True)
            self.active_tracks = load_initial_detections(tracks=True, every_x_frame=self.every_x_frame, overlap=self.overlap)

            return self.register()
        return None
    
    def greedy(
            self,
            cost_mat,
            num_act,
            detections,
            inactive_tracks_to_use,
            active_first=False,
            timestamp=None,
            gt_id_tracks=None,
            gt_id_detections=None):
        re_activate = list()
        matched_tracks = list()
        matched_dets = list()
        fps = 0
        tps = 0

        # greedy matching: first match the one with lowest cost
        if active_first:
            act = torch.argsort(torch.min(cost_mat[:num_act], dim=1).values)
            inact = torch.argsort(torch.min(cost_mat[num_act:], dim=1).values) + num_act
            min_idx = torch.cat([act, inact])
        else:
            min_idx = torch.argsort(torch.min(cost_mat, dim=1).values)

        for idx in min_idx:
            # check if active or inactive track and get threshold
            act = idx < num_act
            thresh = self.a_threshold if act else self.i_threshold
            det_traj = torch.argmin(cost_mat[idx])
            # print(thresh, idx, det_traj, cost_mat[idx, det_traj])
            # match only of cost smaller than threshold

            if cost_mat[idx, det_traj] < thresh:
                matched_dets.append(det_traj.item())
                # if matched to inactive track, reactivate
                if act: 
                    self.active_tracks[idx].add_detection(detections[det_traj])
                    matched_tracks.append(idx.item())
                else:
                    inactive_idx = inactive_tracks_to_use[idx-num_act]
                    self.inactive_tracks[inactive_idx].add_detection(detections[det_traj])
                    re_activate.append(inactive_idx)
                tps = tps + 1 if gt_id_tracks[idx] == gt_id_detections[det_traj] else tps
                fps = fps + 1 if gt_id_tracks[idx] != gt_id_detections[det_traj] else fps
                # set costs of all other tracks to det to 1000 to not get matches again
                cost_mat[:, det_traj] = 10000

        return re_activate, matched_tracks, matched_dets, tps, fps

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
        # print('Active')
        trajs = [t._get_traj_and_convert_time(
            timestamp, self.av2_loader, overlap=True) for t in self.active_tracks]
        cano_points = [t._get_canonical_points_and_convert_time(
            timestamp, self.av2_loader, overlap=True).float().to(self.rank) for t in self.active_tracks]
        # print('Inactive')

        # get detections and canonical points of inactive track if 
        # overlap still predicted in previous added detection
        inactive_tracks_to_use = list()
        if len(self.inactive_tracks):
            for i, t in enumerate(self.inactive_tracks):
                if not t.dead:
                    # not always 1 cos there can me timestamps wo moving objects
                    t0 = torch.where(
                        self.ordered_timestamps==t.detections[
                            -1].timestamps[0, 0])[0][0].item()
                    t1 = torch.where(
                        self.ordered_timestamps==timestamp)[0][0].item()
                    time_dist = t1 - t0
                    # < cos for example if len_traj = 2, inactive_count=1, overlap=1
                    # then <= will be true but should not cos index starts at 0 not 1
                    if (t.inactive_count + time_dist) * self.every_x_frame + self.overlap \
                            < t.detections[-1].length:
                        trajs.extend([
                            t._get_traj_and_convert_time(timestamp, self.av2_loader, overlap=True)])
                        cano_points.extend([
                            t._get_canonical_points_and_convert_time(timestamp, self.av2_loader, overlap=True).float().to(self.rank)])
                        inactive_tracks_to_use.append(i)
                    else:
                        t.dead = True
        # print()
        # trajectories from canonical point of last added detections
        trajs_from_cano = [t[1].float().to(self.rank) for t in trajs]
        # tracejtories from current frame as canonical frame
        trajs_from_overlap = [t[0].float().to(self.rank) for t in trajs]
        
        # trajectories and canonical points of new detections
        det_trajs = [t._get_traj().float().to(self.rank) for t in detections]
        det_cano_points = [t._get_canonical_points().float().to(self.rank) for t in detections]

        # initialize position distances and trajectory distances
        num_time = trajs_from_overlap[0].shape[1]
        if matching != 'majority':
            # dists = torch.zeros((len(trajs_from_overlap), len(det_trajs)))
            dists_p = torch.zeros((len(trajs_from_overlap), len(det_trajs)))
        else:
            # dists = torch.zeros((len(trajs_from_overlap), len(det_trajs), num_time))
            dists_p = torch.zeros((len(trajs_from_overlap), len(det_trajs), num_time))

        self.vis(
            det_cano_points, cano_points[:len(self.active_tracks)], cano_points[len(self.active_tracks):], detections[0].timestamps[0,0].item(), detections[0].log_id)

        # get trajectory and position distances
        # iterate over trajectories
        for i, (track_traj, track_traj_c, track_p) in enumerate(zip(trajs_from_overlap, trajs_from_cano, cano_points)):

            # iterate over detections
            for j, (det_traj, det_p) in enumerate(zip(det_trajs, det_cano_points)):
                # if timestamp == 1507677181475568 or timestamp == 1507677182775411 or timestamp == 1507677182975295:
                #     print(i, j)
                # initialize distances for overlap time
                # dists_time = torch.zeros(num_time)
                dists_time_p = torch.zeros(num_time)

                # iterate over time, compute minimum distance from each point in track point
                # cloud to points in detection point cloud and get mean over track points
                for time in range(num_time):
                    # d = det_traj[:, time, :].unsqueeze(0)
                    # t = track_traj[:, time, :].unsqueeze(0)
                    # dists_time[time] = pytorch3d.loss.chamfer_distance(t, d)[0].cpu()
                    # dist = sklearn.metrics.pairwise_distances(t, d)
                    # dists_time[time] = dist.min(axis=1).mean()

                    # from get positions using previous canonical point at time
                    # _traj_p = track_p + track_traj_c[:, time, :].unsqueeze(0)
                    _traj_p = track_traj_c[:, time, :].unsqueeze(0)

                    # from get positions using current canonical point at time
                    _det_p = det_p + det_traj[:, time, :].unsqueeze(0)
                    # if timestamp == 1507677190273584 or timestamp == 1507677190273584 or timestamp == 1507677190273584:
                    #     print(_traj_p.mean(axis=1)[:, :-1], _det_p.mean(axis=1)[:, :-1])
                    # dists_time_p[time] = pytorch3d.loss.chamfer_distance(_traj_p, _det_p)[0].cpu()
                    dists_time_p[time] = self.pdist(_traj_p.mean(axis=1)[:, :-1].unsqueeze(0), _det_p.mean(axis=1)[:, :-1].unsqueeze(0))[0].cpu()
                # if timestamp == 1507677190273584 or timestamp == 1507677190273584 or timestamp == 1507677190273584:
                #     print(dists_time_p)
                # depending on matching strategy get mean over time or not
                if matching != 'majority':
                    # dists[i, j] = dists_time.mean()
                    dists_p[i, j] = dists_time_p.mean()
                else:
                    # dists[i, j, :] = dists_time
                    dists_p[i, j, :] = dists_time_p
        # if timestamp == 1507677190273584 or timestamp == 1507677190273584 or timestamp == 1507677190273584:
        #     print(timestamp, dists_p)

        # dists = dists * alpha + dists_p * (1-alpha)
        dists = dists_p

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
            if not self.precomp_tracks:
                store_initial_detections(self.detections)
                self.detections = load_initial_detections()
                # per timestampdetections
                for i, timestamp in enumerate(sorted(self.detections.keys())):
                    dets = self.detections[timestamp]
                    try:
                        time = timestamp.item()
                    except:
                        time = timestamp
                    detections = self.associate(
                        dets,
                        matching='greedy',
                        timestamp=time,
                        alpha=0.0,
                        make_cost_mat_dist=False,
                        last=len(self.detections)==i+1)
            else:
                self.active_tracks = load_initial_detections(
                    tracks=True, every_x_frame=self.every_x_frame, overlap=self.overlap)
                detections = self.register()

            check = defaultdict(list)
            for d in detections:
                check[d.timestamps[0, 0].item()].append(d)
            lens = list()
            for k in sorted(check.keys()):
                lens.append(len(check[k]))

            for det in detections:
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
                    det.num_interior,
                    det.pts_density,
                    det.log_id] + det.trajectory.numpy().tolist()
                track_vals.append(values)
        
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
                        det.num_interior,
                        det.pts_density,
                        det.log_id] + det.trajectory.numpy().tolist()
                    track_vals.append(values)
                    
        track_vals = np.asarray(track_vals)

        if track_vals.shape[0] == 0:
            return False

        df = pd.DataFrame(
            data=track_vals,
            columns=column_names_dets)
        df = df.astype(column_dtypes_dets)
        self.detections = dict()

        os.makedirs(os.path.join(self.out_path, self.split, self.log_id), exist_ok=True)
        os.makedirs(os.path.join(self.out_path, self.split, 'feathers'), exist_ok=True)
        write_path = os.path.join(self.out_path, self.split, 'feathers', f'all_{self.rank}.feather') 
        # write_path = os.path.join(self.out_path, self.split, self.log_id, 'annotations.feather')

        if os.path.isfile(write_path):
            # with open(write_path, 'rb') as f:
            df_all = feather.read_feather(write_path)
            df_all = df_all.append(df)
        else:
            df_all = df

        # with open(write_path, 'wb') as f:
        feather.write_feather(df_all, write_path)
        logger.info(f'wrote {write_path}')
        return True

    def register(self, max_interior_thresh=50):
        detections = list()
        init_R = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float().cuda()
        init_s = torch.tensor([1]).float().cuda()
        for j, track in enumerate(self.active_tracks):
            track_dets = list()
            # we start from timestep with most points and then go
            # from max -> end -> start -> max
            max_interior_idx = torch.argmax(torch.tensor([d.num_interior for d in track.detections])).item()
            max_interior = torch.max(torch.tensor([d.num_interior for d in track.detections])).item()
            start_in_t0 = torch.atleast_2d(track._get_canonical_points(i=max_interior_idx))
            if len(track) > 1 and max_interior > max_interior_thresh:
                max_to_end = range(max_interior_idx, len(track)-1)
                end_to_start = range(len(track)-1, 0, -1)
                start_to_max = range(0, max_interior_idx)
                iterators = [max_to_end, end_to_start, start_to_max]
                increment = [+1, -1, +1]

                SimilarityTransforms = [None] * len(track)
                for iterator, increment in zip(iterators, increment):
                    for i in iterator:
                        # number of interior points at next time stamp
                        num_interior = track.detections[i+increment].num_interior

                        # convert pc from ego frame t=i to ego frame t=i+increment
                        t0 = track.detections[i].timestamps[0, 0].item()
                        t1 = track.detections[i+increment].timestamps[0, 0].item()
                        start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)
                        # start_in_t1 = start_in_t0
                        trans0_in_t1 = (start_in_t1.min(axis=0).values + start_in_t1.max(axis=0).values)/2
                        
                        # get canconcial points at t
                        cano0_in_t0 = torch.atleast_2d(track._get_canonical_points(i=i))
                        cano0_in_t1 = track._convert_time(t0, t1, self.av2_loader, cano0_in_t0)
                        # trans0_in_t1 = (cano0_in_t1.min(axis=0).values + cano0_in_t1.max(axis=0).values)/2

                        # get canonical poitns at t+increment
                        cano1_in_t1 = torch.atleast_2d(track._get_canonical_points(i=i+increment))
                        trans1_in_t1 = (cano1_in_t1.min(axis=0).values + cano1_in_t1.max(axis=0).values)/2
                        
                        # init translation by distance between center points, R unit, s=1
                        init_T = (
                            trans1_in_t1 - trans0_in_t1).float().cuda().unsqueeze(0)
                        init = pytorch3d.ops.points_alignment.SimilarityTransform(
                            R=init_R, T=init_T, s=init_s)

                        if num_interior > max_interior_thresh*0.5 and num_interior > 40:
                            # ICP
                            ICPSolution = pytorch3d.ops.points_alignment.iterative_closest_point(
                                start_in_t1.float().cuda().unsqueeze(0),
                                cano1_in_t1.float().cuda().unsqueeze(0),
                                init_transform=init)

                            ### force rotation around z-axis
                            # ICPSolution.RTs.R[:, 0, 2] = 0
                            # ICPSolution.RTs.R[:, 1, 2] = 0
                            # ICPSolution.RTs.R[:, 2, 2] = 1
                            # ICPSolution.RTs.R[:, 2, 0] = 0
                            # ICPSolution.RTs.R[:, 2, 1] = 0
                            # start_registered = pytorch3d.ops.points_alignment._apply_similarity_transform(
                            #     start_in_t1.cuda().float().unsqueeze(0),
                            #     ICPSolution.RTs.R,
                            #     ICPSolution.RTs.T,
                            #     ICPSolution.RTs.s).squeeze().cpu()

                            ### use registered solution, could be some rotation around z-axis
                            start_registered = torch.atleast_2d(ICPSolution.Xt.cpu().squeeze())

                            # store for later
                            if SimilarityTransforms[i+increment] is None:
                                SimilarityTransforms[i+increment] = ICPSolution.RTs
                        else:
                            '''# if point cloud small just use distance as transform and assume no rotation
                            start_registered = pytorch3d.ops.points_alignment._apply_similarity_transform(
                                    start_in_t1.cuda().float().unsqueeze(0),
                                    init.R,
                                    init.T,
                                    init.s).squeeze()
                            start_registered = torch.atleast_2d(start_in_t0.cpu().squeeze())
                            if SimilarityTransforms[i+increment] is None:
                                SimilarityTransforms[i+increment] = init'''
                            pass
                        
                        # concatenate cano points at t+increment and registered points as
                        # point cloud for next timestamp / final point cloud
                        start_in_t0 = start_registered # torch.cat([start_registered, cano1_in_t1])

                max_interior_pc = start_in_t0
                mins, maxs = max_interior_pc.min(axis=0).values, max_interior_pc.max(axis=0).values
                lwh = maxs - mins
                translation = (maxs + mins)/2
                num_interior = max_interior_pc.shape[0]
                
                # setting last detection
                track.detections[max_interior_idx].lwh = lwh
                track.detections[max_interior_idx].translation = translation
                track.detections[max_interior_idx].num_interior = num_interior
                track_dets.append(track.detections[max_interior_idx])

                # Iterating from max -> end and max -> start
                max_to_end = range(max_interior_idx, len(track)-1)
                max_to_start = range(max_interior_idx, 0, -1)
                iterators = [max_to_end, max_to_start]
                increment = [+1, -1]
                for iterator, increment in zip(iterators, increment):
                    start_in_t0 = max_interior_pc
                    for i in iterator:
                        cano1_in_t1 = torch.atleast_2d(track._get_canonical_points(i=i+increment))
                        
                        # no registering from pc 
                        # from t0 (i+1) --> t1 (i)
                        t0 = track.detections[i].timestamps[0, 0].item()
                        t1 = track.detections[i+increment].timestamps[0, 0].item()
                        start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0.cpu())

                        if SimilarityTransforms[i+increment] is not None:
                            # apply stored similarity transform
                            start_in_t1_trans = pytorch3d.ops.points_alignment._apply_similarity_transform(
                                start_in_t1.cuda().float().unsqueeze(0),
                                SimilarityTransforms[i+increment].R,
                                SimilarityTransforms[i+increment].T,
                                SimilarityTransforms[i+increment].s).squeeze()
                            start_in_t1_trans = torch.atleast_2d(start_in_t1_trans)
                            # plt.plot(start_in_t1[:, 0].cpu(), start_in_t1[:, 1].cpu(), c='green')
                            # plt.plot(start_in_t1_trans[:, 0].cpu(), start_in_t1_trans[:, 1].cpu(), c='blue')
                            # plt.plot(cano1_in_t1[:, 0], cano1_in_t1[:, 1], c='red')
                            # plt.savefig(f'{j}_{max_interior_idx}_{i+increment}.jpg')
                            # plt.close()
                            os.makedirs("../../../registration", exist_ok=True)
                            np.savez(
                                os.path.join(f"../../../registration/{j}_{max_interior_idx}_{i+increment}.npz"),
                                cano1_in_t1=cano1_in_t1.cpu().numpy(),
                                start_in_t1_trans=start_in_t1_trans.cpu().numpy(),
                                start_in_t1=start_in_t1.cpu().numpy(), #[p.cpu().numpy() for p in pc_orig],
                            )
                        else:
                            start_in_t1_trans = torch.atleast_2d(track._get_canonical_points(i=i+increment))

                        # get bounding box from registered
                        mins_prev, maxs_prev = cano1_in_t1.min(axis=0).values, cano1_in_t1.max(axis=0).values
                        mins, maxs = start_in_t1_trans.min(axis=0).values, start_in_t1_trans.max(axis=0).values
                        mins[2] = mins_prev[2]
                        maxs[2] = maxs_prev[2]

                        track.detections[i+increment].translation = (maxs + mins)/2
                        track.detections[i+increment].lwh = maxs - mins
                        track.detections[i+increment].num_interior = num_interior
                        track_dets.append(track.detections[i+increment])
                        start_in_t0 = start_in_t1_trans
            else:
                for i in range(len(track.detections)):
                    points = track._get_canonical_points(i=i)
                    mins, maxs = points.min(axis=0).values, points.max(axis=0).values
                    lwh = maxs - mins
                    translation = (maxs + mins)/2
                    
                    # setting last detection
                    track.detections[i].lwh = lwh
                    track.detections[i].translation = translation
                    track_dets.append(track.detections[i])

            detections.extend(track_dets)
        return detections


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
        self.lwh = None
        self.translation = None
        self.lwh, self.translation = get_center_and_lwh(
            self.canonical_points, self.lwh, self.translation)
    
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
        mean_flow = (self.trajectory[:, 1, :] - self.trajectory[:, 0, :]).mean(dim=0)
        alpha = torch.arctan(mean_flow[1]/mean_flow[0])
        rot = torch.tensor([
            [torch.cos(alpha), -torch.sin(alpha), 0],
            [torch.sin(alpha), torch.cos(alpha), 0],
            [0, 0, 1]]).double()
        
        # rotate bounding box to get lwh in object coordinate system
        pc = torch.stack([
            self.translation + torch.tensor([0.5, 0, 0]) * self.lwh,
            self.translation + torch.tensor([-0.5, 0, 0]) * self.lwh,
            self.translation + torch.tensor([0, 0.5, 0]) * self.lwh,
            self.translation + torch.tensor([0, -0.5, 0]) * self.lwh,
            self.translation + torch.tensor([0, 0, 0.5]) * self.lwh,
            self.translation + torch.tensor([0, 0, -0.5]) * self.lwh]).double()
        print(pc, rot, self.translation)
        pc = pc @ rot.T + (-self.translation.double() @ rot.T)
        self.lwh, _ = get_center_and_lwh(pc)

        pts_density = (self.lwh[0] * self.lwh[1] * self.lwh[2]) / self.num_interior
        trajectory = torch.mean(self.trajectory, dim=0)
        self.final = Detection(rot, alpha, self.translation, self.lwh, self.timestamps[0, 0], self.log_id, self.num_interior, pts_density=pts_density, trajectory=trajectory)

        return self.final


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

def store_initial_detections(detections, tracks=False):
    if not tracks:
        p = '/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/initial_dets'
    else:
        p = '/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/initial_tracks'
    
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

def load_initial_detections(tracks=False, every_x_frame=1, overlap=1):
    if not tracks:
        p = '/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/initial_dets'
    else:
        p = '/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/initial_tracks'
    
    detections = defaultdict(list)
    for d in os.listdir(p):
        dict_key = int(d.split('_')[0])
        d = np.load(os.path.join(p, d))
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