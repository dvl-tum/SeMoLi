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
    def __init__(self, out_path='out', a_threshold=0.8, i_threshold=0.8, split='val', every_x_frame=1, num_interior=10, overlap=5, av2_loader=None) -> None:
        self.active_tracks = list()
        self.inactive_tracks = list()
        self.trajectories = list()
        self.track_id = 0
        self.log_id = -1
        self.split = split
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.num_interior = num_interior
        self.av2_loader = av2_loader

        self.a_threshold = a_threshold
        self.i_threshold = i_threshold
        self.out_path = os.path.join(out_path)
        
        self.filtered_gt = '../../../data/argoverse2/val_0.833_per_frame_remove_non_move_remove_far_remove_non_drive_filtered_version.feather'
    
    def new_log_id(self, log_id, only_dets=True):
        # save tracks to feather and reset variables
        if self.log_id != -1:
            self.active_tracks += self.inactive_tracks
            found = self.to_feather(only_dets=only_dets)
            if not found:
                logger.info(f'No detections found in {log_id}')
            self.active_tracks = list()
            self.inactive_tracks = list()
            self.trajectories = list()

        self.log_id = log_id
        logger.info(f"New log id {log_id}...")
    
    def get_detections(self, points, traj, clusters, timestamps, log_id, gt_instance_ids, last=False, only_dets=True):
        # set new log id
        if self.log_id != log_id:
            self.new_log_id(log_id, only_dets)

        points = points.numpy()
        traj = traj.numpy()

        # iterate over clusters that were found and get detections with their 
        # corresponding flows, trajectories and canonical points
        trajectories = list()
        for c in np.unique(clusters):
            num_interior = np.sum([clusters==c])

            gt_id = None #(np.bincount(gt_instance_ids[clusters==c]).argmax())

            # filter if cluster too small
            if num_interior < self.num_interior:
                continue
            # filter if 'junk' cluster
            if c == -1:
                continue
            # get points, bounding boxes
            point_cluster = points[clusters==c]
            mins, maxs = point_cluster.min(axis=0), point_cluster.max(axis=0)                                                                                                
            lwh = maxs - mins
            # remove bb if l, w, or h is 0
            if lwh[2] == 0 or lwh[0] == 0 or lwh[1] == 0:                                                                                                                                                  
                continue

            '''if lwh[0] > 50 or lwh[1] > 50 or lwh[2] > 4:
                continue
            
            if (lwh[0]/lwh[2] > 4 or lwh[1]/lwh[2] < 4) and lwh[2] < 0.5:                                                                                                    
                continue'''

            # generate new detected trajectory
            traj_cluster = traj[clusters==c]
            trajectories.append(Trajectory(
                traj_cluster,
                point_cluster,
                log_id=log_id,
                timestamps=timestamps,
                num_interior=num_interior,
                overlap=self.overlap,
                gt_id=gt_id))
            
        if only_dets:
            self.trajectories += trajectories

        if last and only_dets:
            self.to_feather(only_dets=only_dets)
            self.trajectories = list()
            
        return trajectories

    def associate(self, trajectories, last=False, matching='greedy', timestamp=None, alpha=0.95, make_cost_mat_dist=True):
        # add tracks if no tracks yet, initialize tracks
        if not len(self.active_tracks):
            self.active_tracks = list()
            for t in trajectories:
                self.active_tracks.append(Track(t, self.track_id, self.every_x_frame, self.overlap))
                self.track_id += 1
            return
        
        if not len(trajectories):
            # move tracks with no matching to inactive track
            for tr in self.active_tracks:
                if idx not in matched_tracks and len(tr) >= 2:
                    self.inactive_tracks.append(tr)

            # increase inactive count
            for idx in range(len(self.inactive_tracks)):
                self.inactive_tracks[idx].inactive_count += 1
            
            # reset active tracks
            self.active_tracks = list()
            return

        # calculate matching costs between detections and tracks
        cost_mat, num_act, num_inact, inactive_tracks_to_use = \
            self._calculate_traj_dist(trajectories, timestamp, matching=matching, alpha=alpha)

        if make_cost_mat_dist:
            if matching != 'majority':
                cost_mat = torch.nn.functional.softmax(-torch.from_numpy(cost_mat), dim=1)
            else:
                cost_mat = torch.from_numpy(cost_mat)
                tr, de, ti = cost_mat.shape
                cost_mat = cost_mat.view(tr, -1)
                cost_mat = torch.nn.functional.softmax(-cost_mat, dim=1)
                cost_mat = cost_mat.view(tr, de, ti)
            cost_mat = cost_mat.numpy()
            cost_mat = 1 - cost_mat
        
        # match detections and tracks
        re_activate = list()
        matched_tracks = list()
        matched_trajs = list()
        if matching == 'greedy':
            # greedy matching: first match the one with lowest cost
            min_idx = np.argsort(np.min(cost_mat, axis=1))
            for idx in min_idx:
                # check if active or inactive track and get threshold
                act = idx < num_act
                thresh = self.a_threshold if act else self.i_threshold
                det_traj = np.argmin(cost_mat[idx])
                # match only of cost smaller than threshold
                if cost_mat[idx, det_traj] < thresh:
                    matched_trajs.append(det_traj)
                    # if matched to inactive track, reactivate
                    if act: 
                        self.active_tracks[idx].add_trajectory(trajectories[det_traj])
                        matched_tracks.append(idx)
                    else:
                        inactive_idx = inactive_tracks_to_use[idx-num_act]
                        self.inactive_tracks[inactive_idx].add_trajectory(trajectories[det_traj])
                        self.inactive_tracks[inactive_idx].inactive_count = 0
                        self.active_tracks.append(self.inactive_tracks[inactive_idx])
                        re_activate.append(inactive_idx)
                    # set costs of all other tracks to det to 1000 to not get matches again
                    cost_mat[:, det_traj] = 10000

        elif matching == 'majority':
            # majority voting over tracks within overlap
            # cost_mat shape: trajs, dats, time
            # get closest vote (trajs, time) for all trajectories over overlap
            votes = np.zeros((cost_mat.shape[0], cost_mat.shape[2]))
            for t in range(cost_mat.shape[2]):
                cost = cost_mat[:, :, t]
                vote = np.argsort(cost, axis=1)
                votes[:, t] = vote[:, 0]

            # get majority counts
            count = list()
            max_vote = list()
            for track_votes in votes:
                c = np.bincount(track_votes.astype(np.int64))
                count.append(c)
                max_vote.append(np.max(c))

            # assign the one with the 
            for idx in np.argsort(np.asarray(max_vote))[::-1]:
                if np.max(np.asarray(count[idx])) == 0:
                    continue
                det_traj = np.argmax(np.asarray(count[idx]))
                act = idx < num_act
                thresh = self.a_threshold if act else self.i_threshold
                if cost_mat[idx, det_traj, :].mean() < thresh:
                    matched_trajs.append(det_traj)
                    if act: 
                        self.active_tracks[idx].add_trajectory(trajectories[det_traj])
                        matched_tracks.append(idx)
                    else:
                        inactive_idx = inactive_tracks_to_use[idx-num_act]
                        self.inactive_tracks[inactive_idx].add_trajectory(trajectories[det_traj])
                        self.inactive_tracks[inactive_idx].inactive_count = 0
                        self.active_tracks.append(self.inactive_tracks[inactive_idx])
                        re_activate.append(inactive_idx)

                    for i, c in enumerate(count):
                        if i == idx:
                            continue
                        if c.shape[0] - 1 >= det_traj:
                            c[det_traj] = 0
        else:
            re_activate = list()
            matched_tracks = list()
            matched_trajs = list()
            row, col = linear_sum_assignment(cost_mat)
            for idx, traj in zip(row, col):
                act = idx < num_act
                thresh = self.a_threshold if act else self.i_threshold
                if cost_mat[idx, traj] < thresh:
                    matched_trajs.append(traj)
                    if act: 
                        self.active_tracks[idx].add_trajectory(trajectories[traj])
                        matched_tracks.append(idx)
                    else:
                        inactive_idx = inactive_tracks_to_use[idx-num_act]
                        self.inactive_tracks[inactive_idx].add_trajectory(trajectories[traj])
                        self.inactive_tracks[inactive_idx].inactive_count = 0
                        self.active_tracks.append(self.inactive_tracks[inactive_idx])
                        re_activate.append(inactive_idx)

        self.inactive_tracks = [t for r, t in enumerate(self.inactive_tracks)\
            if r not in re_activate]

        # move tracks with no matching to inactive track
        deactivate = list()
        for idx in range(num_act):
            if idx not in matched_tracks and len(self.active_tracks[idx]) >= 2:
                deactivate.append(idx)
                self.inactive_tracks.append(self.active_tracks[idx])
            elif idx not in matched_tracks and len(self.active_tracks[idx]) < 2:
                deactivate.append(idx)
        self.active_tracks = [t for r, t in enumerate(self.active_tracks)\
            if r not in deactivate]

        # increase inactive count
        for idx in range(len(self.inactive_tracks)):
            self.inactive_tracks[idx].inactive_count += 1
        
        # start new tracks
        for idx in range(len(trajectories)):
            if idx not in matched_trajs:
                self.active_tracks.append(Track(
                    trajectories[idx],
                    self.track_id,
                    overlap=self.overlap,
                    every_x_frame=self.every_x_frame))
                self.track_id += 1
        if last:
            self.active_tracks += self.inactive_tracks
            self.to_feather()
            self.active_tracks = list()
            self.inactive_tracks = list()

    def _calculate_traj_dist(self, trajectories, timestamp, alpha=0.95, matching='greedy', visualize=False):
        # get detections and canonical points of last x frames 
        # and convert to current time for all active tracks
        trajs = [t._get_last_traj_and_convert_time(
            timestamp, self.av2_loader) for t in self.active_tracks]
        cano_points = [t._get_canonical_point_at_start_of_overlap_and_convert_time(
            timestamp, self.av2_loader) for t in self.active_tracks]

        # get detections and canonical points of inactive track if 
        # overlap still predicted in previous added detection
        inactive_tracks_to_use = list()
        if len(self.inactive_tracks):
            for i, t in enumerate(self.inactive_tracks):
                if (t.inactive_count + 1) * self.every_x_frame + self.overlap + 1 \
                            < t.trajectories[-1].length:
                    trajs.extend([t._get_last_traj_and_convert_time(timestamp, self.av2_loader)])
                    cano_points.extend([
                        t._get_canonical_point_at_start_of_overlap_and_convert_time(timestamp, self.av2_loader)])
                    inactive_tracks_to_use.append(i)

        # trajectories from canonical of last added detections
        trajs_from_cano = [t[1] for t in trajs]
        # tracejtories from current frame as canonical frame
        trajs = [t[0] for t in trajs]
        
        # trajectories and canonical points of new detections
        det_trajs = [t._get_traj() for t in trajectories]
        det_cano_points = [t._get_canonical_point() for t in trajectories]

        # visualize current step
        if visualize:
            self.visualize(trajs_from_cano, cano_points, det_trajs, det_cano_points, timestamp)

        # initialize position distances and trajectory distances
        num_time = trajs[0].shape[1]
        if matching != 'majority':
            dists = np.zeros((len(trajs), len(det_trajs)))
            dists_p = np.zeros((len(trajs), len(det_trajs)))
        else:
            dists = np.zeros((len(trajs), len(det_trajs), num_time))
            dists_p = np.zeros((len(trajs), len(det_trajs), num_time))

        # get trajectory and position distances
        # iterate over trajectories
        for i, (traj, traj_p) in enumerate(zip(trajs, cano_points)):

            # iterate over detections
            for j, (det, det_p) in enumerate(zip(det_trajs, det_cano_points)):
                # initialize distances for overlap time
                dists_time = np.zeros(num_time)
                dists_time_p = np.zeros(num_time)

                # iterate over time, compute minimum distance from each point in track point
                # cloud to points in detection point cloud and get mean over track points
                for time in range(num_time):
                    d = torch.from_numpy(det[:, time, :]).unsqueeze(0).float()
                    t = torch.from_numpy(traj[:, time, :]).unsqueeze(0).float()
                    dists_time[time] = pytorch3d.loss.chamfer_distance(t, d)[0]
                    # dist = sklearn.metrics.pairwise_distances(t, d)
                    # dists_time[time] = dist.min(axis=1).mean()

                    # from get positions using previous canonical point at time
                    _traj_p = torch.from_numpy(traj_p + trajs_from_cano[i][:, time, :]).unsqueeze(0).float()
                    # from get positions using current canonical point at time
                    _det_p = torch.from_numpy(det_p + det_trajs[j][:, time, :]).unsqueeze(0).float()
                    dists_time_p[time] = pytorch3d.loss.chamfer_distance(_traj_p, _det_p)[0]
                    # dist = sklearn.metrics.pairwise_distances(_traj_p, _det_p)
                    # dists_time_p[time] = dist.min(axis=1).mean()

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
    
    def visualize(self, traj0, cano0, traj1, cano1, timestamp):
        os.makedirs('../../../vis_cano', exist_ok=True)
        fig = plt.figure(figsize=(100, 100))
        for i, (t, p) in enumerate(zip(traj0, cano0)):
            plt.scatter(p[:, 0], p[:, 1], color='red', marker='v')

        for i, (t, p) in enumerate(zip(traj1, cano1)):
            plt.scatter(p[:, 0], p[:, 1], color='blue', marker='*')

        plt.savefig(f'../../../vis_cano/{timestamp}.jpg')
        plt.close()

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

    def to_feather(self, visualize=False, only_dets=False):
        track_vals = list()
        if not only_dets:
            for track in self.active_tracks:
                track.final_detections(self.av2_loader)
                for det in track:
                    # quaternion rotation around z axis
                    quat = np.array([np.cos(det.rotation/2), 0, 0, np.sin(det.rotation/2)])
                    # REGULAR_VEHICLE = only dummy class
                    values = [
                        int(det.timestamp.item()),
                        track.track_id,
                        'REGULAR_VEHICLE',
                        det.lwh[0],
                        det.lwh[1],
                        det.lwh[2],
                        quat[0],
                        quat[1],
                        quat[2],
                        quat[3],
                        det.translation[0],
                        det.translation[1],
                        det.translation[2],
                        det.num_interior]
                    track_vals.append(values)
            
            track_vals = np.asarray(track_vals)
            df = pd.DataFrame(
                data=track_vals,
                columns=column_names)
            df = df.astype(column_dtypes)
        
        else:
            for track in self.trajectories:
                det = track.final_detection()
                # quaternion rotation around z axis
                quat = np.array([np.cos(det.heading/2), 0, 0, np.sin(det.heading/2)])
                # REGULAR_VEHICLE = only dummy class
                values = [
                    det.translation[0],
                    det.translation[1],
                    det.translation[2],
                    det.lwh[0],
                    det.lwh[1],
                    det.lwh[2],
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
        
        os.makedirs(os.path.join(self.out_path, self.split, self.log_id), exist_ok=True)
        write_path = os.path.join(self.out_path, self.split, self.log_id, 'annotations.feather')
        logger.info(f'Stored tracks for sequence {self.log_id} at {os.getcwd()}/{write_path}')
        feather.write_feather(df, write_path)

        if visualize:
            self.visualize_whole(df, track.log_id)
        
        return True

    def visualize_whole(self, df, seq):
        gt = feather.read_feather(self.filtered_gt)
        gt = gt[gt['seq'] == self.log_id]

        track_colors = {t: c for t, c in zip(df['track_uuid'].unique(), cols)}
        
        x_lim = (np.min(df['tx_m'].values) - 10, np.max(df['tx_m'].values) + 10)
        y_lim = (np.min(df['ty_m'].values) - 10, np.max(df['ty_m'].values) + 10)

        os.makedirs(f'../../../Visualization_Whole/{seq}', exist_ok=True)
        for i, timestamp in enumerate(sorted(df['timestamp_ns'].unique())):
            time_df = df[df['timestamp_ns'] == timestamp]
            time_gt = gt[gt['timestamp_ns'] == timestamp]
            fig, ax = plt.subplots()
            for i, track_id in enumerate(time_df['track_uuid'].unique()):
                track_df = time_df[time_df['track_uuid'] == track_id]
                plt.scatter(track_df['tx_m'], track_df['ty_m'], color=track_colors[track_id], marker='o')
                x_0 = track_df['tx_m'].values-0.5*track_df['length_m'].values
                y_0 = track_df['ty_m'].values-0.5*track_df['width_m'].values
                rect = patches.Rectangle(
                    (x_0, y_0),
                    track_df['length_m'].values, 
                    track_df['width_m'].values,
                    linewidth=1,
                    edgecolor=track_colors[track_id],
                    facecolor='none')
                ax.add_patch(rect)

            for i, track_id in enumerate(time_gt['track_uuid'].unique()):
                track_gt = time_gt[time_gt['track_uuid'] == track_id]
                if _class_dict[track_gt['category'].values.item()] == 'PEDESTRIAN':
                    marker = '*'
                else:
                    marker = 'o'
                plt.scatter(track_gt['tx_m'], track_gt['ty_m'], color='black', marker=marker)
                x_0 = track_gt['tx_m'].values-0.5*track_gt['length_m'].values
                y_0 = track_gt['ty_m'].values-0.5*track_gt['width_m'].values
                rect = patches.Rectangle(
                    (x_0, y_0),
                    track_gt['length_m'].values,
                    track_gt['width_m'].values,
                    linewidth=1,
                    edgecolor='black',
                    facecolor='none')
                ax.add_patch(rect)

            plt.axis('off')
            plt.xlim(x_lim)
            plt.ylim(y_lim)
            plt.savefig(f'../../../Visualization_Whole/{seq}/frame_{timestamp}.jpg')
            plt.close()
            

class Track():
    def __init__(self, trajectory, track_id, every_x_frame, overlap) -> None:
        self.trajectories = [trajectory]
        self.inactive_count = 0
        self.track_id = track_id
        self.log_id = trajectory.log_id
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.final = list()
    
    def add_trajectory(self, trajectory):
        self.trajectories.append(trajectory)
    
    def final_detections(self, av2_loader):
        for i, traj in enumerate(self.trajectories):
            if i != len(self.trajectories) or self.inactive_count != 0:
                _range = self.every_x_frame
            else:
                _range = traj.trajectory.shape[1]

            city_SE3_ego0 = av2_loader.get_city_SE3_ego(self.log_id, traj.timestamps[0].item())

            for time in range(_range):
                points_c_time = traj.canonical_points + traj.trajectory[:, time, :]

                city_SE3_ego = av2_loader.get_city_SE3_ego(self.log_id, traj.timestamps[time].item())
                ego_SE3_ego0 = city_SE3_ego.inverse().compose(city_SE3_ego0)
                points_c_time = ego_SE3_ego0.transform_point_cloud(points_c_time)

                mins, maxs = points_c_time.min(axis=0), points_c_time.max(axis=0)
                lwh = maxs - mins

                translation = (maxs + mins)/2
                if time < traj.trajectory.shape[1] - 1:
                    mean_flow = (traj.trajectory[:, time+1, :] - traj.trajectory[:, time, :]).mean(axis=0)
                else:
                    mean_flow = (traj.trajectory[:, time, :] - traj.trajectory[:, time-1, :]).mean(axis=0)
                alpha = np.arctan(mean_flow[1]/mean_flow[0])
                rot = np.array([
                    [np.cos(alpha), np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])

                '''city_SE3_ego = av2_loader.get_city_SE3_ego(self.log_id, traj.timestamps[time].item())
                ego_SE3_ego0 = city_SE3_ego.inverse().compose(city_SE3_ego0)
                from av2.geometry.se3 import SE3
                ego0_SE3_object = SE3(rotation=rot, translation=translation)
                target_SE3_object = ego_SE3_ego0.compose(ego0_SE3_object)
                rot = target_SE3_object.rotation
                translation = target_SE3_object.translation'''
                self.final.append(Detection(rot, translation, lwh, traj.timestamps[time], traj.log_id, traj.num_interior))
            
    def _get_last_traj(self):
        traj = self.trajectories[-1].trajectory[:, self.every_x_frame:self.every_x_frame+self.overlap]
        traj = traj - np.tile(np.expand_dims(traj[:, 0, :], axis=1) , (1, traj.shape[1], 1))
        return traj
    
    def _get_last_traj_and_convert_time(self, timestamp, av2_loader):
        start = (1 + self.inactive_count) * (self.every_x_frame)
        end = start + self.overlap + 1

        assert end < self.trajectories[-1].length, 'inactive track too short'

        traj = copy.deepcopy(self.trajectories[-1].trajectory[:, start:end])
        t0 = self.trajectories[-1].timestamps[0].item()
        t1 = timestamp

        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)
        city_SE3_t1 = av2_loader.get_city_SE3_ego(self.log_id, t1)
        t1_SE3_t0 = city_SE3_t1.inverse().compose(city_SE3_t0)
        traj = t1_SE3_t0.transform_point_cloud(traj)
        traj_from_overlap = traj - np.tile(np.expand_dims(traj[:, 0, :], axis=1) , (1, traj.shape[1], 1))
        return traj_from_overlap[:, 1:], traj[:, 1:]
    
    def _get_last_canonical_point(self):
        canonical_points = self.trajectories[-1].canonical_points
        return canonical_points
    
    def _get_last_canonical_point_and_convert_time(self, timestamp, av2_loader):
        canonical_points = copy.deepcopy(self.trajectories[-1].canonical_points)
        t0 = self.trajectories[-1].timestamps[0].item()
        t1 = timestamp
        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)
        city_SE3_t1 = av2_loader.get_city_SE3_ego(self.log_id, t1)
        t1_SE3_t0 = city_SE3_t1.inverse().compose(city_SE3_t0)
        
        canonical_points = t1_SE3_t0.transform_point_cloud(canonical_points)
        return canonical_points
    
    def _get_canonical_point_at_start_of_overlap_and_convert_time(self, timestamp, av2_loader):
        start = (1 + self.inactive_count) * (self.every_x_frame)
        canonical_points = copy.deepcopy(self.trajectories[-1].canonical_points)
        canonical_points += copy.deepcopy(self.trajectories[-1].trajectory[:, start])
        t0 = self.trajectories[-1].timestamps[0].item()
        t1 = timestamp
        city_SE3_t0 = av2_loader.get_city_SE3_ego(self.log_id, t0)
        city_SE3_t1 = av2_loader.get_city_SE3_ego(self.log_id, t1)
        t1_SE3_t0 = city_SE3_t1.inverse().compose(city_SE3_t0)

        canonical_points = t1_SE3_t0.transform_point_cloud(canonical_points)
        return canonical_points

    def __len__(self):
        if len(self.final):
            return len(self.final)
        return len(self.trajectories)
    
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


class Trajectory():
    def __init__(self, trajectory, canonical_points, timestamps, log_id, num_interior, overlap, gt_id) -> None:
        self.trajectory = trajectory
        self.canonical_points = canonical_points
        self.timestamps = timestamps
        self.log_id = log_id
        self.num_interior = num_interior
        self.overlap = overlap
        self.gt_id = gt_id
        self.length = trajectory.shape[1]
    
    def _get_traj(self):
        traj = self.trajectory[:, 1:self.overlap+1]
        return traj
    
    def _get_canonical_point(self):
        canonical_points = self.canonical_points
        return canonical_points

    def final_detection(self):
        points_c_time = self.canonical_points
        mins, maxs = points_c_time.min(axis=0), points_c_time.max(axis=0)
        lwh = maxs - mins

        translation = (maxs + mins)/2
        mean_flow = (self.trajectory[:, 1, :] - self.trajectory[:, 0, :]).mean(axis=0)
        alpha = np.arctan(mean_flow[1]/mean_flow[0])
        rot = np.array([
            [np.cos(alpha), np.sin(alpha), 0],
            [np.sin(alpha), np.cos(alpha), 0],
            [0, 0, 1]])

        final = Detection(rot, alpha, translation, lwh, self.timestamps[0], self.log_id, self.num_interior)

        return final

class Detection():
    def __init__(self, rotation, heading, translation, lwh, timestamp, log_id, num_interior) -> None:
        self.rotation = rotation
        self.heading = heading
        self.translation = translation
        self.lwh = lwh
        self.timestamp = timestamp
        self.log_id = log_id
        self.num_interior = num_interior
