from gettext import translation
import os
import sklearn.metrics
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.optimize import linear_sum_assignment
from pyarrow import feather
import pandas as pd
import logging

logger = logging.getLogger("Model.Tracker")


class Tracker3D():
    def __init__(self, out_path='out', a_threshold=0.8, i_threshold=0.8, split='val', inact_thresh=10, every_x_frame=1, num_interior=10) -> None:
        self.active_tracks = list()
        self.inactive_tracks = list()
        self.track_id = 0
        self.log_id = -1
        self.split = split
        self.inact_thresh = inact_thresh
        self.every_x_frame = every_x_frame
        self.num_interior = num_interior

        self.a_threshold = a_threshold
        self.i_threshold = i_threshold
        self.out_path = os.path.join(out_path)
    
    def new_log_id(self, log_id):
        if self.log_id != -1:
            self.active_tracks += self.inactive_tracks
            self.to_feather()
            self.active_tracks = list()
            self.inactive_tracks = list()
        self.log_id = log_id
    
    def get_detections(self, points, traj, clusters, flows, timestamp, log_id):
        if self.log_id != log_id:
            self.new_log_id(log_id)
        
        points = points.numpy()
        flows = flows.numpy()

        # iterate over clusters that were found
        detections = list()
        for c in np.unique(clusters):
            num_interior = sum([clusters==c])
            if num_interior < self.num_interior:
                continue
            c_detections = list()
            if c == -1:
                continue
            point_cluster = points[clusters==c]
            traj_cluster = traj[clusters==c]
            for time in range(traj_cluster.shape):
                points_c_time = point_cluster + traj_cluster[:, time, :]

                num_interior = points_c_time.shape[0]
                mins, maxs = points_c_time.min(axis=0), points_c_time.max(axis=0)
                lwh = maxs - mins
                if lwh[0] > 50 or lwh[1] > 50 or lwh[2] > 4:
                    continue
                if lwh[2] == 0:
                    continue
                if (lwh[0]/lwh[2] > 4 or lwh[1]/lwh[2] < 4) and lwh[2] < 0.5:
                    continue

                translation = (maxs + mins)/2
                mean_flow = (traj_cluster[:, time+1, :] - traj_cluster[:, time, :]).mean(axis=0)
                alpha = np.arctan(mean_flow[1]/mean_flow[0])
                rot = np.array([
                    [np.cos(alpha), -np.sin(alpha), 0],
                    [np.sin(alpha), np.cos(alpha), 0],
                    [0, 0, 1]])
                
                c_detections.append(Detection(rot, translation, lwh, timestamp, log_id, num_interior))
            
            detections.append(c_detections)
        
        return detections

    def associate(self, detections, last=False):
        # add tracks if no tracks yet
        if not len(self.active_tracks):
            self.active_tracks = list()
            for d in detections:
                self.active_tracks.append(Track(d, self.track_id))
                self.track_id += 1
            return
        
        if len(detections) == 0:
            return

        # calculate costs and match
        cost_mat, num_act, num_inact = self._calculate_3d_euclidean(detections)
        # cost_mat[cost_mat > self.threshold - np.finfo('float').eps] = 1
        match_rows, match_cols = linear_sum_assignment(cost_mat)

        a = match_rows < cost_mat.shape[0] - num_inact
        i = match_rows >= cost_mat.shape[0] - num_inact

        actually_matched_rows = match_rows[a][cost_mat[match_rows[a], match_cols[a]] < self.a_threshold]
        actually_matched_cols = match_cols[a][cost_mat[match_rows[a], match_cols[a]] < self.a_threshold]

        inact_actually_matched_rows = match_rows[i][cost_mat[match_rows[i], match_cols[i]] < self.a_threshold]
        inact_actually_matched_cols = match_cols[i][cost_mat[match_rows[i], match_cols[i]] < self.a_threshold]

        if inact_actually_matched_rows.shape[0]:
            actually_matched_rows = np.hstack([actually_matched_rows, inact_actually_matched_rows])
            actually_matched_cols = np.hstack([actually_matched_cols, inact_actually_matched_cols])

        to_move = list()
        for r, c in zip(actually_matched_rows, actually_matched_cols):
            if r <= num_act - 1:
                self.active_tracks[r].add_detection(detections[c])
            else:
                self.inactive_tracks[r-(num_act)].add_detection(detections[c])
                self.active_tracks.append(self.inactive_tracks[r-(num_act)])
                to_move.append(r-(num_act))

        self.inactive_tracks = [t for r, t in enumerate(self.inactive_tracks)\
            if r not in to_move]

        # move tracks with no matching to inactive track
        to_move = list()
        for r in range(num_act):
            if r not in actually_matched_rows and len(self.active_tracks[r]) > 2:
                to_move.append(r)
                self.inactive_tracks.append(self.active_tracks[r])
            elif r not in actually_matched_rows and len(self.active_tracks[r]) <= 2:
                to_move.append(r)
        self.active_tracks = [t for r, t in enumerate(self.active_tracks)\
            if r not in to_move]

        # increase inactive count
        for i in range(len(self.inactive_tracks)):
            self.inactive_tracks[i].inactive_count += 1
        
        # start new tracks
        for c in range(len(detections)):
            if c not in actually_matched_cols:
                self.active_tracks.append(Track(detections[c], self.track_id))
                self.track_id += 1

        if last:
            self.active_tracks += self.inactive_tracks
            self.to_feather()
            self.active_tracks = list()
            self.inactive_tracks = list()

    def _calculate_3d_euclidean(self, detections, clip_val=10):
        '''
        input values of numpy array:
            "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"
        '''
        tracks = np.vstack([t.last_pos() for t in self.active_tracks])
        num_act = tracks.shape[0]

        in_tracks = [t.last_pos() for t in self.inactive_tracks \
                if t.inactive_count < self.inact_thresh]
        num_inact = len (in_tracks)
        if num_inact:
            in_tracks = np.vstack(in_tracks)
            tracks = np.vstack([tracks, in_tracks])

        detections = np.vstack([[t.translation for t in detections]])

        dist = sklearn.metrics.pairwise_distances(tracks, detections)
        dist = np.clip(dist, a_min=0, a_max=clip_val)
        dist = dist/clip_val

        return dist, num_act, num_inact
    
    def _calculate_3d_rotation(self, detections):
        '''
        FROM AV2 geometry / detection evaluation
        input values of numpy array:
            "tx_m", "ty_m", "tz_m", "length_m", "width_m", "height_m", "qw", "qx", "qy", "qz"
        '''
        tracks = np.vstack([t.rot for t in self.active_tracks])
        detections = np.vstack([[t.rot for t in detections]])
        # Convert a 3D rotation matrix to a sequence of _extrinsic_ rotations.
        xyz_rad1 = Rotation.from_matrix(tracks).as_euler("xyz", degrees=False)
        xyz_rad2 = Rotation.from_matrix(detections).as_euler("xyz", degrees=False)

        idx_tracks = [i for i in range(len(tracks)) for _ in range(len(detections))]
        idx_dets = [j for _ in range(len(tracks)) for j in range(len(detections))]

        xyz_rad1 = xyz_rad1[idx_tracks]
        xyz_rad2 = xyz_rad1[idx_dets]

        # angle distance
        angles = xyz_rad2 - xyz_rad1

        # Map angles to [0, ∞].
        angles = np.abs(angles)

        # Calculate floor division and remainder simultaneously.
        divs, mods = np.divmod(angles, np.pi)

        # Select angles which exceed specified period.
        angle_complement_mask = np.nonzero(divs)

        # Take set complement of `mods` w.r.t. the set [0, π].
        # `mods` must be nonzero, thus the image is the interval [0, π).
        angles[angle_complement_mask] = np.pi - mods[angle_complement_mask]
        angles = np.reshape(angles, (len(tracks, len(detections))))
        return angles
    
    def _calculate_3d_distance(self, bboxes1, bboxes2, dist='euclidean'):
        if dist == '3dIoU':
            similarity_scores = self._calculate_box_ious3D(bboxes1, bboxes2)
        elif dist == 'euclidean':
            similarity_scores = self._calculate_3d_euclidean(bboxes1, bboxes2)
        elif dist == 'rot':
            similarity_scores = self._calculate_3d_rotation(bboxes1, bboxes2)
        
        return similarity_scores
    
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
        for track in self.active_tracks:
            for det in track:
                quat = self.mat_to_quat(det.rotation)
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
            columns=['timestamp_ns', 'track_uuid', 'category', 'length_m', 'width_m',
            'height_m', 'qw', 'qx', 'qy', 'qz', 'tx_m', 'ty_m', 'tz_m', 'num_interior_pts'])

        os.makedirs(os.path.join(self.out_path, self.split, self.log_id), exist_ok=True)
        write_path = os.path.join(self.out_path, self.split, self.log_id, 'annotations.feather')
        # logger.info(f'Stored tracks for sequence {self.log_id} at {write_path}')
        feather.write_feather(df, write_path)


class Track():
    def __init__(self, detection, track_id) -> None:
        self.detections = [detection]
        self.inactive_count = 0
        self.track_id = track_id
        self.log_id = detection.log_id
    
    def add_detection(self, detection):
        self.detections.append(detection)
    
    def last_pos(self):
        return self.detections[-1].translation
    
    def last_rot(self):
        return self.detections[-1].rotation
    
    def last_lwh(self):
        return self.detections[-1].lwh
    
    def __len__(self):
        return len(self.detections)
    
    def __iter__(self):
        self.n = 0
        return self

    def __next__(self):
        if self.n < len(self.detections):
            result = self.detections[self.n]
            self.n += 1
            return result
        else:
            raise StopIteration


class Detection():
    def __init__(self, rotation, translation, lwh, timestamp, log_id, num_interior) -> None:
        self.rotation = rotation
        self.translation = translation
        self.lwh = lwh
        self.timestamp = timestamp
        self.log_id = log_id
        self.num_interior = num_interior
