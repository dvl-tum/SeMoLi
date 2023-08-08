import torch
import pytorch3d.loss
from av2.geometry.se3 import SE3
from .tracking_utils import get_rotated_center_and_lwh


class FlowRegistration():
    def __init__(self, active_tracks, av2_loader, log_id):
        self.active_tracks = active_tracks
        self.av2_loader = av2_loader
        self.ordered_timestamps = av2_loader.get_ordered_log_lidar_timestamps(log_id)
    
    def register(self):
        detections = dict()
        for j, track in enumerate(self.active_tracks):
            track_dets = list()
            # we start from timestep with most points and then go
            # from max -> end -> start -> max

            if len(track) > 1:
                for i in range(len(track)-1):
                    start_in_t0 = torch.atleast_2d(track._get_canonical_points(i=i))
                    traj_in_t0 = torch.atleast_2d(track._get_traj(i=i))

                    # convert pc and trajectory t0 --> t1 from ego frame t=i to ego frame t=i+1
                    t0 = track.detections[i].timestamps[0, 0].item()
                    t1 = track.detections[i+1].timestamps[0, 0].item()
                    dt = self.ordered_timestamps.index(t1) - self.ordered_timestamps.index(t0)
                    start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)
                    traj_dt_in_t1 = track._convert_time(t0, t1, self.av2_loader, traj_in_t0[dt])

                    # transform points to from t0 --> t1
                    # print(start_in_t1.shape, traj_dt_in_t1.shape, start_in_t0.shape, traj_in_t0.shape, traj_in_t0[dt].shape)
                    start_in_t1 = start_in_t1 + traj_dt_in_t1
                    
                    # stack points
                    cano1_in_t1 = torch.atleast_2d(track._get_canonical_points(i=i+1))
                    start_in_t0 = torch.cat([start_in_t1, cano1_in_t1])
                
                num_interior = start_in_t0.shape[0]

                # setting last detection
                lwh, translation = get_rotated_center_and_lwh(start_in_t0,  track.detections[-1].rot)
                track.detections[-1].lwh = lwh
                track.detections[i].translation = translation
                track.detections[i].num_interior = num_interior
                track_dets.append(track.detections[i])

                for i in range(len(track)-1, 0, -1):
                    t0 = track.detections[i].timestamps[0, 0].item()
                    t1 = track.detections[i-1].timestamps[0, 0].item()
                    start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)
                    lwh, translation = get_rotated_center_and_lwh(start_in_t1, track.detections[i-1].rot)
                
                    # setting last detection
                    track.detections[i].lwh = lwh
                    track.detections[i].translation = translation
                    track.detections[i].num_interior = num_interior
                    track_dets.append(track.detections[i])
                    track_dets.reverse()
            else:
                for i in range(len(track.detections)):
                    points = track._get_canonical_points(i=i)
                    lwh, translation = get_rotated_center_and_lwh(points, track.detections[i].rot)
                    
                    # setting last detection
                    track.detections[i].lwh = lwh
                    track.detections[i].translation = translation
                    track_dets.append(track.detections[i])

            detections[j] = track # track_dets
        
        return detections
