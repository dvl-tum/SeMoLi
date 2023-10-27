import torch
import pytorch3d.loss
from av2.geometry.se3 import SE3
from .tracking_utils import get_rotated_center_and_lwh, outlier_removal
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import os


class FlowRegistration():
    def __init__(self, active_tracks, av2_loader, log_id, threshold=0.1, kNN=10, exp_weight_rot=0, registration_len_thresh=4, min_pts_thresh=25):
        self.active_tracks = active_tracks
        if type(self.active_tracks) == list:
            for t in self.active_tracks:
                t._sort_dets()
            self.active_tracks = {t.track_id: t for t in self.active_tracks}

        self.av2_loader = av2_loader
        self.ordered_timestamps = av2_loader.get_ordered_log_lidar_timestamps(log_id)
        self.threshold = threshold
        self.kNN = kNN
        self.log_id = log_id
        self.exp_weight_rot = exp_weight_rot
        self.registration_len_thresh = registration_len_thresh
        self.min_pts_thresh = min_pts_thresh

    def register(self, visualize=False):
        detections = dict()
        for j, track in enumerate(self.active_tracks.values()):
            print(track.min_num_interior, self.min_pts_thresh)
            # only do registration if criteria fullfilled
            if track.min_num_interior >= self.min_pts_thresh and \
                len(track) >= self.registration_len_thresh:
                # we start from t=0
                flows = list()
                dets = list()
                if len(track) > 1:
                    start_in_t0 = torch.atleast_2d(track._get_canonical_points(i=0))
                    for i in range(len(track)-1):
                        dets.append(copy.deepcopy(track.detections[i]))
                        
                        # start_in_t0 = torch.atleast_2d(track._get_canonical_points(i=i))
                        traj_in_t0 = torch.atleast_2d(track._get_traj(i=i))

                        # convert pc and trajectory t0 --> t1 from ego frame t=i to ego frame t=i+1
                        t0 = track.detections[i].timestamps[0, 0].item()
                        t1 = track.detections[i+1].timestamps[0, 0].item()
                        dt = self.ordered_timestamps.index(t1) - self.ordered_timestamps.index(t0)

                        # move point cloud and get mean flow
                        start_in_t0 += traj_in_t0[:, dt].median(dim=0).values
                        flows.append(traj_in_t0[:, dt].median(dim=0).values)
                        start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)

                        # stack points
                        cano1_in_t1 = torch.atleast_2d(track._get_canonical_points(i=i+1))
                        start_in_t0 = torch.cat([start_in_t1, cano1_in_t1])


                    dets.append(copy.deepcopy(track.detections[-1]))

                    # remove ourliers based on kNN
                    if self.kNN > 0:
                        start_in_t0, _ = outlier_removal(start_in_t0, threshold=self.threshold, kNN=self.kNN)

                    # if not only outliers
                    if start_in_t0.shape[0] != 0:
                        # setting last detection
                        rotation = track.detections[-1].rot
                        lwh, translation = get_rotated_center_and_lwh(start_in_t0,  rotation)
                        track.detections[-1].lwh = lwh
                        track.detections[-1].translation = translation #dets[-1].translation #translation
                        num_interior = start_in_t0.shape[0]
                        track.detections[-1].num_interior = num_interior
                        for i in range(len(track)-1, 0, -1):
                            t0 = track.detections[i].timestamps[0, 0].item()
                            t1 = track.detections[i-1].timestamps[0, 0].item()
                            start_in_t0 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)
                            start_in_t0 -= flows[i-1]
                            lwh, translation = get_rotated_center_and_lwh(start_in_t0, track.detections[i-1].rot)
                            # setting last detection
                            track.detections[i-1].lwh = lwh
                            track.detections[i-1].translation = translation #dets[i-1].translation #translation
                            track.detections[i-1].num_interior = num_interior

                # track.fill_detections(self.av2_loader, self.ordered_timestamps, max_time=5)
                if visualize:
                    self.visualize(dets, track.detections, track.track_id, start_in_t0, self.log_id)
            track_dets = track.detections

            detections[j] = track_dets

        return detections

    def visualize(self, dets, registered_dets, track_id, start_in_t0, log_id):
        os.makedirs('/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Visualization_reg', exist_ok=True)
        fig, ax = plt.subplots()
        plt.scatter(start_in_t0[:, 0], start_in_t0[:, 1], color='green', s=2)
        for det in dets:
            self.add_patch(ax, det)

        for det in registered_dets:
            self.add_patch(ax, det, color='blue', add=10)
        ax.axis('equal')
        plt.savefig(
                f'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Visualization_reg/frame_{track_id}_{log_id}.jpg', dpi=1000)
        plt.close()
        
    def add_patch(self, ax, det, color='black', add=0):
        plt.scatter(det.translation[0].cpu(), det.translation[1].cpu(), color=color, marker='o', s=2)
        loc_0 = det.translation[:-1]-0.5*det.lwh[:-1]
        t = matplotlib.transforms.Affine2D().rotate_around(det.translation[0].cpu(), det.translation[1].cpu(), det.alpha) + ax.transData
        rect = patches.Rectangle(
            loc_0.cpu(),
            det.lwh[0].cpu(),
            det.lwh[1].cpu(),
            linewidth=1,
            edgecolor=color,
            facecolor='none',
            transform=t)
        ax.add_patch(rect)
