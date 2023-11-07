import torch
import pytorch3d.loss
from av2.geometry.se3 import SE3
from .tracking_utils import get_rotated_center_and_lwh, outlier_removal, get_alpha_rot_t0_to_t1
import copy
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import os
from scipy.spatial.transform import Rotation as R


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
    
    def get_rot(self, alpha):
        rotation = torch.tensor([
                [torch.cos(alpha), -torch.sin(alpha), 0],
                [torch.sin(alpha), torch.cos(alpha), 0],
                [0, 0, 1]]).double()
        return rotation

    def register(self, visualize=True, weight=0.75):
        detections = dict()
        for j, track in enumerate(self.active_tracks.values()):
            # only do registration if criteria fullfilled
            if track.min_num_interior >= self.min_pts_thresh and \
                len(track) >= self.registration_len_thresh:
                print(track.min_num_interior, self.min_pts_thresh, len(track), self.registration_len_thresh, track.min_num_interior >= self.min_pts_thresh and len(track) >= self.registration_len_thresh)
                pcs = [track._get_canonical_points(i=i) for i in range(len(track))]
                trajs = [track._get_traj(i=i) for i in range(len(track))]
                times = [d.timestamps[0, 0].item() for d in track.detections]
                ego_SE3_objs = [SE3(rotation=self.get_rot(d.alpha).numpy(), translation=d.translation.numpy()) for d in track.detections]
                pcs_obj = [ego_SE3_objs[i].inverse().transform_point_cloud(pcs[i].numpy()) for i in range(len(track))]
                trajs_obj = [ego_SE3_objs[i].inverse().transform_point_cloud(pcs[i].unsqueeze(1) + trajs[i]) for i in range(len(track))]

                # we start from t=0
                dets = list()
                old_SE3_news = list()
                translation_old = None
                alpha_old = None
                registered_pc = pcs[0]
                if len(track) > 1:
                    start_in_t0 = pcs_obj[0]
                    for i in range(len(track)-1):
                        dets.append(copy.deepcopy(track.detections[i]))
                        
                        _, alpha_1 = get_alpha_rot_t0_to_t1(0, 1, trajs_obj[i+1])
                        _, alpha_0 = get_alpha_rot_t0_to_t1(0, 1, trajs_obj[i])
                        alpha = alpha_1 - alpha_0

                        t0 = self.ordered_timestamps.index(track.detections[i].timestamps[0, 0].item())
                        t1 = self.ordered_timestamps.index(track.detections[i+1].timestamps[0, 0].item())
                        dt = t1 - t0
                        translation = trajs[i][:, dt].mean(dim=0)
                        translation[2] = 0

                        if translation_old is not None:
                            translation = translation_old * weight + translation * (1-weight)
                            alpha = alpha_old * weight + alpha * (1-weight)
                        translation_old = translation
                        alpha_old = alpha

                        rotation = torch.tensor([
                            [torch.cos(alpha), -torch.sin(alpha), 0],
                            [torch.sin(alpha), torch.cos(alpha), 0],
                            [0, 0, 1]]).double()

                        old_SE3_new = SE3(rotation=rotation.cpu().numpy(), translation=translation.cpu().numpy())
                        old_SE3_news.append(old_SE3_new)
                        registered_pc = old_SE3_new.inverse().transform_point_cloud(registered_pc)
                        # stack points
                        registered_pc = torch.cat([registered_pc, pcs[i+1]])

                    dets.append(copy.deepcopy(track.detections[-1]))

                    # remove ourliers based on kNN
                    if self.kNN > 0:
                        registered_pc, _ = outlier_removal(registered_pc, threshold=self.threshold, kNN=self.kNN)

                    # if not only outliers
                    if registered_pc.shape[0] != 0:
                        # setting last detection
                        rotation = torch.tensor([
                            [1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]]).double()

                        lwh, translation = get_rotated_center_and_lwh(registered_pc, rotation, median=False)

                        track.detections[-1].lwh = lwh
                        track.detections[-1].translation = translation #dets[-1].translation #translation
                        num_interior = registered_pc.shape[0]
                        track.detections[-1].num_interior = num_interior
                        for i in range(len(track)-1, 0, -1):
                            translation = old_SE3_news[i-1].transform_point_cloud(translation)
                            _translation = ego_SE3_objs[i].transform_point_cloud(translation).squeeze()
                            # setting last detection
                            track.detections[i-1].lwh = lwh
                            track.detections[i-1].translation = _translation #dets[i-1].translation #translation
                            track.detections[i-1].num_interior = num_interior

                # track.fill_detections(self.av2_loader, self.ordered_timestamps, max_time=5)
                if visualize:
                    self.visualize(dets, track.detections, track.track_id, registered_pc, self.log_id)
                quit()
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


class FlowRegistration_From_Max(FlowRegistration):
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

    def rot_to_alpha(self, rot):
        print(rot)
        r = R.from_matrix(rot)
        alphas = r.as_euler('zyx')
        print(alphas)
        return alphas[-1]


    def register(self, visualize=True, weight=0.75):
        detections = dict()
        for j, track in enumerate(self.active_tracks.values()):
            # only do registration if criteria fullfilled
            max_interior_idx = torch.argmax(torch.tensor([d.num_interior for d in track.detections])).item()
            max_interior = torch.max(torch.tensor([d.num_interior for d in track.detections])).item()
            max_interior_pc = torch.atleast_2d(track._get_canonical_points(i=max_interior_idx))
            max_lwh = track.detections[max_interior_idx].lwh
            # print(len(track) > self.registration_len_thresh and track.min_num_interior >= self.min_pts_thresh, len(track), self.registration_len_thresh, track.min_num_interior, self.min_pts_thresh)
            if len(track) > self.registration_len_thresh and track.min_num_interior >= self.min_pts_thresh:
                print(track.min_num_interior, self.min_pts_thresh, len(track), self.registration_len_thresh, track.min_num_interior >= self.min_pts_thresh and len(track) >= self.registration_len_thresh)
                dets = copy.deepcopy(track.detections)
                _pcs = [track._get_canonical_points(i=i) for i in range(len(track))]
                _trajs = [track._get_traj(i=i) for i in range(len(track))]
                times = [d.timestamps[0, 0].item() for d in track.detections]
                pcs = [track._convert_time(times[i], times[max_interior_idx], self.av2_loader, _pcs[i]) for i in range(len(track))]
                trajs = [track._convert_time(times[i], times[max_interior_idx], self.av2_loader, _pcs[i].unsqueeze(1) + _trajs[i]) for i in range(len(track))]
                trajs = [trajs[i] - pcs[i].unsqueeze(1) for i in range(len(track))]
                means = [pcs[i].mean(0) for i in range(len(track))]
                pcs = [pcs[i]-means[i] for i in range(len(track))]

                # take all pcs and normalize them
                max_to_end = range(max_interior_idx, len(track)-1)
                max_to_start = range(max_interior_idx, 0, -1)
                iterators = [max_to_end, max_to_start]
                increments = [+1, -1]

                new_SE3_olds = [None] * len(track)
                registered_pc = pcs[max_interior_idx]
                for iterator, increment in zip(iterators, increments):
                    translation_old = None
                    alpha_old = None
                    for i in iterator:

                        # convert pc and trajectory t0 --> t1 from ego frame t=i to ego frame t=i+1
                        if track.detections[i].timestamps[0, 0].item() < track.detections[i+increment].timestamps[0, 0].item():
                            _, alpha_1 = get_alpha_rot_t0_to_t1(0, 1, trajs[i+increment])
                            _, alpha_0 = get_alpha_rot_t0_to_t1(0, 1, trajs[i])
                            alpha = alpha_1 - alpha_0

                            t0 = self.ordered_timestamps.index(track.detections[i].timestamps[0, 0].item())
                            t1 = self.ordered_timestamps.index(track.detections[i+increment].timestamps[0, 0].item())
                            dt = t1 - t0
                            translation = trajs[i][:, dt].mean(dim=0)
                            translation[2] = 0
                            print('a', i+increment, translation, alpha)

                            if translation_old is not None:
                                translation = translation_old * weight + translation * (1-weight)
                                alpha = alpha_old * weight + alpha * (1-weight)
                            translation_old = translation
                            alpha_old = alpha

                            rotation = torch.tensor([
                                [torch.cos(alpha), -torch.sin(alpha), 0],
                                [torch.sin(alpha), torch.cos(alpha), 0],
                                [0, 0, 1]]).double()

                            old_SE3_new = SE3(rotation=rotation.cpu().numpy(), translation=translation.cpu().numpy())
                            new_SE3_olds[i+increment] = new_SE3_old
                        else:                        
                            _, alpha_1 = get_alpha_rot_t0_to_t1(0, 1, trajs[i])
                            _, alpha_0 = get_alpha_rot_t0_to_t1(0, 1, trajs[i+increment])
                            alpha = alpha_0 -alpha_1

                            t0 = self.ordered_timestamps.index(track.detections[i+increment].timestamps[0, 0].item())
                            t1 = self.ordered_timestamps.index(track.detections[i].timestamps[0, 0].item())
                            dt = t1 - t0
                            translation = trajs[i+increment][:, dt].mean(dim=0)
                            translation[2] = 0 
                            print('b', i+increment, translation, alpha)

                            if translation_old is not None:
                                translation = translation_old * weight + translation * (1-weight)
                                alpha = alpha_old * weight + alpha * (1-weight)
                            translation_old = translation
                            alpha_old = alpha

                            rotation = torch.tensor([
                                [torch.cos(alpha), -torch.sin(alpha), 0],
                                [torch.sin(alpha), torch.cos(alpha), 0],
                                [0, 0, 1]]).double()

                            new_SE3_old = SE3(rotation=rotation.cpu().numpy(), translation=translation.cpu().numpy()).inverse()
                            new_SE3_olds[i+increment] = new_SE3_old
                        registered_pc = new_SE3_old.transform_point_cloud(registered_pc)
                        # stack points
                        registered_pc = torch.cat([registered_pc, pcs[i+increment]])

                # if not only outliers
                if registered_pc.shape[0] != 0:
                    # setting last detection
                    lwh = track.detections[max_interior_idx].lwh
                    print(track.detections[max_interior_idx].translation, means[max_interior_idx])
                    num_interior = track.detections[max_interior_idx].num_interior
                    
                    max_to_end = range(max_interior_idx, len(track)-1)
                    max_to_start = range(max_interior_idx, 0, -1)
                    iterators = [max_to_end, max_to_start]
                    increments = [+1, -1]

                    for iterator, increment in zip(iterators, increments):
                        translation = torch.atleast_2d(track.detections[max_interior_idx].translation  - means[max_interior_idx])
                        for i in iterator:
                            print(max_interior_idx, i+increment)
                            print(translation, track.detections[i+increment].translation)
                            print(new_SE3_olds[i+increment].translation, new_SE3_olds[i+increment].rotation[0, :])
                            translation = new_SE3_olds[i+increment].transform_point_cloud(translation).squeeze()

                            _translation = track._convert_time(times[max_interior_idx], times[i+increment], self.av2_loader, translation) + means[i+increment]
                            print(translation, _translation)
                            print()
                            # setting last detection
                            track.detections[i+increment].lwh = lwh
                            track.detections[i+increment].translation = _translation #dets[i-1].translation #translation
                            track.detections[i+increment].num_interior = num_interior

                # track.fill_detections(self.av2_loader, self.ordered_timestamps, max_time=5)
                if visualize:
                    self.visualize(dets, track.detections, track.track_id, pcs[max_interior_idx], self.log_id)
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
