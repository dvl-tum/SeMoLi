from av2.geometry.se3 import SE3
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import os
import copy
import open3d as o3
import numpy as np
import torch
import pytorch3d
from .tracking_utils import get_rotated_center_and_lwh, _create_box, get_alpha_rot_t0_to_t1, outlier_removal
from numpy.linalg import inv


class ConstrainedICPRegistration():
    def __init__(self, active_tracks, av2_loader, log_id, threshold=0.1, kNN=10, exp_weight_rot=0, registration_len_thresh=4, min_pts_thresh=25):
        self.active_tracks = active_tracks
        self.av2_loader = av2_loader
        self.min_pts_thresh = min_pts_thresh
        self.log_id = log_id
        self.registration_len_thresh = registration_len_thresh
        self.ordered_timestamps = av2_loader.get_ordered_log_lidar_timestamps(log_id)

    def load_pointclouds(self, pt1, pt2, return_numpy=False):
        pc1_centroid = pt1.mean(axis=0)
        if return_numpy:
            return pt1, pt2, pc1_centroid
        pc1 = o3.geometry.PointCloud()
        pc1.points = o3.Vector3dVector(pt1)
        pc2 = o3.geometry.PointCloud()
        pc2.points = o3.Vector3dVector(pt2)
        return pc1, pc2, pc1_centroid
    
    def get_centroid_init(self, pc1, pc2):
        approx_translation = np.mean(np.asarray(pc2.points), axis=0) - np.mean(np.asarray(pc1.points), axis=0)
        init = np.eye(4)
        init[:3, 3] = approx_translation
        return init
    
    def get_grid_init(self, lwh_tgt, alpha_tgt, alpha_src):
        inits = list()
        c = np.cos(alpha_tgt-alpha_src)
        s = np.sin(alpha_tgt-alpha_src)
        for prod_l in [-0.5, -0.25, 0, 0.25, 0.5]:
            for prod_w in [-0.5, -0.25, 0, 0.25, 0.5]:
                init = np.eye(4)
                init[0, 0] = c
                init[0, 1] = -s
                init[1, 0] = s
                init[1, 1] = c
                init[:3, 3] = lwh_tgt*np.array([prod_l, prod_w, 0])
                inits.append(init)
        return inits

    def icp_p2point(self, pc1, pc2, lwh_tgt, alpha_tgt, alpha_src, radius=0.2, its=30, init=None, with_constraint=True):
        pc1, pc2, pc1_centroid = self.load_pointclouds(pc1, pc2)
        # init = self.get_centroid_init(pc1, pc2)
        inits = self.get_grid_init(lwh_tgt, alpha_tgt, alpha_src)
        inlier_rmse = 100000000
        reg_p2p = None
        for init in inits:
            _reg_p2p = o3.registration_icp(
                pc1,
                pc2,
                radius,
                init,
                o3.TransformationEstimationPointToPoint(with_constraint=with_constraint, with_scaling=False),
                o3.registration.ICPConvergenceCriteria(max_iteration=its))
            # print(init, _reg_p2p.inlier_rmse, inlier_rmse,  _reg_p2p.transformation)
            if  np.asarray(_reg_p2p.correspondence_set).shape[0] == 0:
                continue
            # print('ye')
            if _reg_p2p.inlier_rmse < inlier_rmse:
                # print(init, _reg_p2p.inlier_rmse, inlier_rmse,  _reg_p2p.transformation)
                # print()
                reg_p2p = _reg_p2p
                inlier_rmse = _reg_p2p.inlier_rmse 
        if reg_p2p is None:
            print('failed')
            reg_p2p = o3.registration_icp(
                pc1,
                pc2,
                radius,
                inits[12],
                o3.TransformationEstimationPointToPoint(with_constraint=with_constraint, with_scaling=False),
                o3.registration.ICPConvergenceCriteria(max_iteration=its))
        else:
            print('Success')
        pc1.transform(reg_p2p.transformation)
        rotation = reg_p2p.transformation[:-1, :-1]
        translation = reg_p2p.transformation[:-1, -1]
        return np.asarray(pc1.points), np.asarray(pc2.points), reg_p2p.transformation, translation, rotation

    def register(self, max_interior_thresh=50):
        detections = dict()
        for j, track in enumerate(self.active_tracks.values()):
            track_dets = list()
            _start = list()
            _end = list()
            dets = copy.deepcopy(track.detections)
            # we start from timestep with most points and then go
            # from max -> end -> start -> max
            max_interior_idx = torch.argmax(torch.tensor([d.num_interior for d in track.detections])).item()
            max_interior = torch.max(torch.tensor([d.num_interior for d in track.detections])).item()
            max_interior_pc = torch.atleast_2d(track._get_canonical_points(i=max_interior_idx))
            max_lwh = track.detections[max_interior_idx].lwh
            print(len(track), track.min_num_interior)
            if len(track) > self.registration_len_thresh and track.min_num_interior >= self.min_pts_thresh:
                # take all pcs and normalize them
                times = [d.timestamps[0, 0].item() for d in track.detections]
                
                pcs = [track._get_canonical_points(i=i) for i in range(len(track))] 
                pcs = [track._convert_time(times[i], times[max_interior_idx], self.av2_loader, pcs[i]) for i in range(len(track))]
                means = [pcs[i].mean(0) for i in range(len(track))]
                pcs = [pcs[i]-means[i] for i in range(len(track))]

                trajs = [track._get_traj(i=i) for i in range(len(track))]
                trajs = [track._convert_time(times[i], times[max_interior_idx], self.av2_loader, trajs[i]) for i in range(len(track))]
                
                max_to_end = range(max_interior_idx, len(track)-1)
                max_to_start = range(max_interior_idx, 0, -1)
                iterators = [max_to_end, max_to_start]
                increments = [+1, -1]

                tgt_SE3_src = [None] * len(track)

                # get target boudning box for ICP
                pc_tgt = pcs[max_interior_idx]
                lwh_tgt = track.detections[max_interior_idx].lwh
                _alpha_tgt = track.detections[max_interior_idx].alpha
                _, alpha_tgt = get_alpha_rot_t0_to_t1(0, 1, trajs[max_interior_idx])
                _pc_tgt = pc_tgt
                for iterator, increment in zip(iterators, increments):
                    for i in iterator:
                        pc_increment, _ = outlier_removal(pcs[i+increment], threshold=0.5, kNN=10)
                        # ICP
                        _alpha_src = track.detections[i+increment].alpha
                        _, alpha_src = get_alpha_rot_t0_to_t1(0, 1, trajs[i+increment])
                        pc_increment_registered, pc_tgt, transformation, translation, rotation = self.icp_p2point(
                            pc_increment,
                            pc_tgt,
                            lwh_tgt,
                            alpha_tgt,
                            alpha_src)
                    
                        pc_tgt = np.concatenate([pc_tgt, pc_increment_registered])
                        tgt_SE3_src[i+increment] = SE3(rotation=rotation, translation=translation)
                        #  concatenate cano points at t+increment and registered points as
                        # point cloud for next timestamp / final point cloud
                pc_tgt, _ = outlier_removal(torch.from_numpy(pc_tgt), threshold=0.5, kNN=10)
                pc_tgt = pc_tgt.numpy()

                rot = track.detections[max_interior_idx].rot
                rot = torch.tensor([
                    [torch.cos(alpha_tgt), -torch.sin(alpha_tgt), 0],
                    [torch.sin(alpha_tgt), torch.cos(alpha_tgt), 0],
                    [0, 0, 1]]).double()
                lwh, translation_tgt = get_rotated_center_and_lwh(torch.from_numpy(pc_tgt), rot)
                track.detections[max_interior_idx].translation = translation_tgt + means[max_interior_idx]
                track.detections[max_interior_idx].lwh = lwh

                max_to_end = range(max_interior_idx, len(track)-1)
                max_to_start = range(max_interior_idx, 0, -1)
                iterators = [max_to_end, max_to_start]
                increments = [+1, -1]
                for iterator, increment in zip(iterators, increments):
                    translation = torch.atleast_2d(translation_tgt)
                    for i in iterator:
                        translation = torch.atleast_2d(translation_tgt)
                        translation =  tgt_SE3_src[i+increment].inverse().transform_point_cloud(translation)
                        _translation = track._convert_time(times[max_interior_idx], times[i+increment], self.av2_loader, translation).squeeze()
                        track.detections[i+increment].translation = _translation + means[i+increment]
                        track.detections[i+increment].lwh = lwh

                self.visualize(dets, track.detections, track.track_id, pc_tgt, self.log_id, [], [torch.from_numpy(pc_tgt)+means[max_interior_idx]])            

                # self.visualize(list(), list(), track.track_id, registered_pc, self.log_id, [track._get_canonical_points(i=i) for i in range(len(track))], [registered_pc])            
                assert len(dets) == len(track.detections)

            # track.fill_detections(self.av2_loader, self.ordered_timestamps, max_time=25)
            detections[j] = track.detections
        return detections

    def visualize(self, dets, registered_dets, track_id, start_in_t0, log_id, _start, _end):
        os.makedirs(f'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Visualization_reg/Const_ICP/{self.log_id}', exist_ok=True)
        fig, ax = plt.subplots()
        for start_in_t0 in _start:
            plt.scatter(start_in_t0[:, 0], start_in_t0[:, 1], color='green', s=2)
        for start_in_t0 in _end:
            plt.scatter(start_in_t0[:, 0], start_in_t0[:, 1], color='pink', s=2)

        for det in dets:
            self.add_patch(ax, det)

        for det in registered_dets:
            self.add_patch(ax, det, color='blue', add=10)
        ax.axis('equal')
        plt.savefig(
                f'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Visualization_reg/Const_ICP/{self.log_id}/frame_{track_id}_{log_id}.jpg')
        plt.close()
        
    def add_patch(self, ax, det, color='black', add=0):
        plt.scatter(det.translation[0], det.translation[1], color=color, marker='o', s=2)
        loc_0 = det.translation[:-1]-0.5*det.lwh[:-1]
        t = matplotlib.transforms.Affine2D().rotate_around(det.translation[0], det.translation[1], det.alpha) + ax.transData
        rect = patches.Rectangle(
            loc_0,
            det.lwh[0],
            det.lwh[1],
            linewidth=1,
            edgecolor=color,
            facecolor='none',
            transform=t)
        ax.add_patch(rect)
