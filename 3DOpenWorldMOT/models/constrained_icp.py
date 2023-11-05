import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import os
import copy
import open3d as o3
import numpy as np
import torch
import pytorch3d


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
    
    def get_grid_init(self, translation, lwh, alpha_2, alpha_1):
        inits = list()
        # print('------------------> A1 A2', alpha_2, alpha_1)
        c = np.cos(alpha_2-alpha_1)
        s = np.sin(alpha_2-alpha_1)
        for prod_l in [-0.5, -0.25, 0, 0.25, 0.5]:
            for prod_w in [-0.5, -0.25, 0, 0.25, 0.5]:
                init = np.eye(4)
                # init[0, 0] = c
                # init[0, 1] = -s
                # init[1, 0] = s
                # init[1, 1] = c
                init[:3, 3] = lwh*np.array([prod_l, prod_w, 0])
                inits.append(init)
        return inits

    def icp_p2point(self, pc1, pc2, translation, lwh, alpha_2, alpha_1, radius=0.2, its=30, init=None, with_constraint=True):
        pc1, pc2, pc1_centroid = self.load_pointclouds(pc1, pc2)
        # init = self.get_centroid_init(pc1, pc2)
        inits = self.get_grid_init(translation, lwh, alpha_2, alpha_1)
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
            reg_p2p = o3.registration_icp(
                pc1,
                pc2,
                radius,
                np.eye(4),
                o3.TransformationEstimationPointToPoint(with_constraint=with_constraint, with_scaling=False),
                o3.registration.ICPConvergenceCriteria(max_iteration=its))
        
        pc1.transform(reg_p2p.transformation)
        rotation = reg_p2p.transformation[:-1, :-1]
        translation = reg_p2p.transformation[:-1, -1]
        return np.asarray(pc1.points), np.asarray(pc2.points), translation, rotation

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
            print(len(track) > self.registration_len_thresh and track.min_num_interior >= self.min_pts_thresh, len(track), self.registration_len_thresh, track.min_num_interior, self.min_pts_thresh)
            if len(track) > self.registration_len_thresh and track.min_num_interior >= self.min_pts_thresh:
                max_to_end = range(max_interior_idx, len(track)-1)
                max_to_start = range(max_interior_idx, 0, -1)
                iterators = [max_to_end, max_to_start]
                increment = [+1, -1]

                SimilarityTransforms = [None] * len(track)
                for iterator, increment in zip(iterators, increment):
                    start_in_t0 = copy.deepcopy(max_interior_pc)
                    for i in iterator:
                        # convert pc and trajectory t0 --> t1 from ego frame t=i to ego frame t=i+1
                        t0 = track.detections[i].timestamps[0, 0].item()
                        t1 = track.detections[i+increment].timestamps[0, 0].item()
                        start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)

                        # get points in time t_1
                        cano1_in_t1 = torch.atleast_2d(track._get_canonical_points(i=i+increment))

                        # if track.detections[i+1].num_interior > self.min_pts_thresh:
                        # ICP
                        start_registered, cano1_in_t1, translation, rotation = self.icp_p2point(
                            start_in_t1,
                            cano1_in_t1,
                            track.detections[i+increment].translation,
                            track.detections[i+increment].lwh,
                            track.detections[i+increment].alpha,
                            track.detections[i].alpha)
                        _start.append(cano1_in_t1)
                        _end.append(start_registered)
                        print(track.detections[i+increment].translation, track.detections[i+increment].lwh, track.detections[i+increment].alpha)
                        #print('lwh', track.detections[i+increment].lwh)
                        #print('trans', track.detections[i+increment].translation)
                        #print()
                        track.detections[i+increment].update_after_registration(torch.from_numpy(start_registered), translation, rotation, max_lwh)
                        print(track.detections[i+increment].translation, track.detections[i+increment].lwh, track.detections[i+increment].alpha)
                        #print(track.detections[i+increment].lwh)
                        #print(track.detections[i+increment].translation)
                        #quit()
                        # concatenate cano points at t+increment and registered points as
                        # point cloud for next timestamp / final point cloud
                        start_in_t0 = start_registered # torch.cat([start_registered, cano1_in_t1])
                self.visualize(dets, track.detections, track.track_id, max_interior_pc, self.log_id, _start, _end)            
                print(len(dets), len(track.detections))
                quit()
            # track.fill_detections(self.av2_loader, self.ordered_timestamps, max_time=25)
            detections[j] = track.detections
        return detections

    def visualize(self, dets, registered_dets, track_id, start_in_t0, log_id, _start, _end):
        os.makedirs('/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Visualization_reg', exist_ok=True)
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
                f'/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/Visualization_reg/frame_{track_id}_{log_id}.jpg', dpi=1000)
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
