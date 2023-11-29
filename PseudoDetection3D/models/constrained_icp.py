from av2.geometry.se3 import SE3
import copy
import open3d as o3
import numpy as np
import torch
from .tracking_utils import get_rotated_center_and_lwh, get_alpha_rot_t0_to_t1, outlier_removal


class ConstrainedICPRegistration():
    def __init__(self,
                 active_tracks,
                 av2_loader,
                 log_id,
                 outlier_threshold=0.1,
                 outlier_kNN=10,
                 registration_len_thresh=4,
                 min_pts_thresh=25,
                 mode='density',
                 density_thresh=0.005,
                 concat=True,
                 means_before=True,
                 avg_w_prev=False):
        
        self.active_tracks = active_tracks
        self.av2_loader = av2_loader
        self.log_id = log_id
        self.ordered_timestamps = av2_loader.get_ordered_log_lidar_timestamps(log_id)

        self.outlier_threshold = outlier_threshold
        self.outlier_kNN = outlier_kNN
        self.registration_len_thresh = registration_len_thresh
        self.min_pts_thresh = min_pts_thresh
        self.mode = mode
        self.density_thresh = density_thresh
        self.concat = concat
        self.means_before = means_before
        self.avg_w_prev = avg_w_prev

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

            if  np.asarray(_reg_p2p.correspondence_set).shape[0] == 0:
                continue

            if _reg_p2p.inlier_rmse < inlier_rmse:
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
        
        pc1.transform(reg_p2p.transformation)
        rotation = reg_p2p.transformation[:-1, :-1]
        translation = reg_p2p.transformation[:-1, -1]
        return np.asarray(pc1.points), np.asarray(pc2.points), reg_p2p.transformation, translation, rotation

    def register(self):
        detections = dict()
        for j, track in enumerate(self.active_tracks.values()):
            dets = copy.deepcopy(track.detections)
            # we start from timestep with most points and then go
            # from max -> end -> start -> max
            densities = [d.pts_density for d in track.detections]
            lwh_diff = [track.detections[i+1].lwh-track.detections[i].lwh \
                        for i in range(len(track.detections)-1)]
            trans_diff = [track.detections[i+1].translation-track.detections[i].translation \
                          for i in range(len(track.detections)-1)]
            
            if self.mode == 'num_interior':
                max_interior_idx = torch.argmax(torch.tensor([d.num_interior \
                                                              for d in track.detections])).item()
            else:
                max_interior_idx = torch.argmax(torch.tensor(densities)).item()

            if len(track) > self.registration_len_thresh and \
                track.min_num_interior >= self.min_pts_thresh and \
                    sum(densities)/len(densities) > self.density_thresh:
                
                # get all timestamps in track
                times = [d.timestamps[0, 0].item() for d in track.detections]

                # get point clouds of all detections and convert to max interior time
                pcs = [track.detections[i].canonical_points for i in range(len(track))] 
                pcs = [track._convert_time(times[i], times[max_interior_idx], self.av2_loader, pcs[i]) \
                       for i in range(len(track))]

                # normalize point clous
                means = [pcs[i].mean(0) for i in range(len(track))]
                pcs = [pcs[i]-means[i] for i in range(len(track))]

                # get trajs of all detections and convert to max interior time
                trajs = [track._get_traj(i=i) for i in range(len(track))]
                trajs = [track._convert_time(times[i], times[max_interior_idx], self.av2_loader, trajs[i]) \
                         for i in range(len(track))]
                
                # iterators from max interior to end and to start
                max_to_end = range(max_interior_idx, len(track)-1)
                max_to_start = range(max_interior_idx, 0, -1)
                iterators = [max_to_end, max_to_start]
                increments = [+1, -1]

                tgt_SE3_src = [None] * len(track)

                # get target boudning box, lwh and alpha for ICP
                pc_tgt = pcs[max_interior_idx]
                lwh_tgt = track.detections[max_interior_idx].lwh
                _, alpha_tgt = get_alpha_rot_t0_to_t1(0, 1, trajs[max_interior_idx])
                for iterator, increment in zip(iterators, increments):
                    # if not registering but just using max interior pc
                    if not self.concat:
                        pc_tgt = pcs[max_interior_idx]
                    for i in iterator:
                        # get source point cloud and alpha
                        pc_increment = pcs[i+increment]
                        _, alpha_src = get_alpha_rot_t0_to_t1(0, 1, trajs[i+increment])

                        # register point clouds
                        pc_increment_registered, pc_tgt, _, translation, rotation = self.icp_p2point(
                            pc_increment,
                            pc_tgt,
                            lwh_tgt,
                            alpha_tgt,
                            alpha_src)
                        tgt_SE3_src[i+increment] = SE3(rotation=rotation, translation=translation)

                        #  concatenate cano points at t+increment and registered points as
                        if self.concat:
                            pc_tgt = np.concatenate([pc_tgt, pc_increment_registered])
                        
                # remove outliers
                if self.concat:
                    pc_tgt = torch.from_numpy(pc_tgt)
                if self.outlier_kNN:
                    pc_tgt, _ = outlier_removal(pc_tgt, threshold=0.5, kNN=10)
                
                pc_tgt = pc_tgt.numpy()

                # get rotation natrix of target point cloud
                rot = torch.tensor([
                    [torch.cos(alpha_tgt), -torch.sin(alpha_tgt), 0],
                    [torch.sin(alpha_tgt), torch.cos(alpha_tgt), 0],
                    [0, 0, 1]]).double()

                # get bounding box with or without added means
                if not self.means_before:
                    lwh, translation_tgt = get_rotated_center_and_lwh(torch.from_numpy(pc_tgt), rot)
                    track.detections[max_interior_idx].translation = translation_tgt + means[max_interior_idx]
                else:
                    lwh, translation_tgt = get_rotated_center_and_lwh(torch.from_numpy(pc_tgt) + means[max_interior_idx], rot)
                    track.detections[max_interior_idx].translation = translation_tgt
                track.detections[max_interior_idx].lwh = lwh

                # propagate bounding box from max to end to start
                max_to_end = range(max_interior_idx, len(track)-1)
                max_to_start = range(max_interior_idx, 0, -1)
                iterators = [max_to_end, max_to_start]
                increments = [+1, -1]
                for iterator, increment in zip(iterators, increments):
                    translation = torch.atleast_2d(translation_tgt)
                    for i in iterator:
                        # transform translation
                        translation = torch.atleast_2d(translation_tgt)
                        translation =  tgt_SE3_src[i+increment].inverse().transform_point_cloud(translation)
                        _translation = track._convert_time(times[max_interior_idx], times[i+increment], self.av2_loader, translation).squeeze()
                        
                        # if taking average of new and current translation store current
                        if self.avg_w_prev:
                            keep = track.detections[i+increment].translation
                        
                        # add means if not before
                        if not self.means_before:
                            track.detections[i+increment].translation = _translation + means[i+increment]
                        else:
                            track.detections[i+increment].translation = _translation
                        
                        # compute average if wanted
                        if self.avg_w_prev:
                            track.detections[i+increment].translation = (keep + track.detections[i+increment].translation) / 2
                        
                        # use lwh of max interior pc
                        track.detections[i+increment].lwh = lwh

                assert len(dets) == len(track.detections)

            detections[j] = track.detections
        return detections

