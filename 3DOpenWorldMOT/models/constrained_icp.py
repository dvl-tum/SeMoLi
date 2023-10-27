import open3d as o3
import numpy as np
import torch
import pytorch3d


class ConstrainedICPRegistration():
    def __init__(self, active_tracks, av2_loader, log_id, threshold=0.1, kNN=10, exp_weight_rot=0, registration_len_thresh=4, min_pts_thresh=25):
        self.active_tracks = active_tracks
        self.av2_loader = av2_loader
        self.min_pts_thresh = min_pts_thresh
        self.registration_len_thresh = registration_len_thresh

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

    def icp_p2point(self, pc1, pc2, radius=0.2, its=30, init=None, with_constraint=True):
        pc1, pc2, pc1_centroid = self.load_pointclouds(pc1, pc2)
        if init is None:
            #  init = get_median_init(pc1, pc2)
            init = self.get_centroid_init(pc1, pc2)
        reg_p2p = o3.registration_icp(
            pc1,
            pc2,
            radius,
            init,
            o3.TransformationEstimationPointToPoint(with_constraint=with_constraint, with_scaling=False),
            o3.registration.ICPConvergenceCriteria(max_iteration=its))
        pc1.transform(reg_p2p.transformation)
        rotation = reg_p2p.transformation[:-1, :-1]
        translation = reg_p2p.transformation[:-1, -1]
        return np.asarray(pc1.points), np.asarray(pc2.points), translation, rotation

    def register(self, max_interior_thresh=50):
        detections = dict()
        for j, track in enumerate(self.active_tracks):
            print(len(track), self.registration_len_thresh, self.min_pts_thresh, track.max_num_interior, track.min_num_interior, len(track) > self.registration_len_thresh and track.min_num_interior >= self.min_pts_thresh)
            track_dets = list()
            # we start from timestep with most points and then go
            # from max -> end -> start -> max
            max_interior_idx = torch.argmax(torch.tensor([d.num_interior for d in track.detections])).item()
            max_interior = torch.max(torch.tensor([d.num_interior for d in track.detections])).item()
            max_interior_pc = torch.atleast_2d(track._get_canonical_points(i=max_interior_idx))
            start_in_t0 = max_interior_pc
            if len(track) > self.registration_len_thresh and track.min_num_interior >= self.min_pts_thresh:
                max_to_end = range(max_interior_idx, len(track)-1)
                end_to_start = range(len(track)-1, 0, -1)
                start_to_max = range(0, max_interior_idx)
                iterators = [max_to_end, end_to_start, start_to_max]
                increment = [+1, -1, +1]

                SimilarityTransforms = [None] * len(track)
                for iterator, increment in zip(iterators, increment):
                    start_in_t0 = start_in_t0
                    for i in iterator:
                        # convert pc and trajectory t0 --> t1 from ego frame t=i to ego frame t=i+1
                        t0 = track.detections[i].timestamps[0, 0].item()
                        t1 = track.detections[i+increment].timestamps[0, 0].item()
                        start_in_t1 = track._convert_time(t0, t1, self.av2_loader, start_in_t0)

                        # get points in time t_1
                        cano1_in_t1 = torch.atleast_2d(track._get_canonical_points(i=i+increment))

                        # if track.detections[i+1].num_interior > self.min_pts_thresh:
                        # ICP
                        start_registered, cano1_in_t1, translation, rotation = self.icp_p2point(start_in_t1, cano1_in_t1)
                        track.detections[i+increment].update_after_registration(torch.from_numpy(start_registered), translation, rotation)
                        '''else:
                            traj_in_t0 = torch.atleast_2d(track._get_traj(i=i))
                            t0 = track.detections[i].timestamps[0, 0].item()
                            t1 = track.detections[i+1].timestamps[0, 0].item()
                            dt = self.ordered_timestamps.index(t1) - self.ordered_timestamps.index(t0)
                            translation_1 = start_in_t0 + traj_in_t0[:, dt].mean(dim=0)
                            translation_1 = track._convert_time(t0, t1, self.av2_loader, translation_1).mean(dim=0)
                            translation_2 = torch.atleast_2d(track._get_canonical_points(i=i+1)).mean(dim=0)
                            init_T = (
                                translation_2 - translation_1).float().cuda().unsqueeze(0)
                            init_R = torch.tensor([[[1, 0, 0], [0, 1, 0], [0, 0, 1]]]).float().cuda()
                            init_s = torch.tensor([1]).float().cuda()
                            init = pytorch3d.ops.points_alignment.SimilarityTransform(
                                R=init_R, T=init_T, s=init_s)
                            # if point cloud small just use distance as transform and assume no rotation
                            start_registered = pytorch3d.ops.points_alignment._apply_similarity_transform(
                                    start_in_t1.cuda().float().unsqueeze(0),
                                    init.R,
                                    init.T,
                                    init.s).squeeze()
                            start_registered = torch.atleast_2d(start_in_t0.cpu().squeeze())'''

                        # concatenate cano points at t+increment and registered points as
                        # point cloud for next timestamp / final point cloud
                        start_in_t0 = start_registered # torch.cat([start_registered, cano1_in_t1])
                
        return detections
