
import os
from pyarrow import feather
import numpy as np
from collections import defaultdict
import torch
from models.tracking_utils import InitialDetection, load_initial_detections, get_rotated_center_and_lwh, CollapsedDetection, column_names_dets_wo_traj, column_dtypes_dets_wo_traj, _create_box
from pytorch3d.ops import box3d_overlap
import sklearn
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd



def convert_from_t0_to_t1(av2_loader, log_id, t0, t1, points):
    city_SE3_t0 = av2_loader.get_city_SE3_ego(log_id, t0)
    city_SE3_t1 = av2_loader.get_city_SE3_ego(log_id, t1)
    t1_SE3_t0 = city_SE3_t1.inverse().compose(city_SE3_t0)

    return t1_SE3_t0.transform_point_cloud(points)


class Collapser():
    def __init__(self, loader, every_x_frame=1, overlap=25, assign='heuristic', collaps_mode='mean', do_track=False):
        self.loader = loader
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.assign = assign
        self.collaps_mode = collaps_mode
        self.gt_dir = gt_dir
        self.do_track = do_track

    def collaps(self, data, seq, gt_dir):
        """
        assign: heuristic or spectral
        collaps_mode: mean or accumulate
        """
        timestamp_list = sorted([int(p[:-8]) for p in os.listdir(os.path.join(gt_dir, seq, 'sensors', 'lidar'))])
        collapsed_detections = defaultdict(list)
        count = 0
        for k in sorted(data.keys()):
            if count % 50 == 0:
                print(f'{count}/{len(data)}')
            count += 1
            k_idx = timestamp_list.index(k)
            # go only until time = overlap - 2 cos we always need time + 1 for flow
            timestamps = timestamp_list[max(0, k_idx-(self.overlap-2)):k_idx+1]
            if len(timestamps) > 1:
                dts_corners = list()
                trajs_city_t0 = list()
                trajs_city_t1 = list()
                pcs_city = list()
                alphas = list()
                lwhs = list()
                translations = list()
                times = list()
                count = list()
                # get all forward propagated bounding boxes
                for i, time in enumerate(timestamps):
                    if time in data.keys():
                        for j, det in enumerate(data[time]):
                            # transform pc to city
                            city_SE3_t = self.loader.get_city_SE3_ego(seq, time)
                            pc_city = city_SE3_t.transform_point_cloud(
                                det.canonical_points + det.trajectory[:, len(timestamps)-1-i, :])
                            pcs_city.append(pc_city)
                            # transform flow to city
                            traj_city = city_SE3_t.transform_point_cloud(det.trajectory)
                            trajs_city_t0.append(traj_city[:, len(timestamps)-1-i, :])
                            trajs_city_t1.append(traj_city[:, len(timestamps)-i, :])
                            times.append(time)
                            count.append(j)

                            # get rotation, lwh and translation to crete box corners
                            rot, alpha = det.get_alpha_rot_t0_to_t1(
                                len(timestamps)-1-i,
                                len(timestamps)-i,
                                traj_city)
                            lwh, translation = get_rotated_center_and_lwh(pc_city, rot)
                            dts_corners.append(_create_box(translation, lwh, rot))
                            alphas.append(alpha)
                            lwhs.append(lwh)
                            translations.append(translation)

                # num detections in time 
                num_det = len(data[time])
                
                # minus mean to not get numerical instabilities
                dts_corners = torch.stack(dts_corners)
                means = torch.mean(dts_corners.view(
                    dts_corners.shape[0]*dts_corners.shape[1], -1), dim=0)
                dts_corners -= means

                if assign == 'spectral':
                    #### ASSIGN WITH SPECTRAL CLUSTERING
                    # get 3diou
                    _, iou_3d = box3d_overlap(
                        dts_corners.cuda(),
                        dts_corners.cuda(),
                        eps=1e-6)
                    iou_3d = iou_3d.cpu()
                    # get clusters from 3diou
                    sp_clustering = sklearn.cluster.SpectralClustering(n_clusters=len(data[time]), affinity='precomputed').fit(iou_3d)
                    cluster_labels = torch.from_numpy(sp_clustering.labels_)

                    # collaps
                    for l in torch.unique(cluster_labels):
                        idx = torch.where(cluster_labels==l)[0]
                        pc_city = torch.cat([p for i, p in enumerate(pcs_city) if i in idx])
                        traj_t0 = torch.cat([p for i, p in enumerate(trajs_city_t0) if i in idx])
                        traj_t1 = torch.cat([p for i, p in enumerate(trajs_city_t1) if i in idx])
                        # get rotation, lwh and translation to crete box corners
                        rot, alpha = det.get_alpha_rot_t0_to_t1(
                            traj_t0=traj_t0, traj_t1=traj_t1)
                        lwh, translation = get_rotated_center_and_lwh(pc_city, rot)
                        traj = torch.stack([traj_t0, traj_t1])
                        collapsed_detections[k].append(CollapsedDetection(traj, pc_city, time, seq, traj.shape[1], self.overlap))

                elif assign == 'heuristic':
                    #### ASSIGN WITH HEURISTIC
                    # get 3diou
                    pcs_dets = pcs_city[-num_det:]
                    trajs_city_t0_dets = trajs_city_t0[-num_det:]
                    trajs_city_t1_dets = trajs_city_t1[-num_det:]
                    lwhs_dets = lwhs[-num_det:]
                    translations_dets = translations[-num_det:]
                    alphas_dets = alphas[-num_det:]

                    pcs_other = pcs_city[:-num_det]
                    trajs_city_t0_other = trajs_city_t0[:-num_det]
                    trajs_city_t1_other = trajs_city_t1[:-num_det]
                    lwhs_other = lwhs[:-num_det]
                    translations_other = translations[:-num_det]
                    alphas_other = alphas[:-num_det]

                    _, iou_3d = box3d_overlap(
                        dts_corners.cuda()[-num_det:],
                        dts_corners.cuda()[:-num_det],
                        eps=1e-6)
                    
                    arg_max = torch.max(iou_3d, dim=0).indices
                    matches = iou_3d[arg_max.long(), torch.arange(iou_3d.shape[1], dtype=torch.long)]
                    matches = matches > 0.25
                    for i in range(num_det):
                        matched = torch.logical_and(arg_max==i, matches)
                        if torch.where(matched)[0].shape[0] and self.collaps_mode == 'accumulate':
                            collapsed_detections[k].append(accumulate(i,
                                        city_SE3_t,
                                        det,
                                        time,
                                        seq,
                                        self.overlap,
                                        pcs_other,
                                        trajs_city_t0_other,
                                        trajs_city_t1_other,
                                        pcs_dets,
                                        trajs_city_t0_dets,
                                        trajs_city_t1_dets,
                                        matched,
                                        alphas_other,
                                        alphas_dets))
                        elif torch.where(matched)[0].shape[0] and self.collaps_mode == 'mean':
                            collapsed_detections[k].append(self.mean_box(i,
                                            lwhs_dets,
                                            translations_dets,
                                            alphas_dets,
                                            lwhs_other,
                                            translations_other,
                                            alphas_other,
                                            matched,
                                            city_SE3_t,
                                            seq,
                                            time,
                                            self.overlap,
                                            pcs_dets,
                                            trajs_city_t0_other,
                                            trajs_city_t1_other,
                                            trajs_city_t0_dets,
                                            trajs_city_t1_dets,
                                            det))
            else:
                collapsed_detections[k] = [det.final_detection() for det in data[k]]

        return collapsed_detections


    def mean_box(self, i, lwhs_dets, translations_dets, alphas_dets, lwhs_other, translations_other, alphas_other, matched, city_SE3_t, seq, time, overlap, pcs_dets, trajs_city_t0_other, trajs_city_t1_other, trajs_city_t0_dets, trajs_city_t1_dets, det):    
        lwh_city = [lwhs_dets[i]]
        translation_city = [translations_dets[i]]
        alpha_city = [alphas_dets[i]]
        
        lwh_city.extend([lwhs_other[idx] for idx in torch.where(matched)[0]])
        translation_city.extend([translations_other[idx] for idx in torch.where(matched)[0]])
        alpha_city.extend([alphas_other[idx] for idx in torch.where(matched)[0]])

        lwh_city = torch.stack(lwh_city).mean(dim=0)
        translation_city = torch.stack(translation_city).mean(dim=0)
        alpha_city = torch.tensor(alpha_city).mean()
        
        rot = torch.tensor([
            [torch.cos(alpha_city), -torch.sin(alpha_city), 0],
            [torch.sin(alpha_city), torch.cos(alpha_city), 0],
            [0, 0, 1]])
        
        # get 'point cloud' from vertices to get translation and lwh
        vertices = _create_box(translation_city, lwh_city, rot)
        vertices = city_SE3_t.inverse().transform_point_cloud(vertices)

        # rotate rotation into ego vehicle frame --> somehow wrong sign
        '''
        alpha = torch.arccos(torch.from_numpy((city_SE3_t.rotation.T @ rot.numpy()))[0, 0])
        rot = torch.tensor([
                [torch.cos(alpha), -torch.sin(alpha), 0],
                [torch.sin(alpha), torch.cos(alpha), 0],
                [0, 0, 1]]).double()
        '''
        # take mean flow to get rotation
        traj_t0 = [trajs_city_t0_dets[i]]
        traj_t1 = [trajs_city_t1_dets[i]]
        traj_t0.extend([trajs_city_t0_other[idx] for idx in torch.where(matched)[0]])
        traj_t1.extend([trajs_city_t1_other[idx] for idx in torch.where(matched)[0]])
        traj_t0 = torch.cat(traj_t0)
        traj_t1 = torch.cat(traj_t1)
        traj_t0 = city_SE3_t.inverse().transform_point_cloud(traj_t0)
        traj_t1 = city_SE3_t.inverse().transform_point_cloud(traj_t1)
        rot, alpha = det.get_alpha_rot_t0_to_t1(
            traj_t0=traj_t0, traj_t1=traj_t1)
        
        lwh, translation = get_rotated_center_and_lwh(vertices, rot)
        pts_density = (lwh[0] * lwh[1] * lwh[2]) / pcs_dets[i].shape[0]
        return CollapsedDetection(
            rot, alpha, translation, lwh, None, None, time, seq, pcs_dets[i].shape[0], overlap, pts_density)


    def accumulate(self, i, city_SE3_t, det, time, seq, overlap, pcs_other, trajs_city_t0_other, trajs_city_t1_other, pcs_dets, trajs_city_t0_dets, trajs_city_t1_dets, matched, alphas_other, alphas_dets):
        pc_city = [pcs_dets[i]]
        traj_t0 = [trajs_city_t0_dets[i]]
        traj_t1 = [trajs_city_t1_dets[i]]
        alpha = [alphas_dets[i]]

        pc_city.extend([pcs_other[idx] for idx in torch.where(matched)[0]])
        traj_t0.extend([trajs_city_t0_other[idx] for idx in torch.where(matched)[0]])
        traj_t1.extend([trajs_city_t1_other[idx] for idx in torch.where(matched)[0]])
        alpha.extend([alphas_other[idx] for idx in torch.where(matched)[0]])
        print('alpha', alpha)
        pc_city = torch.cat(pc_city)
        traj_t0 = torch.cat(traj_t0)
        traj_t1 = torch.cat(traj_t1)

        pc = city_SE3_t.inverse().transform_point_cloud(pc_city)
        traj_t0 = city_SE3_t.inverse().transform_point_cloud(traj_t0)
        traj_t1 = city_SE3_t.inverse().transform_point_cloud(traj_t1)

        # get rotation, lwh and translation to crete box corners
        rot, alpha = det.get_alpha_rot_t0_to_t1(
            traj_t0=traj_t0, traj_t1=traj_t1)
        print(alpha)
        quit()
        lwh, translation = get_rotated_center_and_lwh(pc, rot)
        traj = torch.stack([traj_t0, traj_t1])
        num_interior = traj.shape[1]
        pts_density = (lwh[0] * lwh[1] * lwh[2]) / num_interior
        return CollapsedDetection(
            rot, alpha, translation, lwh, traj, pc, time, seq, num_interior, overlap, pts_density)



if __name__ == "__main__":
    tracker_dir = "initial_dets"
    gt_dir = "/dvlresearch/jenny/debug_Waymo_Converted_val/Waymo_Converted/val"
    out_path = 'collapsed' 
    split = 'val'
    main(paths=tracker_dir, gt_dir=gt_dir, out_path=out_path, split=split)