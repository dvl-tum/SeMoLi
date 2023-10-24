import pytorch3d
from lapsolver import solve_dense
import torch
from .tracking_utils import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import torchvision
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from .iou import IoUs2D
import matplotlib


class SimpleTracker():
    def __init__(self, every_x_frame, overlap, av2_loader, log_id, rank, logger, a_threshold=0.8, i_threshold=0.8, len_thresh=5, max_time=5, filter_by_width=False, inact_patience=5, fixed_time=False, l_change_thresh=2, w_change_thresh=2, use_temporal_weight=False):
        self.active_tracks = list()
        self.inactive_tracks = list()
        # self.every_x_frame = every_x_frame
        # self.overlap = overlap
        self.av2_loader = av2_loader
        self.track_id = 0
        self.ordered_timestamps = torch.tensor(
            self.av2_loader.get_ordered_log_lidar_timestamps(log_id))
        self.pdist = torch.nn.PairwiseDistance(p=2)
        self.tps = 0
        self.fns = 0
        self.fps = 0
        self.a_threshold = a_threshold
        self.i_threshold = a_threshold
        self.rank = rank
        self.logger = logger
        self.len_thresh = len_thresh
        self.max_time = max_time
        self.inact_patience = inact_patience
        self.filter_by_width = filter_by_width
        self.fixed_time = fixed_time
        self.l_change_thresh = l_change_thresh
        self.w_change_thresh = w_change_thresh
        self.use_temporal_weight = use_temporal_weight
    
    def associate(self, detections):
        # per timestampdetections
        # 1552440195762604, 1552440196162525, 1552440196062571
        
        for i, timestamp in enumerate(sorted(detections.keys())):
            dets = detections[timestamp]
            # filter by width
            if self.filter_by_width:
                detections_new = list()
                for d in dets:
                    if d.lwh[1] < self.filter_by_width:
                        detections_new.append(d)
                    else:
                        pass #print(d.lwh)
                dets = detections_new
            if len(dets) == 0:
                continue
            try:
                time = timestamp.item()
            except:
                time = timestamp
            # associate
            active_tracks = self.associate_timestamp(
                dets,
                timestamp=time,
                last=len(detections)==i+1)
        
        self.active_tracks += self.inactive_tracks
        if self.tps+self.fps and self.tps+self.fns:
            self.logger.info(f'TPA: {self.tps}, FNA: {self.fns}, FPS: {self.fps}, Pr: {self.tps/(self.tps+self.fps)}, Re: {self.tps/(self.tps+self.fns)}')
        else:
            self.logger.info(f'TPA: {self.tps}, FNA: {self.fns}, FPS: {self.fps}')
        self.tps = 0
        self.fns = 0
        self.fps = 0
        print(f'Before len thresh {len(self.active_tracks)}')
        print([len(t) for t in active_tracks])
        self.active_tracks = {t.track_id: t for t in active_tracks if len(t) > self.len_thresh}
        print(f'After len thresh {len(self.active_tracks)}')
        #for track in self.active_tracks:    
        #    track.fill_detections(self.av2_loader, self.ordered_timestamps.numpy().tolist(), self.max_time)
        
        return self.active_tracks

    def associate_timestamp(
            self,
            detections,
            matching='hungarian',
            timestamp=None,
            alpha=0.0,
            make_cost_mat_dist=False,
            last=False):

        out_of_bounds = False
        if len(self.active_tracks):
            t0 = torch.where(
                self.ordered_timestamps==self.active_tracks[
                    0].detections[-1].timestamps[0, 0])[0][0].item()
            t1 = torch.where(
                self.ordered_timestamps==timestamp)[0][0].item()
            out_of_bounds = t1-t0 > detections[0].length-1
            if out_of_bounds:
                self.inactive_tracks = self.inactive_tracks + self.active_tracks
                for i in range(len(self.inactive_tracks)):
                    self.inactive_tracks[i].inactive_count += t1-t0

        # add tracks if no tracks yet, initialize tracks
        if not len(self.active_tracks) or out_of_bounds:
            self.active_tracks = list()
            for d in detections:
                self.active_tracks.append(Track(d, self.track_id))#, self.every_x_frame, self.overlap))
                self.track_id += 1
            return self.active_tracks
        
        if not len(detections):
            # move all tracks to inactive track
            for tr in self.active_tracks:
                self.inactive_tracks.append(tr)

            # increase inactive count
            for idx in range(len(self.inactive_tracks)):
                t0 = torch.where(
                    self.ordered_timestamps==self.inactive_tracks[
                        idx].detections[-1].timestamps[0, 0])[0][0].item()
                t1 = torch.where(
                    self.ordered_timestamps==timestamp)[0][0].item()
                self.inactive_tracks[idx].inactive_count += t1-t0
            
            # reset active tracks
            self.active_tracks = list()
            return self.active_tracks
        
        # calculate matching costs between detections and tracks
        cost_mat, num_act, num_inact, inactive_tracks_to_use = \
            self._calculate_traj_dist(detections, timestamp)

        gt_id_detections = [d.gt_id for d in detections]
        gt_id_tracks = [t.detections[-1].gt_id for t in self.active_tracks] + \
            [self.inactive_tracks[i].detections[-1].gt_id for i in inactive_tracks_to_use] 
        gt_matches = sum([1 for gt_id_d in gt_id_detections if gt_id_d in gt_id_tracks and gt_id_d != 0])
        gt_cat_detections = [d.gt_cat for d in detections]
        gt_cat_tracks = [t.detections[-1].gt_cat for t in self.active_tracks] + \
            [self.inactive_tracks[i].detections[-1].gt_cat for i in inactive_tracks_to_use]
 
        # match detections and tracks
        if matching == 'greedy':
            re_activate, matched_tracks, matched_dets, tps, fps = self.greedy(
                cost_mat,
                num_act,
                detections,
                inactive_tracks_to_use,
                active_first=True,
                timestamp=timestamp,
                gt_id_tracks=gt_id_tracks,
                gt_id_detections=gt_id_detections)
            
        elif matching == 'hungarian':
            re_activate, matched_tracks, matched_dets, tps, fps = self.hungarian(
                    cost_mat,
                num_act,
                detections,
                inactive_tracks_to_use,
                active_first=True,
                timestamp=timestamp,
                gt_id_tracks=gt_id_tracks,
                gt_id_detections=gt_id_detections,
                gt_cat_tracks=gt_cat_tracks,
                gt_cat_detections=gt_cat_detections)
        
        fns = gt_matches - tps
        self.tps += tps
        self.fps += fps
        self.fns += fns

        # reactivated tracks
        reactivated_tracks = list()
        for r, t in enumerate(self.inactive_tracks):
            if r in re_activate:
                t.inactive_count = 0
                reactivated_tracks.append(t)

        # still inactive
        self.inactive_tracks = [t for r, t in enumerate(self.inactive_tracks)\
            if r not in re_activate]

        # move tracks with no matching to inactive track
        deactivate = list()
        for idx in range(num_act):
            if idx not in matched_tracks:
                deactivate.append(idx)
                self.inactive_tracks.append(self.active_tracks[idx])
        
        # still active
        self.active_tracks = [t for r, t in enumerate(self.active_tracks)\
            if r not in deactivate]

        # add reactivated
        self.active_tracks = self.active_tracks + reactivated_tracks

        # increase inactive count
        for idx in range(len(self.inactive_tracks)):
            # not always 1 cos there can me timestamps wo moving objects
            t0 = torch.where(
                self.ordered_timestamps==self.inactive_tracks[
                    idx].detections[-1].timestamps[0, 0])[0][0].item()
            t1 = torch.where(
                self.ordered_timestamps==timestamp)[0][0].item()
            self.inactive_tracks[idx].inactive_count += t1-t0
        
        # start new tracks
        for idx in range(len(detections)):
            if idx not in matched_dets:
                self.active_tracks.append(Track(
                    detections[idx],
                    self.track_id,
                    # overlap=self.overlap,
                    # every_x_frame=self.every_x_frame
                    ))
                self.track_id += 1

        return self.active_tracks
    
    def hungarian(
            self,
            cost_mat,
            num_act,
            detections,
            inactive_tracks_to_use,
            active_first=False,
            timestamp=None,
            gt_id_tracks=None,
            gt_id_detections=None,
            gt_cat_tracks=None,
            gt_cat_detections=None):
        re_activate = list()
        matched_tracks = list()
        matched_dets = list()
        fps = 0
        tps = 0
        rids, cids = solve_dense(cost_mat[2])
        for rid, cid in zip(rids, cids):
            # check if active or inactive track and get threshold
            act = cid < num_act
            thresh = self.a_threshold if act else self.i_threshold
            # match only of cost smaller than threshold
            if cost_mat[2][rid, cid] < thresh:
                matched_dets.append(rid)
                # if matched to inactive track, reactivate
                if act:
                    self.active_tracks[cid].add_detection(detections[rid])
                    matched_tracks.append(cid)
                else:
                    inactive_idx = inactive_tracks_to_use[cid-num_act]
                    self.inactive_tracks[inactive_idx].add_detection(detections[rid])
                    re_activate.append(inactive_idx)
                # tp only if we are not matching noise objects
                d_fg = gt_id_detections[rid] != 0
                tps = tps + 1 if gt_id_tracks[cid] == gt_id_detections[rid] and d_fg else tps
                fps = fps + 1 if gt_id_tracks[cid] != gt_id_detections[rid] else fps
                # print(gt_id_tracks[cid] == gt_id_detections[rid], gt_id_tracks[cid], gt_id_detections[rid], gt_cat_tracks[cid], gt_cat_detections[rid], cost_mat[0][rid, cid], cost_mat[1][rid, cid], cost_mat[2][rid, cid], cost_mat[3][rid, cid], cost_mat[4][rid, cid])
        
        return re_activate, matched_tracks, matched_dets, tps, fps

    def greedy(
            self,
            cost_mat,
            num_act,
            detections,
            inactive_tracks_to_use,
            active_first=False,
            timestamp=None,
            gt_id_tracks=None,
            gt_id_detections=None):
        
        cost_mat = cost_mat.t

        re_activate = list()
        matched_tracks = list()
        matched_dets = list()
        fps = 0
        tps = 0

        # greedy matching: first match the one with lowest cost
        if active_first:
            act = torch.argsort(torch.min(cost_mat[:num_act], dim=1).values)
            inact = torch.argsort(torch.min(cost_mat[num_act:], dim=1).values) + num_act
            min_idx = torch.cat([act, inact])
        else:
            min_idx = torch.argsort(torch.min(cost_mat, dim=1).values)

        for idx in min_idx:
            # check if active or inactive track and get threshold
            act = idx < num_act
            thresh = self.a_threshold if act else self.i_threshold
            det_traj = torch.argmin(cost_mat[idx])
            # match only of cost smaller than threshold

            if cost_mat[idx, det_traj] < thresh:
                matched_dets.append(det_traj.item())
                # if matched to inactive track, reactivate
                if act: 
                    self.active_tracks[idx].add_detection(detections[det_traj])
                    matched_tracks.append(idx.item())
                else:
                    inactive_idx = inactive_tracks_to_use[idx-num_act]
                    self.inactive_tracks[inactive_idx].add_detection(detections[det_traj])
                    re_activate.append(inactive_idx)
                tps = tps + 1 if gt_id_tracks[idx] == gt_id_detections[det_traj] else tps
                fps = fps + 1 if gt_id_tracks[idx] != gt_id_detections[det_traj] else fps
                # set costs of all other tracks to det to 1000 to not get matches again
                cost_mat[:, det_traj] = 10000

        return re_activate, matched_tracks, matched_dets, tps, fps

    def _calculate_traj_dist(self, detections, timestamp, alpha=0.0, dist='L2Center', visualize=False): # L2Center, 2dIoU
        # get detections and canonical points of last x frames 
        # and convert to current time for all active tracks
        trajs = [t._get_whole_traj_and_convert_time(
            timestamp, self.av2_loader, max_time=self.max_time) for t in self.active_tracks]
        cano_points = [t._get_canonical_points_and_convert_time(
            timestamp, self.av2_loader).float().to(self.rank) for t in self.active_tracks]

        # get detections and canonical points of inactive track if 
        # overlap still predicted in previous added detection
        inactive_tracks_to_use = list()
        if len(self.inactive_tracks):
            for i, t in enumerate(self.inactive_tracks):
                if not t.dead:
                    # not always 1 cos there can me timestamps wo moving objects
                    t0 = torch.where(
                        self.ordered_timestamps==t.detections[
                            -1].timestamps[0, 0])[0][0].item()
                    t1 = torch.where(
                        self.ordered_timestamps==timestamp)[0][0].item()
                    time_dist = t1 - t0
                    # < cos for example if len_traj = 2, inactive_count=1, overlap=1
                    # then <= will be true but should not cos index starts at 0 not 1
                    if self.inact_patience != -1:
                        inact_patience = self.inact_patience
                    else:
                        inact_patience = t.detections[-1].length-1
                    
                    if self.fixed_time:
                        inact_patience -= self.max_time
                        
                    if time_dist < inact_patience:
                        trajs.extend([
                            t._get_whole_traj_and_convert_time(timestamp, self.av2_loader, max_time=self.max_time)])
                        cano_points.extend([
                            t._get_canonical_points_and_convert_time(timestamp, self.av2_loader).float().to(self.rank)])
                        inactive_tracks_to_use.append(i)
                    else:
                        t.dead = True

        # trajectories from canonical point of last added detections
        propagated_pos_tracks = [t.float().to(self.rank) for t in trajs]
        propagared_bbs_tracks = [
            get_propagated_bbs(ppt, _2d=True, get_corners=True) for ppt in propagated_pos_tracks]
        traj_lens = [t.shape[1] for t in propagated_pos_tracks]
        num_tracks = len(traj_lens)

        # trajectories and canonical points of new detections
        det_trajs = [t.trajectory.float().to(self.rank) for t in detections]
        det_cano_points = [t._get_canonical_points().float().to(self.rank) for t in detections]
        propagated_pos_dets = [
            torch.tile(det_cano_points[i].unsqueeze(1), (1, det_trajs[i].shape[1], 1)) + det_trajs[i] \
                for i in range(len(det_trajs))]
        propagared_bbs_dets = [
            get_propagated_bbs(ppd, _2d=True, get_corners=True) for ppd in propagated_pos_dets]

        # initialize position distances and trajectory distances
        chamferDist = pytorch3d.loss.chamfer_distance
        cd_dists = torch.zeros(len(propagated_pos_dets), num_tracks)
        mean_dist = torch.zeros(len(propagated_pos_dets), num_tracks)
        bb_iou_2d = torch.zeros(len(propagated_pos_dets), num_tracks)
        l2_center_2d = torch.zeros(len(propagated_pos_dets), num_tracks)
        l_change_2d = torch.zeros(len(propagated_pos_dets), num_tracks)
        h_change_2d = torch.zeros(len(propagated_pos_dets), num_tracks)
        if visualize:
            os.makedirs('/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/Visualization_tracks', exist_ok=True)
            fig, ax = plt.subplots()
        device = det_trajs[0].device
        for k in range(len(propagated_pos_dets)):
            if visualize:
                self.add_patch(ax, propagared_bbs_dets[k])
            for i in range(num_tracks):
                if visualize:
                    self.add_patch(ax, propagared_bbs_tracks[i], color='red')
                if self.use_temporal_weight:
                    weight = torch.arange(traj_lens[i])/self.use_temporal_weight
                    weight = torch.exp(-weight)/torch.exp(-weight).sum()
                else:
                    weight = torch.ones(traj_lens[i])
                    weight = weight/weight.sum()
                weight = weight.to(device)
                l_change_2d[k, i] = (torch.abs(propagared_bbs_tracks[i]['lwh'][:, 0]-\
                                      propagared_bbs_dets[k]['lwh'][:traj_lens[i], 0]).float()).mean()
                h_change_2d[k, i] = (torch.abs(propagared_bbs_tracks[i]['lwh'][:, 1]-\
                                      propagared_bbs_dets[k]['lwh'][:traj_lens[i], 1]).float()).mean()
                l2_center_2d[k, i] = (weight * self.pdist(propagared_bbs_tracks[i]['translation'], 
                                       propagared_bbs_dets[k]['translation'][:traj_lens[i]])).mean()
                
                # bb_iou_2d[k, i] = 1 - (weight * torch.diagonal(torchvision.ops.box_iou(propagared_bbs_tracks[i]['corners'], 
                #                        propagared_bbs_dets[k]['corners'][:traj_lens[i]]))).mean()
                
                bb_iou_2d[k, i] = 1 - (weight * IoUs2D(propagared_bbs_tracks[i]['xylwa'].unsqueeze(0), 
                                       propagared_bbs_dets[k]['xylwa'][:traj_lens[i]].unsqueeze(0))).sum()
                cd_dist_track = chamferDist(
                        propagated_pos_tracks[i].permute(1, 0, 2), propagated_pos_dets[k][:, :traj_lens[i]].permute(1, 0, 2))[0]
                # for t in range(traj_lens[i]):
                #     print(propagated_pos_tracks[i].permute(1, 0, 2)[t].mean(), propagated_pos_dets[k][:, :traj_lens[i]].permute(1, 0, 2)[t].mean())
                mean_dist[k, i] = (weight * self.pdist(propagated_pos_tracks[i].permute(1, 0, 2).mean(dim=1),
                        propagated_pos_dets[k][:, :traj_lens[i]].permute(1, 0, 2).mean(dim=1))).mean()
                cd_dists[k, i] = cd_dist_track
        if visualize:
            ax.axis('equal')
            plt.savefig(
                f'/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/Visualization_tracks/frame_{timestamp}.jpg', dpi=1000)
            plt.close()

        if dist == 'cd':
            dists = cd_dists
        elif dist == 'mean_point':
            dists = mean_dist
        elif dist == '2dIoU':
            dists = bb_iou_2d
        elif dist == 'L2Center':
            dists = l2_center_2d
        
        l_mask = torch.where(l_change_2d > self.l_change_thresh)
        h_mask = torch.where(h_change_2d > self.w_change_thresh)
        bb_iou_2d[l_mask[0], l_mask[1]] = torch.nan
        bb_iou_2d[h_mask[0], h_mask[1]] = torch.nan
        iou2d_mask = torch.where(bb_iou_2d > self.a_threshold)
        bb_iou_2d[iou2d_mask[0], iou2d_mask[1]] = torch.nan

        return [cd_dists, mean_dist, bb_iou_2d, l2_center_2d, l_change_2d, h_change_2d], \
            len(self.active_tracks), \
                len(trajs) - len(self.active_tracks), \
                    inactive_tracks_to_use

    def add_patch(self, ax, propagared_bbs, color='black'):
        j = 0
        for loc, lwh, rot in zip(propagared_bbs['translation'], propagared_bbs['lwh'], propagared_bbs['alphas']):
            plt.scatter(loc[0].cpu(), loc[1].cpu(), color=color, marker='o', s=2)
            loc_0 = loc-0.5*lwh
            t = matplotlib.transforms.Affine2D().rotate_around(loc[0].cpu(), loc[1].cpu(), rot.cpu()) + ax.transData
            rect = patches.Rectangle(
                loc_0.cpu(),
                lwh[0].cpu(),
                lwh[1].cpu(),
                linewidth=1,
                edgecolor=color,
                facecolor='none',
                transfromation=t)

            ax.add_patch(rect)
