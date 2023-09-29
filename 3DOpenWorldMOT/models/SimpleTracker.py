import pytorch3d
from lapsolver import solve_dense
import torch
from .tracking_utils import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class SimpleTracker():
    def __init__(self, every_x_frame, overlap, av2_loader, log_id, rank, logger, a_threshold=0.8, i_threshold=0.8, len_thresh=5):
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
        self.i_threshold = i_threshold
        self.rank = rank
        self.logger = logger
        self.len_thresh = len_thresh
    
    def associate(self, detections):
        # per timestampdetections
        for i, timestamp in enumerate(sorted(detections.keys())):
            dets = detections[timestamp]
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
        self.logger.info(f'TPA: {self.tps}, FNA: {self.fns}, FPS: {self.fps}')
        self.tps = 0
        self.fns = 0
        self.fps = 0
        
        return {t.track_id: t for t in active_tracks if len(t) > self.len_thresh}

    def register(self, tracks):
        pass

    def associate_timestamp(
            self,
            detections,
            matching='hungarian',
            timestamp=None,
            alpha=0.0,
            make_cost_mat_dist=False,
            last=False):
        # add tracks if no tracks yet, initialize tracks
        if not len(self.active_tracks):
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
            [t.detections[-1].gt_id for t in self.inactive_tracks] 
        gt_matches = sum([1 for gt_id_d in gt_id_detections if gt_id_d in gt_id_tracks])
        
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
                gt_id_detections=gt_id_detections)
        
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
            gt_id_detections=None):
        re_activate = list()
        matched_tracks = list()
        matched_dets = list()
        fps = 0
        tps = 0
        rids, cids = solve_dense(cost_mat)
        for rid, cid in zip(rids, cids):
            # check if active or inactive track and get threshold
            act = cid < num_act
            thresh = self.a_threshold if act else self.i_threshold
            # match only of cost smaller than threshold
            if cost_mat[rid, cid] < thresh:
                matched_dets.append(rid)
                # if matched to inactive track, reactivate
                if act:
                    self.active_tracks[cid].add_detection(detections[rid])
                    matched_tracks.append(cid)
                else:
                    inactive_idx = inactive_tracks_to_use[cid-num_act]
                    self.inactive_tracks[inactive_idx].add_detection(detections[rid])
                    re_activate.append(inactive_idx)
                tps = tps + 1 if gt_id_tracks[cid] == gt_id_detections[rid] else tps
                fps = fps + 1 if gt_id_tracks[cid] != gt_id_detections[rid] else fps
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

    def _calculate_traj_dist(self, detections, timestamp, alpha=0.0, dist='cd'):
        # get detections and canonical points of last x frames 
        # and convert to current time for all active tracks
        trajs = [t._get_whole_traj_and_convert_time(
            timestamp, self.av2_loader) for t in self.active_tracks]
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
                    if time_dist < t.detections[-1].length:
                        trajs.extend([
                            t._get_whole_traj_and_convert_time(timestamp, self.av2_loader)])
                        cano_points.extend([
                            t._get_canonical_points_and_convert_time(timestamp, self.av2_loader).float().to(self.rank)])
                        inactive_tracks_to_use.append(i)
                    else:
                        t.dead = True

        # trajectories from canonical point of last added detections
        propagated_pos_tracks = [t.float().to(self.rank) for t in trajs]
        traj_lens = [t.shape[1] for t in propagated_pos_tracks]
        num_tracks = len(traj_lens)

        # trajectories and canonical points of new detections
        det_trajs = [t._get_traj().float().to(self.rank) for t in detections]
        det_cano_points = [t._get_canonical_points().float().to(self.rank) for t in detections]
        '''
        indices_tracks = torch.tensor([
            j for i in range(len(propagated_pos_tracks)) for j in range(propagated_pos_tracks[i].shape[1])])
        indices_time = torch.tensor([
            i for i in range(len(propagated_pos_tracks)) for j in range(propagated_pos_tracks[i].shape[1])])
        propagated_pos_tracks = [
            propagated_pos_tracks[i][:, j, :] for i in range(num_tracks) for j in range(propagated_pos_tracks[i].shape[1])]
        '''
        propagated_pos_dets = [
            torch.tile(det_cano_points[i].unsqueeze(1), (1, det_trajs[i].shape[1], 1)) + det_trajs[i] \
                for i in range(len(det_trajs))]

        # initialize position distances and trajectory distances
        chamferDist = pytorch3d.loss.chamfer_distance
        cd_dists = torch.zeros(len(propagated_pos_dets), num_tracks)
        mean_dist = torch.zeros(len(propagated_pos_dets), num_tracks)
        for k in range(len(propagated_pos_dets)):
            for i in range(num_tracks):
                cd_dist_track = chamferDist(
                        propagated_pos_tracks[i].permute(1, 0, 2), propagated_pos_dets[k][:, :propagated_pos_tracks[i].shape[1]].permute(1, 0, 2))[0]
                # for t in range(propagated_pos_tracks[i].shape[1]):
                #     print(propagated_pos_tracks[i].permute(1, 0, 2)[t].mean(), propagated_pos_dets[k][:, :propagated_pos_tracks[i].shape[1]].permute(1, 0, 2)[t].mean())
                mean_dist[k, i] = self.pdist(propagated_pos_tracks[i].permute(1, 0, 2).mean(dim=1),
                        propagated_pos_dets[k][:, :propagated_pos_tracks[i].shape[1]].permute(1, 0, 2).mean(dim=1)).mean()
                cd_dists[k, i] = cd_dist_track
        if dist == 'cd':
            dists = cd_dists
        else:
            dists = mean_dist
        return dists, \
            len(self.active_tracks), \
                len(trajs) - len(self.active_tracks), \
                    inactive_tracks_to_use
