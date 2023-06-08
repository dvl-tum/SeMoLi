import torch
from .tracking_utils import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class SimpleTracker():
    def __init__(self, every_x_frame, overlap, av2_loader, log_id, logger, a_threshold=0.8, i_threshold=0.8, rank=0):
        self.active_tracks = list()
        self.inactive_tracks = list()
        self.every_x_frame = every_x_frame
        self.overlap = overlap
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

        return active_tracks

    def associate_timestamp(
            self,
            detections,
            matching='greedy',
            timestamp=None,
            alpha=0.0,
            make_cost_mat_dist=False,
            last=False):
        
        # add tracks if no tracks yet, initialize tracks
        if not len(self.active_tracks):
            self.active_tracks = list()
            for d in detections:
                self.active_tracks.append(Track(d, self.track_id, self.every_x_frame, self.overlap))
                self.track_id += 1
            return
        
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
            return
        
        # calculate matching costs between detections and tracks
        cost_mat, num_act, num_inact, inactive_tracks_to_use = \
            self._calculate_traj_dist(detections, timestamp, matching=matching, alpha=alpha)

        gt_id_detections = [d.gt_id for d in detections]
        gt_id_tracks = [t.detections[-1].gt_id for t in self.active_tracks] + \
            [t.detections[-1].gt_id for t in self.inactive_tracks] 
        gt_matches = sum([1 for gt_id_d in gt_id_detections if gt_id_d in gt_id_tracks])

        if make_cost_mat_dist:
            if matching != 'majority':
                cost_mat = torch.nn.functional.softmax(-cost_mat, dim=1)
            else:
                tr, de, ti = cost_mat.shape
                cost_mat = cost_mat.view(tr, -1)
                cost_mat = torch.nn.functional.softmax(-cost_mat, dim=1)
                cost_mat = cost_mat.view(tr, de, ti)
            cost_mat = 1 - cost_mat
        
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
            
            fns = gt_matches - tps
            self.tps += tps
            self.fps += fps
            self.fns += fns

        elif matching == 'majority':
            re_activate, matched_tracks, matched_dets = \
                self.majority_vote(cost_mat, num_act, detections, inactive_tracks_to_use)
        
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
                    overlap=self.overlap,
                    every_x_frame=self.every_x_frame))
                self.track_id += 1

        if last:
            self.active_tracks += self.inactive_tracks
            self.loggerlogger.info(f'TPA: {self.tps}, FNA: {self.fns}, FPS: {self.fps}')
            self.tps = 0
            self.fns = 0
            self.fps = 0

        return self.active_tracks
    
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

    def majority_vote(self, cost_mat, num_act, detections, inactive_tracks_to_use):
        re_activate = list()
        matched_tracks = list()
        matched_dets = list()
        # majority voting over tracks within overlap
        # cost_mat shape: trajs, dats, time
        # get closest vote (trajs, time) for all trajectories over overlap
        votes = torch.zeros((cost_mat.shape[0], cost_mat.shape[2]))
        for t in range(cost_mat.shape[2]):
            cost = cost_mat[:, :, t]
            vote = torch.argsort(cost, dim=1)
            votes[:, t] = vote[:, 0]

        # get majority counts
        count = list()
        max_vote = list()
        for track_votes in votes.numpy():
            c = np.bincount(track_votes.astype(np.int64))
            count.append(c)
            max_vote.append(np.max(c))

        # assign the one with the 
        for idx in torch.argsort(torch.tensor(max_vote))[::-1]:
            if torch.max(torch.tensor(count[idx])) == 0:
                continue
            det_traj = torch.argmax(torch.tensor(count[idx]))
            act = idx < num_act
            thresh = self.a_threshold if act else self.i_threshold
            if cost_mat[idx, det_traj, :].mean() < thresh:
                matched_dets.append(det_traj)
                if act: 
                    self.active_tracks[idx].add_detection(detections[det_traj])
                    matched_tracks.append(idx)
                else:
                    inactive_idx = inactive_tracks_to_use[idx-num_act]
                    self.inactive_tracks[inactive_idx].add_detection(detections[det_traj])
                    self.inactive_tracks[inactive_idx].inactive_count = 0
                    self.active_tracks.append(self.inactive_tracks[inactive_idx])
                    re_activate.append(inactive_idx)

                for i, c in enumerate(count):
                    if i == idx:
                        continue
                    if c.shape[0] - 1 >= det_traj:
                        c[det_traj] = 0
        
        return re_activate, matched_tracks, matched_dets

    def _calculate_traj_dist(self, detections, timestamp, alpha=0.0, \
                             matching='greedy'):
        # get detections and canonical points of last x frames 
        # and convert to current time for all active tracks
        trajs = [t._get_traj_and_convert_time(
            timestamp, self.av2_loader, overlap=True) for t in self.active_tracks]
        cano_points = [t._get_canonical_points_and_convert_time(
            timestamp, self.av2_loader, overlap=True).float().to(self.rank) for t in self.active_tracks]

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
                    if (t.inactive_count + time_dist) * self.every_x_frame + self.overlap \
                            < t.detections[-1].length:
                        trajs.extend([
                            t._get_traj_and_convert_time(timestamp, self.av2_loader, overlap=True)])
                        cano_points.extend([
                            t._get_canonical_points_and_convert_time(timestamp, self.av2_loader, overlap=True).float().to(self.rank)])
                        inactive_tracks_to_use.append(i)
                    else:
                        t.dead = True

        # trajectories from canonical point of last added detections
        trajs_from_cano = [t[1].float().to(self.rank) for t in trajs]
        # tracejtories from current frame as canonical frame
        trajs_from_overlap = [t[0].float().to(self.rank) for t in trajs]
        
        # trajectories and canonical points of new detections
        det_trajs = [t._get_traj().float().to(self.rank) for t in detections]
        det_cano_points = [t._get_canonical_points().float().to(self.rank) for t in detections]

        # initialize position distances and trajectory distances
        num_time = trajs_from_overlap[0].shape[1]
        if matching != 'majority':
            dists_p = torch.zeros((len(trajs_from_overlap), len(det_trajs)))
        else:
            dists_p = torch.zeros((len(trajs_from_overlap), len(det_trajs), num_time))

        # get trajectory and position distances
        # iterate over trajectories
        for i, (track_traj, track_traj_c, track_p) in enumerate(zip(trajs_from_overlap, trajs_from_cano, cano_points)):

            # iterate over detections
            for j, (det_traj, det_p) in enumerate(zip(det_trajs, det_cano_points)):
                # initialize distances for overlap time
                dists_time_p = torch.zeros(num_time)

                # iterate over time, compute minimum distance from each point in track point
                # cloud to points in detection point cloud and get mean over track points
                for time in range(num_time):
                    # from get positions using previous canonical point at time
                    _traj_p = track_traj_c[:, time, :].unsqueeze(0)

                    # from get positions using current canonical point at time
                    _det_p = det_p + det_traj[:, time, :].unsqueeze(0)
                    dists_time_p[time] = self.pdist(_traj_p.mean(axis=1)[:, :-1].unsqueeze(0), _det_p.mean(axis=1)[:, :-1].unsqueeze(0))[0].cpu()

                # depending on matching strategy get mean over time or not
                if matching != 'majority':
                    dists_p[i, j] = dists_time_p.mean()
                else:
                    dists_p[i, j, :] = dists_time_p

        dists = dists_p

        return dists, \
            len(self.active_tracks), \
                len(trajs) - len(self.active_tracks), \
                    inactive_tracks_to_use
