import torch
from .tracking_utils import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import copy


class OfflineOracleTracker():
    def __init__(self, every_x_frame, overlap, av2_loader, log_id, rank, logger):
        self.active_tracks = dict()
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.logger = logger
        self.same_id_updater = 10000
        self.same_id_updater_dict = defaultdict(list)
        self.ordered_timestamps = torch.tensor(
            av2_loader.get_ordered_log_lidar_timestamps(log_id))

    def associate(self, detections):
        # per timestampdetections
        for i, timestamp in enumerate(sorted(detections.keys())):
            dets = detections[timestamp]
            try:
                time = timestamp.item()
            except:
                time = timestamp
            gt_ids = np.array([d.gt_id_box for d in dets])
            num_interior = np.array([d.num_interior for d in dets])
            for gt_id in np.unique(gt_ids):
                same_id = np.where(gt_ids==gt_id)[0]
                if same_id.shape[0] == 1:
                    continue
                idxs = np.argsort(num_interior[same_id])[::-1]
                for j, idx in enumerate(idxs[1:]):
                    if len(self.same_id_updater_dict[gt_id]):
                        if len(self.same_id_updater_dict[gt_id]) < j+1:
                            self.same_id_updater_dict[gt_id].append(self.same_id_updater)
                            self.same_id_updater += 1
                    else:
                        self.same_id_updater_dict[gt_id].append(self.same_id_updater)
                        self.same_id_updater += 1
                    dets[same_id[idx]].gt_id_box = self.same_id_updater_dict[gt_id][j]
            # associate
            active_tracks = self.associate_timestamp(
                i,
                dets,
                timestamp=time,
                last=len(detections)==i+1)

        return active_tracks

    def associate_timestamp(self, i, detections, timestamp, last):
        for det in detections:
            if det.gt_id_box not in self.active_tracks or i==1:
                self.active_tracks[det.gt_id_box] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
            else:
                self.active_tracks[det.gt_id_box].add_detection(det)
        return self.active_tracks


class OnlineOracleTracker(OfflineOracleTracker):
    def __init__(self, every_x_frame, overlap, av2_loader, log_id, rank, logger):
        super().__init__(every_x_frame, overlap, av2_loader, log_id, rank, logger)
        self.inactive_tracks = dict()
        self.inactive_match = defaultdict(list)
        self.inactive_match_id = 5000

    def associate_timestamp(self, i, detections, timestamp, last):
        # increase inactive count
        # for idx in self.inactive_tracks.keys():
        #     t0 = torch.where(
        #         self.ordered_timestamps==self.inactive_tracks[
        #             idx].detections[-1].timestamps[0, 0])[0][0].item()
        #     t1 = torch.where(
        #         self.ordered_timestamps==timestamp)[0][0].item()
        #     self.inactive_tracks[idx].inactive_count += t1-t0

        # decide which inactive tracks to use
        inactive_tracks_to_use = dict()
        inactive_tracks_not_to_use = dict()
        if len(self.inactive_tracks):
            for track_id, t in self.inactive_tracks.items():
                # increase inactive count
                t0 = torch.where(
                    self.ordered_timestamps==t.detections[
                        -1].timestamps[0, 0])[0][0].item()
                t1 = torch.where(
                    self.ordered_timestamps==timestamp)[0][0].item()
                # not always 1 cos there can me timestamps wo moving objects
                # < cos for example if len_traj = 2, inactive_count=1, overlap=1
                # then <= will be true but should not cos index starts at 0 not 1
                time_dist = t1-t0
                t.inactive_count += time_dist
                if not t.dead:
                    if t.inactive_count * self.every_x_frame + self.overlap \
                            < t.detections[-1].length:
                        inactive_tracks_to_use[track_id] = t
                    else:
                        t.dead = True
                        inactive_tracks_not_to_use[track_id] = t
                else:
                    inactive_tracks_not_to_use[track_id] = t
        
        # add detections to active / inactive tracks
        tracks_to_use = copy.deepcopy(self.active_tracks)
        tracks_to_use.update(inactive_tracks_to_use)
        active_tracks = dict()
        new_tracks = dict()
        re_activate_tracks = dict()
        for det in detections:
            det_in_act = det.gt_id_box in self.active_tracks
            det_in_dead = det.gt_id_box in inactive_tracks_not_to_use
            det_in_inact = det.gt_id_box in inactive_tracks_to_use
            det_in_match = det.gt_id_box in self.inactive_match
            # if gt not in active or dead
            if (not det_in_act and not det_in_dead and not det_in_inact) or i == 0:
                new_tracks[det.gt_id_box] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
            # if gt not in active but in dead
            elif not det_in_act and det_in_dead:
                # if not in det match
                if not det_in_match:
                    self.inactive_match[det.gt_id_box].append(self.inactive_match_id) # 'm_' + str(self.inactive_match_id))
                    self.inactive_match_id += 1
                    new_tracks[self.inactive_match[det.gt_id_box][-1]] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
                # if in det match add another match
                elif det_in_match and self.inactive_match[det.gt_id_box][-1] in inactive_tracks_not_to_use:
                    self.inactive_match[det.gt_id_box].append(self.inactive_match_id) # 'm_' + str(self.inactive_match_id))
                    self.inactive_match_id += 1
                    new_tracks[self.inactive_match[det.gt_id_box][-1]] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
                # if match in active
                elif self.inactive_match[det.gt_id_box][-1] in self.active_tracks:
                    active_tracks[self.inactive_match[det.gt_id_box][-1]] = self.active_tracks[self.inactive_match[det.gt_id_box][-1]]
                    active_tracks[self.inactive_match[det.gt_id_box][-1]].add_detection(det)
                elif det_in_match and self.inactive_match[det.gt_id_box][-1] in inactive_tracks_to_use:
                    re_activate_tracks[self.inactive_match[det.gt_id_box][-1]] = inactive_tracks_to_use[self.inactive_match[det.gt_id_box][-1]]
                    re_activate_tracks[self.inactive_match[det.gt_id_box][-1]].add_detection(det)
                    re_activate_tracks[self.inactive_match[det.gt_id_box][-1]].inactive_count = 0
            elif det_in_inact:
                re_activate_tracks[det.gt_id_box] = inactive_tracks_to_use[det.gt_id_box]
                re_activate_tracks[det.gt_id_box].add_detection(det)
                re_activate_tracks[det.gt_id_box].inactive_count = 0
                # inactive_tracks_to_use.pop(det.gt_id_box)
            elif det_in_match and self.inactive_match[det.gt_id_box][-1] in inactive_tracks_to_use:
                re_activate_tracks[self.inactive_match[det.gt_id_box][-1]] = \
                        inactive_tracks_to_use[self.inactive_match[det.gt_id_box][-1]]
                re_activate_tracks[self.inactive_match[det.gt_id_box][-1]].add_detection(det)
                re_activate_tracks[self.inactive_match[det.gt_id_box][-1]].inactive_count = 0
                # inactive_tracks_to_use.pop(self.inactive_match[det.gt_id_box][-1])
            elif det_in_act:
                # active_tracks[det.gt_id_box] = self.active_tracks[self.inactive_match[det.gt_id_box][-1]]
                active_tracks[det.gt_id_box] = self.active_tracks[det.gt_id_box]
                active_tracks[det.gt_id_box].add_detection(det)

        # add unused active tracks to inactive
        self.inactive_tracks = inactive_tracks_not_to_use
        self.inactive_tracks.update({k: v for k, v in inactive_tracks_to_use.items() if k not in re_activate_tracks.keys()})
        for k in self.active_tracks:
            if k not in active_tracks.keys():
                self.inactive_tracks[k] = self.active_tracks[k]
        # re-initialize active tracks
        self.active_tracks = active_tracks
        self.active_tracks.update(new_tracks)
        self.active_tracks.update(re_activate_tracks)
        assert len(self.active_tracks) == len(detections)
        # if last add active and inactive tracks
        if last:
            self.active_tracks.update(self.inactive_tracks)
            self.active_tracks = {t_id: t for t_id, t in self.active_tracks.items() if len(t) > 5}     
        return self.active_tracks
