import torch
from .tracking_utils import *
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt


class OfflineOracleTracker():
    def __init__(self, every_x_frame, overlap, av2_loader, log_id, logger):
        self.active_tracks = dict()
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.logger = logger
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
            # associate
            active_tracks = self.associate_timestamp(
                dets,
                timestamp=time,
                last=len(detections)==i+1)

        return active_tracks

    def associate_timestamp(self, detections, timestamp, last):
        for det in detections:
            if det.gt_id_box not in self.active_tracks:
                self.active_tracks[det.gt_id_box] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
            else:
                self.active_tracks[det.gt_id_box].add_detection(det)
        return self.active_tracks


class OnlineOracleTracker(OfflineOracleTracker):
    def __init__(self, every_x_frame, overlap, av2_loader, log_id, logger):
        super().__init__(every_x_frame, overlap, av2_loader, log_id, logger)
        self.inactive_tracks = dict()
        self.inactive_match = defaultdict(list)
        self.inactive_match_id = 0

    def associate_timestamp(self, detections, timestamp, last):
        # increase inactive count
        for idx in range(len(self.inactive_tracks)):
            t0 = torch.where(
                self.ordered_timestamps==self.inactive_tracks[
                    idx].detections[-1].timestamps[0, 0])[0][0].item()
            t1 = torch.where(
                self.ordered_timestamps==timestamp)[0][0].item()
            self.inactive_tracks[idx].inactive_count += t1-t0

        # decide which inactive tracks to use
        inactive_tracks_to_use = dict()
        inactive_tracks_not_to_use = dict()
        if len(self.inactive_tracks):
            for track_id, t in self.inactive_tracks.items():
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
                        inactive_tracks_to_use[track_id] = t
                    else:
                        t.dead = True
                        inactive_tracks_not_to_use[track_id] = t
        
        # add detections to active / inactive tracks
        self.active_tracks.update(inactive_tracks_to_use)
        active_tracks = list()
        for det in detections:
            det_in_act = det.gt_id_box in self.active_tracks
            det_in_dead = det.gt_id_box in inactive_tracks_not_to_use
            det_in_match = det.gt_id_box in self.inactive_match
            # if gt not in active or dead
            if not det_in_act and not det_in_dead:
                active_tracks[det.gt_id_box] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
            # if gt not in active but in dead
            elif not det_in_act and det_in_dead:
                # if not in det match
                if not det_in_match:
                    self.inactive_match[det.gt_id_box].append(self.inactive_match_id)
                    active_tracks[self.inactive_match[det.gt_id_box][-1]] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
                # if in det match add another match
                elif det_in_match and self.inactive_match[det.gt_id_box][-1] in inactive_tracks_not_to_use:
                    self.inactive_match[det.gt_id_box].append(self.inactive_match_id)
                    active_tracks[self.inactive_match[det.gt_id_box][-1]] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
                # if match in active
                elif self.inactive_match[det.gt_id_box][-1] in self.active_tracks:
                    active_tracks[self.inactive_match[det.gt_id_box][-1]] = Track(det, det.gt_id_box, self.every_x_frame, self.overlap)
            else:
                active_tracks[det.gt_id_box] = self.active_tracks[self.inactive_match[det.gt_id_box][-1]]
                active_tracks[det.gt_id_box].add_detection(det)
        
        # add unused active tracks to inactive
        for k in self.active_tracks:
            if k not in active_tracks.keys():
                inactive_tracks_not_to_use[k] = self.active_tracks[k]
        self.inactive_tracks = inactive_tracks_not_to_use

        # re-initialize active tracks
        self.active_tracks = active_tracks

        # if last add active and inactive tracks
        if last:
            self.active_tracks.update(self.inactive_tracks)

        return self.active_tracks