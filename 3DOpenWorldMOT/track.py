
from .models import _tracker_factory, _registration_factory
import logging
from .data_utils import MOT3DTrackDataset
from .models.tracking_utils import load_initial_detections, store_initial_detections, to_feather
import os
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path


logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class InitialDetProcessor():
    def __init__(self, tracker_type, tracker_params, registration_type, 
                 every_x_frame, overlap, av2_loader, rank, initial_dets_path, 
                 collapsed_dets_path, tracked_dets_path, registered_dets_path, gt_path, split):
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.av2_loader = av2_loader
        self.rank = rank
        self.tracker_params = tracker_params
        self._tracker = _tracker_factory[tracker_type]
        self._registration = _registration_factory[registration_type]
        self.dataset = MOT3DTrackDataset
        self.initial_dets_path = initial_dets_path
        self.collapsed_dets_path = collapsed_dets_path
        self.tracked_dets_path = tracked_dets_path
        self.registered_dets_path = registered_dets_path
        self.gt_path = gt_path
        self.split = split

    def track(self, log_id, split):
        tracker = self._tracker(
            self.every_x_frame,
            self.overlap,
            self.av2_loader,
            log_id,
            self.rank,
            logger)

        detections = self.dataset(self.collapsed_dets_path, self.gt_path, log_id, self.split)
        detections = tracker.associate(detections)
        store_initial_detections(detections, log_id, self.tracked_dets_path, split, tracks=True)
    
    def register(self, log_id, split):
        tracks = self.dataset(self.tracked_dets_path, self.gt_path, log_id, tracks=True)
        detections = self._registration(tracks, self.av2_loader)
        store_initial_detections(detections, log_id, self.tracked_dets_path, split, tracks=True)

    def collaps(self, log_id, split):
        detections = self.dataset(self.initial_dets_path, self.gt_path, log_id, self.split)
        detections = self._collaps(detections)
        store_initial_detections(detections, log_id, self.collapsed_dets_path, split, tracks=True)

    def to_feather(self, detections, log_id, out_path, split, rank,):
        detections = [d for track_dets in detections for d in track_dets]
        to_feather(detections, log_id, out_path, split, rank, precomp_dets=False)
        write_path = os.path.join(self.out_path, self.split, 'feathers', f'all_{self.rank}.feather')
        logger.info(f'wrote {write_path}')

if __name__ == "__main__":
    tracker_dir = "initial_dets"
    gt_dir = "/dvlresearch/jenny/debug_Waymo_Converted_val/Waymo_Converted/val"
    out_path = 'collapsed' 
    split = 'val'
    loader = AV2SensorDataLoader(data_dir=Path(gt_dir), labels_dir=Path(gt_dir))
    for i, log_id in enumerate(os.listdir(os.path.join(tracker_dir))):
        detsprocessor = InitialDetProcessor(
            tracker_type='SimpleTracker',
            registration_type='ICP',
            every_x_frame=1,
            overlap=25, 
            av2_loader=loader,
            rank=0,
            initial_dets_path='initial_dets',
            collapsed_dets_path='collapsed_dets',
            tracked_dets_path='tracked_dets',
            registered_dets_path='registered_dets',
            gt_path=gt_dir,
            split=split)