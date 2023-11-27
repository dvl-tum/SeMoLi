import numpy as np
import warnings
warnings.filterwarnings("ignore")
import hydra
from omegaconf import OmegaConf
from models import _tracker_factory, _registration_factory
import logging
from data_utils import MOT3DTrackDataset, MOT3DSeqDataset, DistributedSeqSampler
from models.tracking_utils import load_initial_detections, store_initial_detections, to_feather
import os
from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from pathlib import Path
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader
from evaluation import eval_detection
import shutil
from data_utils.splits import get_seq_list_fixed_val


logger = logging.getLogger("Model")
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

ch = logging.StreamHandler()
ch.setFormatter(formatter)
logger.addHandler(ch)


class InitialDetProcessor():
    def __init__(self, tracker_type, tracker_params, registration_type,
                 every_x_frame, overlap, av2_loader, rank, track_data_path, initial_dets_path, 
                 tracked_dets_path, registered_dets_path, gt_path, split,
                 detection_set, percentage, a_threshold, i_threshold, len_thresh, outlier_threshold,
                 outlier_kNN, max_time_track, filter_by_width, fixed_time, l_change_thresh,
                 w_change_thresh, inact_patience, use_temporal_weight_track, exp_weight_rot, 
                 registration_len_thresh, min_pts_thresh, mode, density_thresh, concat, means_before, avg_w_prev):
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.av2_loader = av2_loader
        self.rank = rank
        self.tracker_params = tracker_params
        self._tracker = _tracker_factory[tracker_type]
        self._registration = _registration_factory[registration_type]
        self.dataset = MOT3DSeqDataset
        self.initial_dets_path = os.path.join(track_data_path, initial_dets_path)
        self.tracked_dets_path = os.path.join(track_data_path, tracked_dets_path)
        self.registered_dets_path = os.path.join(track_data_path, registered_dets_path)
        self.gt_path = gt_path
        self.split = split
        self.detection_set = detection_set
        self.percentage = percentage
        self.a_threshold = a_threshold
        self.i_threshold = i_threshold
        self.len_thresh = len_thresh
        self.outlier_threshold = outlier_threshold
        self.outlier_kNN = outlier_kNN
        self.max_time_track = max_time_track
        self.filter_by_width = filter_by_width
        self.fixed_time = fixed_time
        self.l_change_thresh = l_change_thresh
        self.w_change_thresh = w_change_thresh
        self.inact_patience = inact_patience
        self.use_temporal_weight_track = use_temporal_weight_track
        self.exp_weight_rot = exp_weight_rot
        self.min_pts_thresh = min_pts_thresh
        self.registration_len_thresh = registration_len_thresh
        self.mode = mode
        self.density_thresh = density_thresh
        self.concat = concat
        self.means_before = means_before
        self.avg_w_prev = avg_w_prev 

    def track(self, log_id, split, detections=None):
        # track detections over sequences
        tracker = self._tracker(
            self.every_x_frame,
            self.overlap,
            self.av2_loader,
            log_id,
            self.rank,
            logger,
            self.a_threshold,
            self.i_threshold,
            self.len_thresh,
            self.max_time_track,
            self.filter_by_width,
            self.inact_patience,
            self.fixed_time,
            self.l_change_thresh,
            self.w_change_thresh,
            self.use_temporal_weight_track)

        if detections is None:
            detections = self.dataset(self.initial_dets_path, self.gt_path, log_id, self.split, self.detection_set).dets
        tracks = tracker.associate(detections)
        p = f'{self.tracked_dets_path}/{self.split}/{log_id}'
        if os.path.isdir(p):
            shutil.rmtree(p)
        store_initial_detections(tracks, log_id, self.tracked_dets_path, split, tracks=True)

        return tracks
    
    def register(self, log_id, tracks=None):
        # register over tracks
        if tracks is None:
            tracks = self.dataset(self.tracked_dets_path, self.gt_path, log_id, self.split, self.detection_set, tracks=True).dets
        tracks = self._registration(
            tracks,
            self.av2_loader,
            log_id,
            self.outlier_threshold,
            self.outlier_kNN,
            self.exp_weight_rot,
            self.registration_len_thresh,
            self.min_pts_thresh,
            self.mode, self.density_thresh, self.concat, self.means_before, self.avg_w_prev).register()
        return tracks
    
    def get_initial_dets(self, log_id):
        # get initial detections
        detections = self.dataset(self.initial_dets_path, self.gt_path, log_id, self.split, self.detection_set).dets
        return detections

    def to_feather(self, detections, log_id, out_path, split):
        # write detections to feather file
        p = f'{out_path}/{self.split}/{log_id}'
        if os.path.isdir(p):
            shutil.rmtree(p)
        to_feather(detections, log_id, out_path, self.split, self.rank, precomp_dets=False)
        write_path = os.path.join(out_path, split, log_id, 'annotations.feather')
        logger.info(f'wrote {write_path}')

    @staticmethod
    def eval(cfg, detector_dir, seq_list, name):
        if cfg.data.debug:
            seq_list = ['10023947602400723454']
        _, detection_metric, all_results_df = eval_detection.eval_detection(
                    gt_folder=os.path.join(os.getcwd(), cfg.data.data_dir + '_train/Waymo_Converted'),
                    trackers_folder=detector_dir,
                    split='val' if 'evaluation' in cfg.data.detection_set else 'train',
                    seq_to_eval=seq_list,
                    remove_far=True,#'80' in cfg.data.trajectory_dir,
                    remove_non_drive='non_drive' in cfg.data.trajectory_dir,
                    remove_non_move=cfg.data.remove_static_gt,
                    remove_non_move_strategy=cfg.data.remove_static_strategy,
                    remove_non_move_thresh=cfg.data.remove_static_thresh,
                    visualize=False,
                    filter_class=cfg.filter_class, #'NO_FILTER', #'CONVERT_ALL_TO_CARS',
                    filter_moving=cfg.filter_moving,
                    use_matched_category=cfg.use_matched_category,
                    debug=cfg.data.debug,
                    name=name,
                    store_matched=cfg.store_matched,
                    velocity_evaluation=cfg.vel_evaluation,
                    heuristics=cfg.heuristics,
                    waymo_style=cfg.waymo_style,
                    use_aff_as_score=False,
                    inflate_bb=cfg.inflate_bb)

        # print(f'Detection metric of detections from detector_dir {detection_metric}')
        return all_results_df

def main(cfg):
    # amodalize
    if cfg.multi_gpu:
        world_size = torch.cuda.device_count()
        in_args = (cfg, world_size)
        mp.spawn(amodalize, args=in_args, nprocs=world_size, join=True)
    else:
        amodalize(1, cfg, world_size=1)
    
    # get sequences to eval
    seq_list = get_seq_list_fixed_val(
        cfg.data.data_dir,
        detection_set=cfg.data.detection_set,
        percentage=cfg.data.percentage_data_val)
    if cfg.data.debug:
        seq_list = [seq_list[0]]

    # evaluate intial detections
    if cfg.registration.convert_initial:
        logger.info('Evaluating initial bounding boxes...')
        InitialDetProcessor.eval(
            cfg,
            os.path.join(cfg.registration.out_path_for_eval, cfg.registration.initial_dets, cfg.registration.split),
            seq_list,
            'Initial')

    # evaluate registered detections
    if cfg.registration.register:
        logger.info('Evaluating registered bounding boxes...')
        InitialDetProcessor.eval(
            cfg,
            os.path.join(cfg.registration.out_path_for_eval, cfg.registration.registered_dets, cfg.registration.split), 
            seq_list,
            'Registered')
        

def amodalize(rank, cfg, world_size):
    if cfg.multi_gpu:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    
    # initialize dataset
    dataset = MOT3DTrackDataset(
        os.path.join(cfg.registration.track_data_path, cfg.registration.initial_dets),
        cfg.data.data_dir,
        cfg.data.detection_set,
        cfg.data.percentage_data_val,
        cfg.data.debug)
    cfg.registration.split = dataset.split

    # get distributed dataloader if multi gpu
    if not cfg.multi_gpu:
        dataloader = DataLoader(
            dataset,
            batch_size=1)
    else:
        sampler = DistributedSeqSampler(
            dataset,
            num_replicas=torch.cuda.device_count(),
            rank=rank)
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler)
    
    # iterate over sequences
    for i, data in enumerate(dataloader):
        seq_name, dataset_path, gt_path, split, detection_set, percentage = data
        seq_name, dataset_path, gt_path, split, detection_set, percentage = seq_name[0], dataset_path[0], gt_path[0], split[0], detection_set[0], percentage[0]
        loader = AV2SensorDataLoader(data_dir=Path(f'{cfg.data.data_dir}_{split}/{split}'), labels_dir=Path(f'{cfg.data.data_dir}_{split}/{split}'))    
        detsprocessor = InitialDetProcessor(
            tracker_type=cfg.registration.tracker_type,
            tracker_params=cfg.registration,
            registration_type=cfg.registration.registration_type,
            every_x_frame=cfg.registration.every_x_frame,
            overlap=cfg.registration.overlap, 
            av2_loader=loader,
            rank=0,
            track_data_path=cfg.registration.track_data_path,
            initial_dets_path=cfg.registration.initial_dets,
            tracked_dets_path=cfg.registration.tracked_dets,
            registered_dets_path=cfg.registration.registered_dets,
            gt_path=f'{gt_path}_{split}/{split}',
            split=split,
            detection_set=detection_set,
            percentage=percentage,
            a_threshold=cfg.registration.a_threshold,
            i_threshold=cfg.registration.i_threshold,
            len_thresh=cfg.registration.len_thresh,
            outlier_threshold=cfg.registration.outlier_threshold,
            outlier_kNN=cfg.registration.outlier_kNN,
            max_time_track=cfg.registration.max_time_track,
            filter_by_width=cfg.registration.filter_by_width,
            fixed_time=cfg.registration.fixed_time,
            l_change_thresh=cfg.registration.l_change_thresh,
            w_change_thresh=cfg.registration.w_change_thresh, 
            inact_patience=cfg.registration.inact_patience,
            use_temporal_weight_track=cfg.registration.use_temporal_weight_track,
            exp_weight_rot=cfg.registration.exp_weight_rot,
            registration_len_thresh=cfg.registration.registration_len_thresh,
            min_pts_thresh=cfg.registration.min_pts_thresh,
            mode=cfg.registration.mode,
            density_thresh=cfg.registration.density_thresh,
            concat=cfg.registration.concat,
            means_before=cfg.registration.means_before,
            avg_w_prev=cfg.registration.avg_w_prev)
        
        detections = None
        tracks = None

        # convert initial detections as comparison
        if cfg.registration.convert_initial:
            logger.info('Converting initial...')
            detections = detsprocessor.get_initial_dets(seq_name, detection_set)
            if not len(detections):
                print(f'No initial detections for sequences {seq_name}')
                continue
            detsprocessor.to_feather(detections, seq_name, os.path.join(cfg.registration.out_path_for_eval, cfg.registration.initial_dets), detection_set)
        
        # track for registration
        if cfg.registration.track:
            logger.info('Tracking...')
            tracks = detsprocessor.track(seq_name, detection_set, detections=detections)
            detections = {k: v.detections for k, v in tracks.items()}
            detsprocessor.to_feather(detections, seq_name, os.path.join(cfg.registration.out_path_for_eval, cfg.registration.tracked_dets), detection_set)
        
        # register over tracks
        if cfg.registration.register:
            logger.info('Registration...')
            detections = detsprocessor.register(seq_name, detection_set, tracks=tracks)
            detsprocessor.to_feather(detections, seq_name, os.path.join(cfg.registration.out_path_for_eval, cfg.registration.registered_dets), detection_set)

