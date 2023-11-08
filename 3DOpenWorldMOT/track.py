import numpy as np
import warnings
warnings.filterwarnings("ignore")
import hydra
from omegaconf import OmegaConf
from models import _tracker_factory, _registration_factory, _collaps_factory
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
    def __init__(self, tracker_type, tracker_params, registration_type, collaps_type,
                 every_x_frame, overlap, av2_loader, rank, track_data_path, initial_dets_path, 
                 collapsed_dets_path, tracked_dets_path, registered_dets_path, gt_path, split,
                 detection_set, percentage, a_threshold, i_threshold, len_thresh, outlier_threshold,
                 outlier_kNN, max_time_track, filter_by_width, fixed_time, l_change_thresh,
                 w_change_thresh, inact_patience, use_temporal_weight_track, exp_weight_rot, 
                 registration_len_thresh, min_pts_thresh):
        self.every_x_frame = every_x_frame
        self.overlap = overlap
        self.av2_loader = av2_loader
        self.rank = rank
        self.tracker_params = tracker_params
        self._tracker = _tracker_factory[tracker_type]
        self._registration = _registration_factory[registration_type]
        self._collaps = _collaps_factory[collaps_type]
        self.dataset = MOT3DSeqDataset
        self.initial_dets_path = os.path.join(track_data_path, initial_dets_path)
        self.collapsed_dets_path = os.path.join(track_data_path, collapsed_dets_path)
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

    def track(self, log_id, split, detections=None):
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

        # detections = self.dataset(self.collapsed_dets_path, self.gt_path, log_id, self.split).dets
        if detections is None:
            detections = self.dataset(self.initial_dets_path, self.gt_path, log_id, self.split, self.detection_set).dets
        tracks = tracker.associate(detections)
        p = f'{self.tracked_dets_path}/{self.split}/{log_id}'
        if os.path.isdir(p):
            shutil.rmtree(p)
        store_initial_detections(tracks, log_id, self.tracked_dets_path, split, tracks=True)
        return tracks
    
    def register(self, log_id, split, tracks=None):
        if tracks is None:
            tracks = self.dataset(self.tracked_dets_path, self.gt_path, log_id, self.split, self.detection_set, tracks=True).dets
        print(sum([len(t) for t in tracks.values()]))
        tracks = self._registration(
            tracks,
            self.av2_loader,
            log_id,
            self.outlier_threshold,
            self.outlier_kNN,
            self.exp_weight_rot,
            self.registration_len_thresh,
            self.min_pts_thresh).register()
        # p = f'{self.registered_dets_path}/{self.split}/{log_id}'
        # if os.path.isdir(p):
        #     shutil.rmtree(p)
        print(sum([len(t) for t in tracks.values()]))
        # store_initial_detections(tracks, log_id, self.registered_dets_path, split, tracks=False)
        return tracks

    def collaps(self, log_id, split, detections=None):
        if detections is None:
            detections = self.dataset(self.initial_dets_path, self.gt_path, log_id, self.split, self.detection_set).dets
        self.collaps = self._collaps(self.av2_loader)
        detections = self.collaps.collaps(detections, log_id, self.gt_path)
        store_initial_detections(detections, log_id, self.collapsed_dets_path, split, tracks=False, gt_path=self.gt_path)
        return detections
    
    def get_initial_dets(self, log_id, split):
        detections = self.dataset(self.initial_dets_path, self.gt_path, log_id, self.split, self.detection_set).dets
        if False: #self.filter_by_width:
            detections_new = dict()
            for i, timestamp in enumerate(sorted(detections.keys())):
                dets = detections[timestamp]
                # filter by width
                detections_new_time = list()
                for d in dets:
                    if d.lwh[0] < 0.1 or d.lwh[1] < 0.1 or d.lwh[2] < 0.1:
                        continue
                    # d.lwh = d.lwh * 1.25
                    d.lwh[0] = np.clip(d.lwh[0], a_min=1, a_max=None)
                    d.lwh[1] = np.clip(d.lwh[1], a_min=1, a_max=None)
                    d.lwh[2] = np.clip(d.lwh[2], a_min=2, a_max=None)
                    if d.lwh[1] < self.filter_by_width:
                        detections_new_time.append(d)
                if len(detections_new_time):
                    detections_new[timestamp] = detections_new_time
            detections = detections_new
        return detections

    def to_feather(self, detections, log_id, out_path, split):
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


@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    import pandas as pd
    results_df = pd.DataFrame(columns=[
        'dets',
        'thresh',
        'len',
        'max time',
        'inact patience',
        'filter by width',
        'fixed time',
        'pr 0.2', 'pr 0.4', 'pr 0.6', 'pr 0.8',
        'map 0.2', 'map 0.4', 'map 0.6', 'map 0.8',
        'tps 0.2', 'tps 0.4', 'tps 0.6', 'tps 0.8',
        'fps 0.2', 'fps 0.4', 'fps 0.6', 'fps 0.8',
        'fns 0.2', 'fns 0.4', 'fns 0.6', 'fns 0.8',
        'num gt'])
    
    # threshs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    # lens = [2, 4, 6, 8, 10]
    # max_times = [-1, 1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23]
    # inact_patience = [-1, 1, 3, 5, 9, 11, 13, 15, 17, 19, 21, 23]
    # cfg.tracker_options.convert_initial = True
    # for t in inact_patience:
        # cfg.tracker_options.a_threshold = t
        # cfg.tracker_options.i_threshold = t
        # cfg.tracker_options.len_thresh = t
        # cfg.tracker_options.max_time_track = t
        # cfg.tracker_options.inact_patience = t
        # print(f"\n \n {t} \n")

    _params = f'{cfg.tracker_options.filter_by_width}_{cfg.tracker_options.a_threshold}_{cfg.tracker_options.len_thresh}_{cfg.tracker_options.max_time_track}_{cfg.tracker_options.fixed_time}_{cfg.tracker_options.l_change_thresh}_{cfg.tracker_options.w_change_thresh}_{cfg.tracker_options.inact_patience}_{cfg.tracker_options.use_temporal_weight_track}_{cfg.tracker_options.outlier_threshold}_{cfg.tracker_options.outlier_kNN}'
    print(_params)
    
    cfg.tracker_options.initial_dets = f'{cfg.tracker_options.initial_dets}/{cfg.tracker_options.model}'
    cfg.tracker_options.collapsed_dets = f'{cfg.tracker_options.collapsed_dets}/{cfg.tracker_options.model}/{_params}' 
    cfg.tracker_options.tracked_dets = f'{cfg.tracker_options.tracked_dets}/{cfg.tracker_options.model}/{_params}'
    cfg.tracker_options.registered_dets = f'{cfg.tracker_options.registered_dets}/{cfg.tracker_options.model}/{_params}'
    print(cfg.tracker_options.registered_dets)
    
    results_df = _main(cfg, results_df=results_df)
    print('\n\n\n')
    print(results_df)
    # results_df.to_csv('/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT/track_reg_results.csv')

def _main(cfg, results_df):
    #'''
    if cfg.multi_gpu:
        world_size = torch.cuda.device_count()
        in_args = (cfg, world_size)
        mp.spawn(track, args=in_args, nprocs=world_size, join=True)
    else:
        track(1, cfg, world_size=1)
    #'''
    params = [cfg.tracker_options.a_threshold,
              cfg.tracker_options.len_thresh,
              cfg.tracker_options.max_time_track,
              cfg.tracker_options.inact_patience,
              cfg.tracker_options.filter_by_width,
              cfg.tracker_options.fixed_time]
    
    seq_list = get_seq_list_fixed_val(
        cfg.data.data_dir,
        detection_set=cfg.data.detection_set,
        percentage=cfg.data.percentage_data_val)

    if cfg.tracker_options.convert_initial:
        logger.info('Evaluating initial bounding boxes...')
        all_results_df = InitialDetProcessor.eval(
            cfg,
            os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.initial_dets, cfg.tracker_options.split),
            seq_list,
            'Initial')
        results_df.loc[len(results_df.index)] = ['initial'] + params + all_results_df.loc['TYPE_VECHICLE'].values.tolist()

    if cfg.tracker_options.collaps:
        logger.info('Evaluating collapsed bounding boxes...')
        all_results_df = InitialDetProcessor.eval(
            cfg,
            os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.collapsed_dets, cfg.tracker_options.split),
            seq_list,
            'Collapsed')
        results_df.loc[len(results_df.index)] = ['collapsed'] + params + all_results_df.loc['TYPE_VECHICLE'].values.tolist()

    if cfg.tracker_options.track:
        logger.info('Evaluating tracked bounding boxes...')
        all_results_df = InitialDetProcessor.eval(
            cfg,
            os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.tracked_dets, cfg.tracker_options.split),
            seq_list,
            'Tracked')
        results_df.loc[len(results_df.index)] = ['tracked'] + params + all_results_df.loc['TYPE_VECHICLE'].values.tolist()

    if cfg.tracker_options.register:
        logger.info('Evaluating registered bounding boxes...')
        all_results_df = InitialDetProcessor.eval(
            cfg,
            os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.registered_dets, cfg.tracker_options.split), 
            seq_list,
            'Registered')
        results_df.loc[len(results_df.index)] = ['registered'] + params + all_results_df.loc['TYPE_VECHICLE'].values.tolist()
        
    return results_df

def track(rank, cfg, world_size):
    if cfg.multi_gpu:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    dataset = MOT3DTrackDataset(
        os.path.join(cfg.tracker_options.track_data_path, cfg.tracker_options.initial_dets),
        cfg.data.data_dir,
        cfg.data.detection_set,
        cfg.data.percentage_data_val,
        cfg.data.debug)
    cfg.tracker_options.split = dataset.split

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
    
    for i, data in enumerate(dataloader):
        seq_name, dataset_path, gt_path, split, detection_set, percentage = data
        seq_name, dataset_path, gt_path, split, detection_set, percentage = seq_name[0], dataset_path[0], gt_path[0], split[0], detection_set[0], percentage[0]
        # if os.path.isdir(os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.registered_dets, split, seq_name)):
        #     continue
        if '10212406498497081993' not in seq_name:
            continue
        loader = AV2SensorDataLoader(data_dir=Path(f'{cfg.data.data_dir}_{split}/{split}'), labels_dir=Path(f'{cfg.data.data_dir}_{split}/{split}'))    
        detsprocessor = InitialDetProcessor(
            tracker_type=cfg.tracker_options.tracker_type,
            tracker_params=cfg.tracker_options,
            registration_type=cfg.tracker_options.registration_type,
            collaps_type=cfg.tracker_options.collaps_type,
            every_x_frame=cfg.tracker_options.every_x_frame,
            overlap=cfg.tracker_options.overlap, 
            av2_loader=loader,
            rank=0,
            track_data_path=cfg.tracker_options.track_data_path,
            initial_dets_path=cfg.tracker_options.initial_dets,
            collapsed_dets_path=cfg.tracker_options.collapsed_dets,
            tracked_dets_path=cfg.tracker_options.tracked_dets,
            registered_dets_path=cfg.tracker_options.registered_dets,
            gt_path=f'{gt_path}_{split}/{split}',
            split=split,
            detection_set=detection_set,
            percentage=percentage,
            a_threshold=cfg.tracker_options.a_threshold,
            i_threshold=cfg.tracker_options.i_threshold,
            len_thresh=cfg.tracker_options.len_thresh,
            outlier_threshold=cfg.tracker_options.outlier_threshold,
            outlier_kNN=cfg.tracker_options.outlier_kNN,
            max_time_track=cfg.tracker_options.max_time_track,
            filter_by_width=cfg.tracker_options.filter_by_width,
            fixed_time=cfg.tracker_options.fixed_time,
            l_change_thresh=cfg.tracker_options.l_change_thresh,
            w_change_thresh=cfg.tracker_options.w_change_thresh, 
            inact_patience=cfg.tracker_options.inact_patience,
            use_temporal_weight_track=cfg.tracker_options.use_temporal_weight_track,
            exp_weight_rot=cfg.tracker_options.exp_weight_rot,
            registration_len_thresh=cfg.tracker_options.registration_len_thresh,
            min_pts_thresh=cfg.tracker_options.min_pts_thresh)
        detections = None
        tracks = None
        if cfg.tracker_options.convert_initial:
            logger.info('Converting initial...')
            detections = detsprocessor.get_initial_dets(seq_name, detection_set)
            if not len(detections):
                print(f'No initial detections for sequences {seq_name}')
                continue
            detsprocessor.to_feather(detections, seq_name, os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.initial_dets), detection_set)
        if cfg.tracker_options.collaps:
            logger.info('Collapsing...')
            detections = detsprocessor.collaps(seq_name, detection_set, detections)
            detsprocessor.to_feather(detections, seq_name, os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.collapsed_dets), detection_set)
        if cfg.tracker_options.track:
            logger.info('Tracking...')
            tracks = detsprocessor.track(seq_name, detection_set, detections=detections)
            detections = {k: v.detections for k, v in tracks.items()}
            detsprocessor.to_feather(detections, seq_name, os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.tracked_dets), detection_set)
        if cfg.tracker_options.register:
            logger.info('Registration...')
            detections = detsprocessor.register(seq_name, detection_set, tracks=tracks)
            detsprocessor.to_feather(detections, seq_name, os.path.join(cfg.tracker_options.out_path_for_eval, cfg.tracker_options.registered_dets), detection_set)


if __name__ == "__main__":
        main()

        
