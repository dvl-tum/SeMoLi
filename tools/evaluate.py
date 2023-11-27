from data_utils.splits import get_seq_list_fixed_val
import hydra
from omegaconf import OmegaConf
from evaluation import eval_detection
import os
from PseudoDetection3D.utils.get_name import get_name



@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    seq_list = get_seq_list_fixed_val(
        cfg.data.data_dir,
        detection_set=cfg.data.detection_set,
        percentage=cfg.data.percentage_data_val)
    if cfg.data.debug:
        seq_list = [seq_list[0]]

    name = get_name(cfg) 
    
    if cfg.eval_dir == '':
        out_path = os.path.join(cfg.out_path, 'out/')
        experiment_dir = os.path.join(out_path, f'detections_{cfg.data.detection_set}/')
        detector_dir = os.path.join(experiment_dir + name, cfg.data.detection_set)
    else:
        detector_dir = cfg.eval_dir

    inflate = 'detections_from_pp_sv2_format' not in detector_dir and cfg.inflate_bb 
    print(f'Evaluating detections from {detector_dir} \n \
          1. on only moving objexts {cfg.filter_moving}, \n \
          2. using matched categories {cfg.use_matched_category} \n \
          3. using heuristics {cfg.heuristics} \n \
          4. filtering waymo style {cfg.waymo_style} \n \
          5. inflating bounding boxes {inflate}')

    _, _, _ = eval_detection.eval_detection(
            gt_folder=os.path.join(os.getcwd(), cfg.data.data_dir),
            trackers_folder=detector_dir,
            split='val' if 'evaluation' in cfg.data.detection_set else 'train',
            seq_to_eval=seq_list,
            remove_far=True,
            remove_non_drive='non_drive' in cfg.data.trajectory_dir,
            remove_non_move=cfg.data.remove_static_gt,
            remove_non_move_strategy=cfg.data.remove_static_strategy,
            remove_non_move_thresh=cfg.data.remove_static_thresh,
            visualize=False,
            filter_class=cfg.evaluation.filter_class,
            filter_moving=cfg.evaluation.filter_moving,
            use_matched_category=cfg.evaluation.use_matched_category,
            debug=cfg.data.debug,
            name=name,
            store_matched=cfg.evaluation.store_matched,
            velocity_evaluation=cfg.evaluation.vel_evaluation,
            heuristics=cfg.evaluation.heuristics,
            waymo_style=cfg.evaluation.waymo_style,
            use_aff_as_score=False,
            inflate_bb=cfg.evaluation.inflate_bb,
            min_num_interior_pts=cfg.detector.num_interior,
            store_input_to_eval=cfg.evaluation.store_input_to_eval,
            discard_last_25=cfg.evaluation.discard_last_25,
            inflation_factor=cfg.evaluation.inflation_factor,
            remove_gt_with_pts_leq=cfg.evaluation.remove_gt_with_pts_leq)
    
    print('\n\n')

if __name__ == "__main__":
        main()
