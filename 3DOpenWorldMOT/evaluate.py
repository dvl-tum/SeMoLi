from data_utils.splits import get_seq_list_fixed_val
import hydra
from omegaconf import OmegaConf
from evaluation import eval_detection
import os


def get_name(cfg):
    if 'DBSCAN' not in cfg.models.model_name and cfg.models.model_name != 'SpectralClustering':
        if cfg.models.model_name != 'SimpleGraph':
            node = '_NS' if cfg.models.hyperparams.use_node_score else ''
            # cluster = '_' + cfg.models.hyperparams.clustering
            if cfg.models.hyperparams.graph == 'radius':
                my_graph = f"_MG_{cfg.models.hyperparams.k}_{cfg.models.hyperparams.r}" if cfg.models.hyperparams.my_graph else f'_TG_{cfg.models.hyperparams.k}_{cfg.models.hyperparams.r}'
            else:
                my_graph = f"_MG_{cfg.models.hyperparams.k}" if cfg.models.hyperparams.my_graph else f'_TG_{cfg.models.hyperparams.k}'
            layer_norm = "_LN_" if cfg.models.hyperparams.layer_norm else ""
            batch_norm = "_BN_" if cfg.models.hyperparams.batch_norm else ""
            drop = "_DR_" if cfg.models.hyperparams.drop_out else ""
            augment = "_AU_" if cfg.models.hyperparams.augment else ""

            name = cfg.models.hyperparams.graph_construction + '_' + cfg.models.hyperparams.edge_attr + "_" + cfg.models.hyperparams.node_attr
            name = node + my_graph + layer_norm + batch_norm + drop + augment + '_' + name
            name = f'{cfg.data.num_points_eval}' + "_" + name if not cfg.data.use_all_points_eval else name
            name = f'{cfg.data.num_points}' + "_" + name if not cfg.data.use_all_points else name
            name = f'{cfg.training.optim.base_lr}' + "_" + name
            name = f'{cfg.training.optim.weight_decay}' + "_" + name
            if cfg.models.loss_hyperparams.focal_loss_node:
                name = f'{cfg.models.loss_hyperparams.gamma_node}' + "_" + name
                name = f'{cfg.models.loss_hyperparams.alpha_node}' + "_" + name
            if cfg.models.loss_hyperparams.focal_loss_edge:
                name = f'{cfg.models.loss_hyperparams.gamma_edge}' + "_" + name
                name = f'{cfg.models.loss_hyperparams.alpha_edge}' + "_" + name
            edge_size = '_'.join([str(v) for v in cfg.models.hyperparams.layers_edge.values()])
            name = f'{edge_size}' + "_" + name
            node_size = '_'.join([str(v) for v in cfg.models.hyperparams.layers_node.values()])
            name = f'{node_size}' + "_" + name
            
            name = 'nooracle' + "_" + name if not cfg.models.hyperparams.oracle_node and not cfg.models.hyperparams.oracle_edge else name
            name = 'oracleedge' + "_" + name if cfg.models.hyperparams.oracle_edge else name
            name = 'oraclenode' + "_" + name if cfg.models.hyperparams.oracle_node else name
            name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
            print(f'Using this name: {name}')

    elif cfg.models.model_name == 'DBSCAN':
        name = cfg.models.hyperparams.input + "_" + str(cfg.models.hyperparams.thresh) + "_" + str(cfg.models.hyperparams.min_samples)
        name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
    elif cfg.models.model_name == 'DBSCAN_Intersection':
        name = cfg.models.hyperparams.input_traj + "_" + str(cfg.models.hyperparams.thresh_traj) + "_" + str(cfg.models.hyperparams.min_samples_traj) + "_" + str(cfg.models.hyperparams.thresh_pos) + "_" + str(cfg.models.hyperparams.min_samples_pos) + "_" + str(cfg.models.hyperparams.flow_thresh)
        name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
    
    name = cfg.job_name + '_' + str(cfg.data.percentage_data_train) + '_' + str(cfg.data.percentage_data_val) + '_' + name

    return name

@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    seq_list = get_seq_list_fixed_val(
        cfg.data.data_dir,
        detection_set=cfg.data.detection_set,
        percentage=cfg.data.percentage_data_val)
    if cfg.data.debug:
        seq_list = ['13310437789759009684']
    name = get_name(cfg) 
    if cfg.eval_dir == '':
        out_path = os.path.join(cfg.out_path, 'out/')
        experiment_dir = os.path.join(out_path, f'detections_{cfg.data.detection_set}/')
        detector_dir = os.path.join(experiment_dir + name, cfg.data.detection_set)
    else:
        detector_dir = cfg.eval_dir
    # seq_list = ['1022527355599519580', '10023947602400723454', '10212406498497081993', '10153695247769592104', '1005081002024129653', '10082223140073588526', '10094743350625019937', '10072140764565668044']
    #_seq_list = seq_list
    # for seq in _seq_list:
    #    seq_list = [seq]
    #    print(seq_list)
    #    print(detector_dir, cfg.filter_moving, '\n')
    # evaluate detection
    inflate = 'detections_from_pp_sv2_format' not in detector_dir and cfg.inflate_bb 
    print(f'Evaluating detections from {detector_dir} \n 1. on only moving objexts {cfg.filter_moving}, \n 2. using matched categories {cfg.use_matched_category} \n 3. using heuristics {cfg.heuristics} \n 4. filtering waymo style {cfg.waymo_style} \n 5. inflating bounding boxes {inflate}')

    _, detection_metric, _ = eval_detection.eval_detection(
            gt_folder=os.path.join(os.getcwd(), cfg.data.data_dir),
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
            inflate_bb=cfg.inflate_bb,
            min_num_interior_pts=cfg.detection_options.num_interior,
            store_input_to_eval=cfg.store_input_to_eval,
            discard_last_25=cfg.discard_last_25,
            inflation_factor=cfg.inflation_factor,
            remove_gt_with_pts_leq=cfg.remove_gt_with_pts_leq)
    
    print('\n\n')

if __name__ == "__main__":
        main()
