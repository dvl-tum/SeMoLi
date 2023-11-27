def get_name(cfg):
    if 'DBSCAN' not in cfg.models.model_name:
        name = cfg.models.hyperparams.graph_construction 
        name += '_' + cfg.models.hyperparams.edge_attr 
        name += "_" + cfg.models.hyperparams.node_attr
        
        # graph construction
        node = '_NS' if cfg.models.hyperparams.use_node_score else ''
        # cluster = '_' + cfg.models.hyperparams.clustering
        if cfg.models.hyperparams.graph == 'radius':
            graph = f"_{cfg.models.hyperparams.k}_{cfg.models.hyperparams.r}"
        else:
            graph = f"_{cfg.models.hyperparams.k}"
        layer_norm = "_LN_" if cfg.models.hyperparams.layer_norm else ""
        batch_norm = "_BN_" if cfg.models.hyperparams.batch_norm else ""
        drop = "_DR_" if cfg.models.hyperparams.drop_out else ""
        augment = "_AU_" if cfg.models.hyperparams.augment else ""
        name = node + graph + layer_norm + batch_norm + drop + augment + '_' + name
        name = f'{cfg.data.num_points_eval}' + "_" + name if not cfg.data.use_all_points_eval else name
        name = f'{cfg.data.num_points}' + "_" + name if not cfg.data.use_all_points else name

        # optimization
        name = f'{cfg.training.optim.base_lr}' + "_" + name
        name = f'{cfg.training.optim.weight_decay}' + "_" + name
        if cfg.models.loss_hyperparams.focal_loss_node and cfg.models.loss_hyperparams.node_loss:
            name = f'{cfg.models.loss_hyperparams.gamma_node}' + "_" + name
            name = f'{cfg.models.loss_hyperparams.alpha_node}' + "_" + name
        if cfg.models.loss_hyperparams.focal_loss_edge:
            name = f'{cfg.models.loss_hyperparams.gamma_edge}' + "_" + name
            name = f'{cfg.models.loss_hyperparams.alpha_edge}' + "_" + name
        edge_size = '_'.join([str(v) for v in cfg.models.hyperparams.layers_edge.values()])
        name = f'{edge_size}' + "_" + name
        node_size = '_'.join([str(v) for v in cfg.models.hyperparams.layers_node.values()])
        name = f'{node_size}' + "_" + name
        
        # oracles 
        name = 'nooracle' + "_" + name if not cfg.models.hyperparams.oracle_node and \
            not cfg.models.hyperparams.oracle_edge else name
        name = 'oracleedge' + "_" + name if cfg.models.hyperparams.oracle_edge else name
        name = 'oraclenode' + "_" + name if cfg.models.hyperparams.oracle_node else name
        name = 'oraclecluster' + "_" + name if cfg.models.hyperparams.oracle_cluster else name

    elif cfg.models.model_name == 'DBSCAN':
        name = cfg.models.hyperparams.input + "_" + \
            str(cfg.models.hyperparams.thresh) + "_" + \
                str(cfg.models.hyperparams.min_samples)

    elif cfg.models.model_name == 'DBSCAN_Intersection':
        name = cfg.models.hyperparams.input_traj + "_" + \
            str(cfg.models.hyperparams.thresh_traj) + "_" + \
                str(cfg.models.hyperparams.min_samples_traj) + "_" + \
                    str(cfg.models.hyperparams.thresh_pos) + "_" + \
                        str(cfg.models.hyperparams.min_samples_pos) + "_" + \
                            str(cfg.models.hyperparams.flow_thresh)
    
    # dataset
    name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
    name = cfg.job_name + '_' + \
        str(cfg.data.percentage_data_train) + '_' + \
            str(cfg.data.percentage_data_val) + '_' + name
    
    return name