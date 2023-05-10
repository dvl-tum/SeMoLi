#!/bin/bash

for lr in 0.1 0.05 0.01 0.005 #0.1
do
	for loss in False True
	do
		ngc batch run --name "GNN_motion_patterns" --priority NORMAL --order 50 --ace nv-us-west-2 --instance dgx1v.32g.8.norm  --result /workspace/result --image "nvidian/dvl/pytorch:chamferdist" --org nvidian --team dvl --datasetid 1604021:/workspace/Waymo_Converted_val --datasetid 1606505:/workspace/all_egocomp_margin0.6_width25_train --datasetid 1605459:/workspace/Waymo_Converted_val_filtered --datasetid 1606413:/workspace/all_egocomp_margin0.6_width25_val --workspace m59iURZnSJmGWVJ5lmqE1Q:/workspace/3DOpenWorldMOT_motion_patterns:RW --label _wl___computer_vision --commandline "cd 3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT; HYDRA_FULL_ERROR=1 python3 trainer.py models=GNN job_name=GNN_RealDataParamSearch_Overfitting_Francesco data=waymo_traj multi_gpu=False out_path=/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT models.hyperparams.edge_attr=min_mean_max_diffpostrajtime models.hyperparams.node_attr=min_mean_max_vel models.hyperparams.graph_construction=mean_dist_over_time models.hyperparams.k=16 models.hyperparams.k_eval=16 wandb=True models.loss_hyperparams.node_loss=True models.loss_hyperparams.bce_loss=True models.loss_hyperparams.focal_loss_node=False models.loss_hyperparams.focal_loss_edge=False models.loss_hyperparams.node_weight=1 models.loss_hyperparams.edge_weight=1 models.hyperparams.use_node_score=True models.hyperparams.clustering=correlation models.hyperparams.my_graph=True models.hyperparams.graph=radius data.trajectory_dir=/workspace/all_egocomp_margin0.6_width25 data.processed_dir=/workspace/all_egocomp_margin0.6_width25 data.data_dir=/workspace/Waymo_Converted_val/Waymo_Converted data.use_all_points=False data.use_all_points_eval=False data.num_points_eval=8000 training.batch_size=16 training.batch_size_val=1 data.debug=True just_eval=False models.hyperparams.oracle_node=False models.hyperparams.oracle_edge=False models.hyperparams.do_visualize=False tracker_options.num_interior=20 training.eval_per_seq=5 training.optim.base_lr=0.01 lr_scheduler.params.step_size=20 lr_scheduler.params.gamma=0.7 training.optim.optimizer.o_class=Adam data.do_process=False data.static_thresh=0.0"
	done
done




