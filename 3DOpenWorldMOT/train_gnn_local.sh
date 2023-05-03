#!/bin/bash
directory_name = "/trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25"

HYDRA_FULL_ERROR=1 python3 trainer.py \
	models=GNN \
	job_name=GNN_RealData \
	data=waymo_traj \
	multi_gpu=False \
	out_path=/dvlresearch/jenny/Documents/3DOpenWorldMOT/3DOpenWorldMOT \
	models.hyperparams.edge_attr=min_mean_max_diffpostrajtime \
	models.hyperparams.node_attr=min_mean_max_vel \
	models.hyperparams.graph_construction=mean_dist_over_time \
	models.hyperparams.k=16 \
	models.hyperparams.k_eval=16 \
	wandb=True \
	models.loss_hyperparams.node_loss=True \
	models.loss_hyperparams.bce_loss=True \
	models.loss_hyperparams.focal_loss_node=False \
	models.loss_hyperparams.focal_loss_edge=False \
	models.loss_hyperparams.node_weight=1 \
	models.loss_hyperparams.edge_weight=1 \
	models.hyperparams.use_node_score=True \
	models.hyperparams.clustering=correlation \
	models.hyperparams.my_graph=True \
	models.hyperparams.graph=radius \
	data.trajectory_dir=/dvlresearch/jenny/debug_trajectories_waymo/traj_dataset_waymo/trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25/trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25 \
	data.processed_dir=/dvlresearch/jenny/debug_trajectories_waymo/processed/normal/trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25/trajectories_removestatic1_numcd44_removegroundrc1_removefar1_removeheight1_len25 \
	data.data_dir=/dvlresearch/jenny/debug_trajectories_waymo/point_clouds_preprocessed \
	data.use_all_points=False \
	data.use_all_points_eval=False \
	data.num_points_eval=8000\
	training.batch_size=32 \
	training.batch_size_val=1 \
	data.debug=True	\
	just_eval=False \
	models.hyperparams.oracle_node=False \
	models.hyperparams.oracle_edge=False \
	models.hyperparams.do_visualize=False \
	tracker_options.num_interior=20 \
	training.eval_per_seq=5 \
	training.optim.base_lr=0.01 \
	lr_scheduler.params.step_size=20 \
	lr_scheduler.params.gamma=0.7 \
	training.optim.optimizer.o_class=Adam \
	data.do_process=True \
	data.static_thresh=0.0


