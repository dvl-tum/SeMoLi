#!/bin/bash
#SBATCH --job-name=2TRAINGNN
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:48GB
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="CVPR Cam Ready"
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python3 trainer.py \
	models=GNN \
	job_name=GNN_ORACLE_TORCH_GRAPH \
	data=waymo_traj \
	models.hyperparams.edge_attr=min_mean_max_diffpostrajtime \
	models.hyperparams.node_attr=min_mean_max_vel \
	models.hyperparams.graph_construction=pos \
	models.hyperparams.k=32 \
	models.hyperparams.k_eval=32 \
	wandb=True \
	models.loss_hyperparams.node_loss=True \
	models.hyperparams.use_node_score=True \
	models.hyperparams.clustering=correlation \
	models.hyperparams.my_graph=False \
	data.trajectory_dir=/storage/user/seidensc/datasets/trajectories_waymo/traj_dataset_waymo/gt_all_egocomp_margin0.6_width25 \
	data.use_all_points=False \
	data.use_all_points_eval=False\
	data.num_points_eval=32000\
	training.batch_size=8 \
	data.debug=False \
	just_eval=True \
	models.hyperparams.oracle_node=True \
	models.hyperparams.oracle_edge=True






