#!/bin/bash
#SBATCH --job-name=TRAINGNN
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-gpu=20
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python3 trainer.py models=GNN job_name=GNN_WHOLE_TRAIN data=waymo_traj models.hyperparams.edge_attr=diffpostraj models.hyperparams.node_attr=postraj wandb=True models.loss_hyperparams.node_loss=True models.hyperparams.use_node_score=True models.hyperparams.clustering=correlation just_eval=False data.debug=False models.hyperparams.my_graph=True data.trajectory_dir=/storage/user/seidensc/datasets/trajectories_waymo/traj_dataset_waymo/gt_all_egocomp_margin0.6_width25

