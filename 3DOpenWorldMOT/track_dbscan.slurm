#!/bin/bash
#SBATCH --job-name=2DBSCANBaselineAgain
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=DEADLINEBIG
#SBATCH --comment="CVPR Cam Ready"
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python trainer.py models=DBSCAN job_name=DBSCAN_WHOLE data=waymo_traj wandb=False training.batch_size=1 just_eval=True models.hyperparams.thresh=1.0 models.hyperparams.input=pos data.debug=False data.trajectory_dir=/storage/user/seidensc/datasets/trajectories_waymo/traj_dataset_waymo/gt_all_egocomp_margin0.6_width25

