#!/bin/bash
#SBATCH --job-name=2GTDBSCANIntersection
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=128G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python trainer.py models=DBSCAN_Intersection job_name=DBSCAN_Intersection data=waymo_traj wandb=False  data.debug=False training.batch_size=1 just_eval=True models.hyperparams.visualization=False data.trajectory_dir=/storage/user/seidensc/datasets/trajectories_waymo/traj_dataset_waymo/gt_all_egocomp_margin0.6_width25



