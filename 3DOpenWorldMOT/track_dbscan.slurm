#!/bin/bash
#SBATCH --job-name=DBSCANBaselineAgain
#SBATCH --nodes=1
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python trainer.py models=DBSCAN job_name=DBSCAN models.hyperparams.thresh=2 models.hyperparams.min_samples=5 models.hyperparams.input=traj data=argoverse2_traj data.static_thresh=0.3


