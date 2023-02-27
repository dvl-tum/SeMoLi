#!/bin/bash
#SBATCH --job-name=DBSCANBaselineAgain
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=128G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python trainer.py models=DBSCAN job_name=DBSCAN_WHOLE data=waymo_traj wandb=False training.batch_size=1 just_eval=True models.hyperparams.thresh=0.1 models.hyperparams.input=traj data.debug=False


