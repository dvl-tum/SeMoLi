#!/bin/bash
#SBATCH --job-name=SimpleGraphBaseline
#SBATCH --nodes=1
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python trainer.py models=SimpleGraph data=waymo_traj wandb=False training.batch_size=1 just_eval=True models.hyperparams.thresh=0.1 models.hyperparams.input=traj data.debug=False


