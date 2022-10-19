#!/bin/bash
#SBATCH --job-name=DBSCANBaselineAgain
#SBATCH --nodes=1
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate sf_av2
python trainer.py models=SpectralClustering job_name=BASELINE_SPECTRAL models.hyperparams.input=traj

