#!/bin/bash
#SBATCH --job-name=TRAINGNN
#SBATCH --nodes=1
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python trainer.py models=GNN job_name=GNN_DEBUG data=argoverse2_traj 

