#!/bin/bash
#SBATCH --job-name=SimpleGraphBaseline
#SBATCH --nodes=1
#SBATCH --exclude=node14,node15,node16,node17,node18
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python trainer.py models=SimpleGraph job_name=SimpleGraph_TRAJ_POS models.hyperparams.edge_attr=difftraj models.hyperparams.node_attr=traj_pos data=argoverse2_traj models.hyperparams.cut_edges=0.75 data.static_thresh=0.3


