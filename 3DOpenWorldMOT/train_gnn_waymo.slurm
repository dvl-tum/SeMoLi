#!/bin/bash
#SBATCH --job-name=TRAINGNN
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=40:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python3 trainer.py models=GNN job_name=GNN_WHOLE_TRAIN data=waymo_traj models.hyperparams.edge_attr=difftraj models.hyperparams.node_attr=postraj wandb=True models.loss_hyperparams.node_loss=True models.hyperparams.use_node_score=True models.hyperparams.clustering=correlation models.hyperparams.my_graph=True training.batch_size=1 data.debug=False


