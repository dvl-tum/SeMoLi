#!/bin/bash
#SBATCH --job-name=TRAINGNN
#SBATCH --nodes=1
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --cpus-per-gpu=1
#SBATCH --mem=64G
#SBATCH --time=120:00:00

source activate sf_av2
HYDRA_FULL_ERROR=1 python trainer.py models=GNN job_name=GNN_WHOLE_TRAIN data=argoverse2_traj models.hyperparams.edge_attr=diffpostraj models.hyperparams.node_attr=postraj wandb=True models.loss_hyperparams.node_loss=True models.hyperparams.use_node_score=True models.hyperparams.clustering=correlation just_eval=False data.debug=False data.short_train=True models.hyperparams.my_graph=False


