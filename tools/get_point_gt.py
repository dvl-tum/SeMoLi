import logging
import random
import hydra
from omegaconf import OmegaConf
from SeMoLi.trainer import Trainer
import torch
import torch.multiprocessing as mp
import shutil


@hydra.main(config_path="../SeMoLi/conf", config_name="conf")   
def main(cfg):
    _ = Trainer(cfg, 0, 1)
    shutil.rmtree(f'{cfg.root_dir}/outputs')


if __name__ == "__main__":
    main()
