import logging
import random
import hydra
from omegaconf import OmegaConf
from SeMoLi.trainer import Trainer
import torch
import torch.multiprocessing as mp
import shutil


def sample_params():
    params_list = list()
    for _  in range(30):
        params = {
            'lr': 10 ** random.choice([-3.5, -3, -2.5, -2, -1.5]),
            'weight_decay': 10 ** random.choice([-10, -9.5, -9, -8.5, -8, -7.5]),
            'alpha_node': random.choice([0.7, 0.8, 0.9]),
            'alpha_edge': random.choice([0.7, 0.8, 0.9]),
            'gamma_node': random.choice([1, 1.5, 2, 2.5, 3, 3.5]),
            'gamma_edge': random.choice([1, 1.5, 2, 2.5, 3, 3.5]),
            # 'node_loss': random.choice([True, False]),
            # 'graph_construction': random.choice(['min_mean_max_vel', 'pos', 'postraj', 'traj', 'mean_dist_over_time'])
        }
        dims = sample_dims()
        params['layer_sizes_edge'] = dims
        params['layer_sizes_node'] = dims
        params_list.append(params)

    return params_list


def sample_dims():
    dims = [16, 32, 64, 128]                                                                                                                    
    num_layers =  random.choice([2, ])                                                                                      
    sampled_dims = [random.choice([16, 32, 64])]                                                                                   
    for i in range(1, num_layers):
        if num_layers > 2 and i == 1:
            dim = random.choice(dims[dims.index(sampled_dims[0]):])                                                      
            sampled_dims.append(dim)                                                                                       
        elif num_layers > 3 and i == 2:
            dim = random.choice(dims[dims.index(sampled_dims[1]):])
            sampled_dims.append(dim)
        else:
            dim = random.choice(dims)
            sampled_dims.append(dim)
    dim_dict = dict()
    for i, d in enumerate(sampled_dims):
        dim_dict[f'l_{i}'] = d
            
    return dim_dict


@hydra.main(config_path="../SeMoLi/conf", config_name="conf")   
def main(cfg):
    iters = 1
    if cfg.training.hypersearch:
        params_list = sample_params()
        iters = 30
    
    print(cfg)
    for iter in range(1, iters+1):
        # cfg.models.hyperparams.thresh = rs[iter-1]

        if cfg.training.hypersearch:
            cfg.training.optim.base_lr = params_list[iter]['lr']
            cfg.training.optim.weight_decay = params_list[iter]['weight_decay']
            cfg.models.loss_hyperparams.gamma_node = params_list[iter]['gamma_node']
            cfg.models.loss_hyperparams.gamma_edge = params_list[iter]['gamma_edge']
            cfg.models.loss_hyperparams.alpha_node = params_list[iter]['alpha_node']
            cfg.models.loss_hyperparams.alpha_edge = params_list[iter]['alpha_edge']
            # cfg.models.loss_hyperparams.node_loss = True #params_list[iter]['node_loss']
            cfg.models.hyperparams.layer_sizes_edge = params_list[iter]['layer_sizes_edge']
            cfg.models.hyperparams.layer_sizes_node = params_list[iter]['layer_sizes_node']
            cfg.training.epochs = 15
            # cfg.models.loss_hyperparams.graph_construction = params_list[iter]['graph_construction']
            cfg.data.percentage_data_train = 0.1 
            cfg.data.percentage_data_val = 0.1
            print(f"Current params: {params_list[iter]}")
            cfg.job_name = cfg.job_name + f'_{iter}'

        OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
        print('hi -----------------------------------')
        # needed for preprocessing
        if cfg.training.multi_gpu:
            world_size = torch.cuda.device_count()
            in_args = (cfg, world_size)
            mp.spawn(train, args=in_args, nprocs=world_size, join=True)
        elif torch.cuda.is_available():
            train(0, cfg, world_size=1)
        else:
            train('cpu', cfg, world_size=1)

        logging.shutdown()
    shutil.rmtree(f'{cfg.root_dir}/outputs')

    
def train(rank, cfg, world_size):
    trainer = Trainer(cfg, rank, world_size)
    trainer.train()


if __name__ == "__main__":
    main()
