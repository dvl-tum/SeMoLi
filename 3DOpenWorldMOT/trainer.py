# CONFIGURATION HANDLING
import os
import hydra
from omegaconf import OmegaConf
import torch
import logging
from tqdm import tqdm 

import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
import torch.multiprocessing as mp
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.distributed as dist

from models import _model_factory, _loss_factory, Detector3D
from data_utils.TrajectoryDataset import get_TrajectoryDataLoader
from data_utils.DistributedTestSampler import DistributedTestSampler
from TrackEvalOpenWorld.scripts.run_av2_ow import evaluate_av2_ow_MOT
import wandb

# FOR DETECTION EVALUATION
from evaluation import eval_detection
from evaluation import calc_nmi
from collections import defaultdict
from pyarrow import feather
import shutil
import numpy as np
import random


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def initialize(cfg):
    '''HYPER PARAMETER'''
    #print(cfg.training.gpu)
    #os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training.gpu
    torch.manual_seed(0)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    np.random.seed(0)
    random.seed(0)

    '''CREATE DIR'''
    out_path = os.path.join(cfg.out_path, 'out/')
    os.makedirs(out_path, exist_ok=True)
    experiment_dir = cfg.job_name
    experiment_dir = "_".join([cfg.models.model_name, cfg.data.dataset_name])
    experiment_dir = os.path.join(out_path, experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoints_dir = os.path.join(cfg.out_path, 'checkpoints/')
    os.makedirs(checkpoints_dir, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(cfg)

    return logger, experiment_dir, checkpoints_dir, out_path


def load_model(cfg, checkpoints_dir, logger, rank=0):
    '''MODEL LOADING'''
    model = _model_factory[cfg.models.model_name](rank=rank, **cfg.models.hyperparams)
    print(model)
    criterion = _loss_factory[cfg.models.model_name]
    start_epoch = 0
    optimizer = None

    if 'DBSCAN' not in cfg.models.model_name and cfg.models.model_name != 'SpectralClustering':
        model = model.to(rank)
        criterion = criterion(**cfg.models.loss_hyperparams, rank=rank).to(rank)

        if cfg.models.model_name != 'SimpleGraph':
            node = '_NS' if cfg.models.hyperparams.use_node_score else ''
            # cluster = '_' + cfg.models.hyperparams.clustering
            my_graph = f"_MG_{cfg.models.hyperparams.k}_{cfg.models.hyperparams.r}" if cfg.models.hyperparams.my_graph else f'_TG_{cfg.models.hyperparams.k}_{cfg.models.hyperparams.r}'
            layer_norm = "_LN_" if cfg.models.hyperparams.layer_norm else ""
            batch_norm = "_BN_" if cfg.models.hyperparams.batch_norm else ""
            drop = "_DR_" if cfg.models.hyperparams.drop_out else ""
            augment = "_AU_" if cfg.models.hyperparams.augment else ""

            # name = cfg.models.hyperparams.graph_construction + '_' + cfg.models.hyperparams.edge_attr + "_" + cfg.models.hyperparams.node_attr + node + my_graph # + cluster
            name = node + my_graph + layer_norm + batch_norm + drop + augment # + cluster
            name = f'{cfg.data.num_points_eval}' + "_" + name if not cfg.data.use_all_points_eval else name
            name = f'{cfg.data.num_points}' + "_" + name if not cfg.data.use_all_points else name
            name = f'{cfg.training.optim.base_lr}' + "_" + name
            name = f'{cfg.training.optim.weight_decay}' + "_" + name
            if cfg.models.loss_hyperparams.focal_loss_node:
                name = f'{cfg.models.loss_hyperparams.gamma_node}' + "_" + name
                name = f'{cfg.models.loss_hyperparams.alpha_node}' + "_" + name
            if cfg.models.loss_hyperparams.focal_loss_edge:
                name = f'{cfg.models.loss_hyperparams.gamma_edge}' + "_" + name
                name = f'{cfg.models.loss_hyperparams.alpha_edge}' + "_" + name
            edge_size = '_'.join([str(v) for v in cfg.models.hyperparams.layer_sizes_edge.values()])
            name = f'{edge_size}' + "_" + name
            node_size = '_'.join([str(v) for v in cfg.models.hyperparams.layer_sizes_node.values()])
            name = f'{node_size}' + "_" + name
            
            name = 'nooracle' + "_" + name if not cfg.models.hyperparams.oracle_node and not cfg.models.hyperparams.oracle_edge else name
            name = 'oracleedge' + "_" + name if cfg.models.hyperparams.oracle_edge else name
            name = 'oraclenode' + "_" + name if cfg.models.hyperparams.oracle_node else name
            name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
            
            logger.info(f'Using this name: {name}')
            os.makedirs(checkpoints_dir + name, exist_ok=True)

            try:
                checkpoint = torch.load(cfg.models.weight_path)
                chkpt_new = dict()
                for k, v in checkpoint['model_state_dict'].items():
                    if 'module' in k:
                        chkpt_new[k[7:]] = v
                    else:
                        chkpt_new[k] = v
                checkpoint['model_state_dict'] = chkpt_new
                start_epoch = checkpoint['epoch'] if not cfg.just_eval else start_epoch
                model.load_state_dict(checkpoint['model_state_dict'])
                met = checkpoint['best_metric']
                metric_mode = checkpoint['metric_mode']
                logger.info(f'Use pretrained model with {metric_mode}: {met}')
            except:
                if cfg.models.weight_path != '':
                    logger.info(f'Did not find pretrained model with {cfg.models.weight_path}')
                    quit()
                else:
                    logger.info('No existing model, starting training from scratch...')

            if cfg.training.optim.optimizer.o_class == 'Adam':
                optimizer = torch.optim.Adam(
                    model.parameters(),
                    lr=cfg.training.optim.optimizer.params.lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=cfg.training.optim.weight_decay
                )
            elif cfg.training.optim.optimizer.o_class == 'RAdam':
                optimizer = torch.optim.RAdam(
                    model.parameters(),
                    lr=cfg.training.optim.optimizer.params.lr,
                    betas=(0.9, 0.999),
                    eps=1e-08,
                    weight_decay=cfg.training.optim.weight_decay
                )
            else:
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=cfg.training.optim.optimizer.params.lr,
                    momentum=0.9)
    elif cfg.models.model_name == 'DBSCAN':
        name = cfg.models.hyperparams.input + "_" + str(cfg.models.hyperparams.thresh) + "_" + str(cfg.models.hyperparams.min_samples)
        name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
    elif cfg.models.model_name == 'DBSCAN_Intersection':
        name = cfg.models.hyperparams.input_traj + "_" + str(cfg.models.hyperparams.thresh_traj) + "_" + str(cfg.models.hyperparams.min_samples_traj) + "_" + str(cfg.models.hyperparams.thresh_pos) + "_" + str(cfg.models.hyperparams.min_samples_pos) + "_" + str(cfg.models.hyperparams.flow_thresh)
        name = os.path.basename(cfg.data.trajectory_dir) + "_" + name
    
    if cfg.wandb and (not cfg.multi_gpu or rank == 0):
        wandb.login(key='3b716e6ab76d92ef92724aa37089b074ef19e29c')
        wandb.init(config=cfg, project=cfg.job_name, name=name)

    return model, start_epoch, name, optimizer, criterion


def bn_momentum_adjust(m, momentum):
    if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
        m.momentum = momentum


def sample_params():
    params_list = list()
    for _  in range(30):
        focal_loss = random.choice([True, False])
        params = {
            'lr': 10 ** random.uniform(-4, -1),
            'weight_decay': 10 ** random.uniform(-10, -5),
            'focal_loss_node': focal_loss,
            'focal_loss_edge': focal_loss,
            'alpha_node': random.uniform(0, 1),
            'alpha_edge': random.uniform(0, 1),
            'gamma_node': random.uniform(1, 4),
            'gamma_edge': random.uniform(1, 4),
            'node_loss': random.choice([True, False]),
        }
        dims = sample_dims()
        params['layer_sizes_edge'] = dims
        params['layer_sizes_node'] = dims
        params_list.append(params)

    return params_list

def sample_dims():
    dims = [16, 32, 64, 128]                                                                                                                    
    num_layers =  random.choice([1, 2, 3, 4])                                                                                      
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

@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    iters = 1
    if cfg.training.hypersearch:
        params_list = sample_params()
        iters = 30

    for iter in range(iters):
        if cfg.training.hypersearch:
            cfg.training.optim.base_lr = params_list[iter]['lr']
            cfg.training.optim.weight_decay = params_list[iter]['weight_decay']
            cfg.models.loss_hyperparams.focal_loss_node = params_list[iter]['focal_loss_node']
            cfg.models.loss_hyperparams.focal_loss_edge = params_list[iter]['focal_loss_edge']
            cfg.models.loss_hyperparams.gamma_node = params_list[iter]['gamma_node']
            cfg.models.loss_hyperparams.gamma_edge = params_list[iter]['gamma_edge']
            cfg.models.loss_hyperparams.alpha_node = params_list[iter]['alpha_node']
            cfg.models.loss_hyperparams.alpha_edge = params_list[iter]['alpha_edge']
            cfg.models.loss_hyperparams.node_loss = False #params_list[iter]['node_loss']
            # cfg.models.hyperparams.layer_sizes_edge = params_list[iter]['layer_sizes_edge']
            # cfg.models.hyperparams.layer_sizes_node = params_list[iter]['layer_sizes_node']
            cfg.training.epochs = 30
        
            print(f"Current params: {params_list[iter]}")

        OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly
        logger, experiment_dir, checkpoints_dir, out_path = initialize(cfg)
        
        if os.path.isdir(os.path.join(out_path, 'val', 'feathers')):
            shutil.rmtree(os.path.join(out_path, 'val', 'feathers'))
        
        # needed for preprocessing
        logger.info("start loading training data ...")
        train_data, val_data, test_data = get_TrajectoryDataLoader(cfg)

        if cfg.multi_gpu:
            world_size = torch.cuda.device_count()
            in_args = (cfg, world_size)
            mp.spawn(train, args=in_args, nprocs=world_size, join=True)
        else:
            train(0, cfg, world_size=1)
        
        wandb.finish()
        logging.shutdown()


def train(rank, cfg, world_size):
    is_neural_net = cfg.models.model_name != 'DBSCAN' \
                and cfg.models.model_name != 'SpectralClustering'\
                    and cfg.models.model_name != 'SimpleGraph'\
                    and cfg.models.model_name != 'DBSCAN_Intersection'

    if cfg.multi_gpu:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'
        torch.cuda.set_device(rank)
        dist.init_process_group('nccl', rank=rank, world_size=world_size)
    logger, experiment_dir, checkpoints_dir, out_path = initialize(cfg)
    
    odel, start_epoch, name, optimizer, criterion = \
        load_model(cfg, checkpoints_dir, logger, rank)

    cfg.data.do_process = False
    train_data, val_data, test_data = get_TrajectoryDataLoader(cfg, name)

    # get dataloaders 
    if train_data is not None:
        if not cfg.multi_gpu:
            train_loader = PyGDataLoader(
                train_data,
                batch_size=cfg.training.batch_size,
                drop_last=False,
                shuffle=True)
        else:
            train_sampler = DistributedSampler(
                    train_data,
                    num_replicas=torch.cuda.device_count(),
                    drop_last=False,
                    rank=rank, 
                    shuffle=True)
            train_loader = PyGDataLoader(
                train_data,
                batch_size=cfg.training.batch_size,
                sampler=train_sampler)
    else:
        train_loader = None

    if val_data is not None:
        if not cfg.multi_gpu:
            val_loader = PyGDataLoader(
                val_data,
                batch_size=cfg.training.batch_size_val)
        else:
            val_sampler = DistributedTestSampler(
                    val_data,
                    num_replicas=torch.cuda.device_count(),
                    rank=rank,
                    shuffle=False,
                    drop_last=False)
            val_loader = PyGDataLoader(
                val_data,
                batch_size=cfg.training.batch_size_val,
                sampler=val_sampler)

            '''val_loader = PyGDataLoader(
                val_data,
                batch_size=cfg.training.batch_size_val)'''
    else:
        val_loader = None

    if test_data is not None:
        if not cfg.multi_gpu:
            test_loader = PyGDataLoader(
                test_data,
                batch_size=cfg.training.batch_size_val)
        else:
            test_sampler = DistributedTestSampler(
                    test_data,
                    num_replicas=torch.cuda.device_count(),
                    rank=rank)
            test_loader = PyGDataLoader(
                test_data,
                batch_size=cfg.training.batch_size_val,
                sampler=test_sampler)
    else:
        test_loader = None

    if train_loader is not None:
        logger.info("The number of training data is: %d" % len(train_loader.dataset))
    if val_loader is not None:
        logger.info("The number of test data is: %d" % len(val_loader.dataset))

    model, start_epoch, name, optimizer, criterion = \
        load_model(cfg, checkpoints_dir, logger, rank)

    if cfg.multi_gpu and is_neural_net:
        model = nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)

    is_neural_net = cfg.models.model_name != 'DBSCAN' \
                and cfg.models.model_name != 'SpectralClustering'\
                    and cfg.models.model_name != 'SimpleGraph'\
                    and cfg.models.model_name != 'DBSCAN_Intersection'
    
    best_metric = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        if not cfg.just_eval:
            '''Train on chopped scenes'''
            logger.info('**** Epoch (%d/%s) ****' % (
                 epoch + 1, cfg.training.epochs))

            if is_neural_net:
                # Adapt learning rate
                lr = max(cfg.training.optim.optimizer.params.lr * (
                    cfg.lr_scheduler.params.gamma ** (
                        epoch // cfg.lr_scheduler.params.step_size)), cfg.lr_scheduler.params.clip)
                logger.info('Learning rate:%f' % lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                # Adapt momentum
                '''
                momentum = max(cfg.training.optim.bn_scheduler.params.bn_momentum * (
                    cfg.training.optim.bn_scheduler.params.bn_decay ** (
                        epoch // cfg.training.optim.bn_scheduler.params.decay_step)), \
                            cfg.training.optim.bn_scheduler.params.bn_clip)
                print('BN momentum updated to: %f' % momentum)
                model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
                '''

                # iterate over dataset
                num_batches = len(train_loader)
                model = model.train()
                logger.info('---- EPOCH %03d TRAINING ----' % (epoch + 1))
                node_loss = torch.zeros(2).to(rank)
                node_acc = torch.zeros(2).to(rank)
                edge_loss = torch.zeros(2).to(rank)
                edge_acc = torch.zeros(2).to(rank)
                if not cfg.multi_gpu:
                    num_node_pos = torch.zeros(len(train_loader)).to(rank)
                    num_node_neg = torch.zeros(len(train_loader)).to(rank)
                    num_edge_pos = torch.zeros(len(train_loader)).to(rank)
                    num_edge_neg = torch.zeros(len(train_loader)).to(rank)

                for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
                    data = data.to(rank)
                    optimizer.zero_grad()
                    logits, edge_index, batch_edge = model(data)
                    loss, log_dict, hist_node, hist_edge = criterion(logits, data, edge_index)
                    if cfg.wandb and not cfg.multi_gpu:
                        if hist_node is not None:
                            wandb.log({"train histogram node":
                                wandb.Histogram(np_histogram=hist_node), "epoch": epoch})
                        if hist_edge is not None:
                            wandb.log({"train histogram edge":
                                wandb.Histogram(np_histogram=hist_edge), "epoch": epoch})
                    loss.backward()
                    optimizer.step()

                    if 'train bce loss edge' in log_dict.keys():
                        edge_loss[0] += float(log_dict['train bce loss edge'])
                        edge_loss[1] += 1
                    if 'train bce loss node' in log_dict.keys():
                        node_loss[0] += float(log_dict['train bce loss node'])
                        node_loss[1] += 1
                    if 'train accuracy edge' in log_dict.keys():
                        edge_acc[0] += float(log_dict['train accuracy edge'])
                        edge_acc[1] += 1
                    if 'train accuracy node' in log_dict.keys():
                        node_acc[0] += float(log_dict['train accuracy node'])
                        node_acc[1] += 1
                    if not cfg.multi_gpu:
                        if 'train num node pos' in log_dict.keys():
                            num_node_pos[i] = float(log_dict['train num node pos'])
                        if 'train num node neg' in log_dict.keys():
                            num_node_neg[i] = float(log_dict['train num node neg'])
                        if 'train num edge pos' in log_dict.keys():
                            num_edge_pos[i] = float(log_dict['train num edge pos'])
                        if 'train num edge neg' in log_dict.keys():
                            num_edge_neg[i] = float(log_dict['train num edge neg'])
                    
                if cfg.multi_gpu:
                    dist.all_reduce(node_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(edge_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(edge_acc,   op=dist.ReduceOp.SUM)
                    dist.all_reduce(node_acc, op=dist.ReduceOp.SUM)

                node_loss = float(node_loss[0] / node_loss[1])
                edge_loss = float(edge_loss[0] / edge_loss[1])
                edge_acc = float(edge_acc[0] / edge_acc[1])
                node_acc = float(node_acc[0] / node_acc[1])

                if rank == 0 or not cfg.multi_gpu:
                    logger.info(f'train bce loss edge: {edge_loss}')
                    logger.info(f'train bce loss node: {node_loss}')
                    logger.info(f'train accuracy edge: {edge_acc}')
                    logger.info(f'train accuracy node: {node_acc}')

                    if cfg.wandb:
                        wandb.log({'train bce loss edge': edge_loss, "epoch": epoch})
                        wandb.log({'train bce loss node': node_loss, "epoch": epoch})
                        wandb.log({'train accuracy edge': edge_acc, "epoch": epoch})
                        wandb.log({'train accuracy node': node_acc, "epoch": epoch})
                        if not cfg.multi_gpu:
                            wandb.log({'train num node pos/neg ratio': wandb.Histogram(
                                np_histogram=np.histogram(num_node_pos.cpu().numpy()/num_node_neg.cpu().numpy(), bins=30, range=(0., 3.))), "epoch": epoch})
                            wandb.log({'train num edge pos/neg ratio': wandb.Histogram(
                                np_histogram=np.histogram(num_edge_pos.cpu().numpy()/num_edge_neg.cpu().numpy(), bins=30, range=(0., 3.))), "epoch": epoch})

                    savepath = str(checkpoints_dir) + name + '/latest_model.pth'
                    logger.info(f'Saving at {savepath}...')

                    state = {
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
        # evaluate
        if epoch % cfg.training.eval_per_seq == 0: # and (epoch != 0 or cfg.just_eval):
            do_corr_clustering = epoch % cfg.training.eval_per_seq_corr == 0 and (epoch != 0 or cfg.just_eval)
            num_batches = len(val_loader)
            node_loss = torch.zeros(2).to(rank)
            node_acc = torch.zeros(2).to(rank)
            edge_loss = torch.zeros(2).to(rank)
            edge_acc = torch.zeros(2).to(rank)
            nmis = torch.zeros(2).to(rank)
            if not cfg.multi_gpu:
                num_node_pos = torch.zeros(len(val_loader)).to(rank)
                num_node_neg = torch.zeros(len(val_loader)).to(rank)
                num_edge_pos = torch.zeros(len(val_loader)).to(rank)
                num_edge_neg = torch.zeros(len(val_loader)).to(rank)

            # intialize detector
            if do_corr_clustering:
                detector = Detector3D(
                    out_path + name,
                    split='val',
                    every_x_frame=cfg.data.every_x_frame,
                    num_interior=cfg.detection_options.num_interior,
                    overlap=cfg.detection_options.overlap,
                    av2_loader=val_data.loader,
                    rank=rank,
                    precomp_dets=cfg.detection_options.precomp_dets)

            with torch.no_grad():
                if is_neural_net:
                    model = model.eval()
                logger.info('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
                # Iterate over validation set
                for i, (data) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                    '''# continue if not moving points
                    if data['empty']:
                        continue'''

                    # compute clusters
                    logits, all_clusters, edge_index, _ = model(data, eval=True, name=name, corr_clustering=do_corr_clustering)

                    _nmis = list()
                    batch_idx = data._slice_dict['pc_list']
                    if do_corr_clustering:
                        for g, clusters in enumerate(all_clusters):
                            # continue if we didnt find clusters
                            if not len(clusters):
                                if i+1 == len(val_loader) and g+1 == len(all_clusters):
                                    found = detector.to_feather()
                                    if not found:
                                        logger.info(f'No detections found in {data.log_id[g]}')
                                continue

                            # compute nmi
                            nmi = calc_nmi.calc_normalized_mutual_information(
                                data['point_instances'].cpu()[batch_idx[g]:batch_idx[g+1]], clusters)
                            _nmis.append(nmi)

                            # generate detections
                            detections = detector.get_detections(
                                data.pc_list[batch_idx[g]:batch_idx[g+1]],
                                data.traj[batch_idx[g]:batch_idx[g+1]],
                                clusters,
                                data.timestamps[g].unsqueeze(0),
                                data.log_id[g],
                                data['point_instances'][batch_idx[g]:batch_idx[g+1]],
                                last= i+1 == len(val_loader) and g+1 == len(all_clusters))

                    if is_neural_net and logits[0] is not None:
                        loss, log_dict, hist_node, hist_edge = criterion(logits, data, edge_index, rank, mode='eval')
                        if cfg.wandb and not cfg.multi_gpu:
                            if hist_node is not None:
                                wandb.log({"eval histogram node":
                                    wandb.Histogram(np_histogram=hist_node), "epoch": epoch})
                            if hist_edge is not None:
                                wandb.log({"eval histogram edge":
                                    wandb.Histogram(np_histogram=hist_edge), "epoch": epoch})

                    if do_corr_clustering:
                        nmi = sum(_nmis) / len(_nmis)
                        nmis[0] += float(nmi)
                        nmis[1] += 1

                    if is_neural_net and logits[0] is not None:
                        if 'eval bce loss edge' in log_dict.keys():
                            edge_loss[0] += float(
                                log_dict['eval bce loss edge'])
                            edge_loss[1] += 1
                        if 'eval bce loss node' in log_dict.keys():
                            node_loss[0] += float(
                                log_dict['eval bce loss node'])
                            node_loss[1] += 1
                        if 'eval accuracy edge' in log_dict.keys():
                            edge_acc[0] += float(
                                log_dict['eval accuracy edge'])
                            edge_acc[1] += 1
                        if 'eval accuracy node' in log_dict.keys():
                            node_acc[0] += float(
                                log_dict['eval accuracy node'])
                            node_acc[1] += 1
                        if not cfg.multi_gpu:
                            if 'edge num node pos' in log_dict.keys():
                                num_node_pos[i] += float(
                                    log_dict['edge num node pos'])
                            if 'edge num node neg' in log_dict.keys():
                                num_node_neg[i] += float(
                                    log_dict['edge num node neg'])
                            if 'edge num edge pos' in log_dict.keys():
                                num_edge_pos[i] += float(
                                    log_dict['edge num edge pos'])
                            if 'edge num edge neg' in log_dict.keys():
                                num_edge_neg[i] += float(
                                    log_dict['edge num edge neg'])

                if is_neural_net:
                    model = model.train()
                if cfg.multi_gpu:
                    if do_corr_clustering:
                        dist.all_reduce(nmis, op=dist.ReduceOp.SUM)
                    if is_neural_net:
                        dist.all_reduce(node_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(edge_loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(edge_acc, op=dist.ReduceOp.SUM)
                        dist.all_reduce(node_acc, op=dist.ReduceOp.SUM)

                if do_corr_clustering:
                    nmis = float(nmis[0] / nmis[1])
                if is_neural_net:
                    node_loss = float(node_loss[0] / node_loss[1])
                    edge_loss = float(edge_loss[0] / edge_loss[1])
                    edge_acc = float(edge_acc[0] / edge_acc[1])
                    node_acc = float(node_acc[0] / node_acc[1])

                if rank == 0 or not cfg.multi_gpu:
                    if do_corr_clustering:
                        logger.info(f'nmi: {nmis}')
                    if is_neural_net:
                        logger.info(f'eval bce loss edge: {edge_loss}')
                        logger.info(f'eval bce loss node: {node_loss}')
                        logger.info(f'eval accuracy edge: {edge_acc}')
                        logger.info(f'eval accuracy node: {node_acc}')

                    if cfg.wandb and is_neural_net:
                        wandb.log({'eval bce loss edge': edge_loss, "epoch": epoch})
                        wandb.log({'eval bce loss node': node_loss, "epoch": epoch})
                        wandb.log({'eval accuracy edge': edge_acc, "epoch": epoch})
                        wandb.log({'eval accuracy node': node_acc, "epoch": epoch})
                        if not cfg.multi_gpu:
                            wandb.log({'eval num node pos/neg ratio': wandb.Histogram(
                                np_histogram=np.histogram(num_node_pos.cpu().numpy()/num_node_neg.cpu().numpy(), bins=30, range=(0., 3.))), "epoch": epoch})
                            wandb.log({'eval num edge pos/neg ratio': wandb.Histogram(
                                np_histogram=np.histogram(num_edge_pos.cpu().numpy()/num_edge_neg.cpu().numpy(), bins=30, range=(0., 3.))), "epoch": epoch})

                    if do_corr_clustering:
                        # get sequence list for evaluation
                        detector_dir = os.path.join(detector.out_path, detector.split)
                        if os.path.isdir(os.path.join(detector_dir, 'feathers')):
                            for i, rank_file in enumerate(os.listdir(os.path.join(detector_dir, 'feathers'))):
                                # with open(os.path.join(detector_dir, 'feathers', rank_file), 'rb') as f:
                                df = feather.read_feather(os.path.join(detector_dir, 'feathers', rank_file))
                                if i == 0:
                                    df_all = df
                                else:
                                    df_all = df_all.append(df)
                            
                            for log_id in os.listdir(detector_dir):
                                if 'feather' in log_id:
                                    continue
                                df_seq = df_all[df_all['log_id']==log_id]
                                df_seq = df_seq.drop(columns=['log_id'])
                                write_path = os.path.join(detector_dir, log_id, 'annotations.feather')
                                with open(write_path, 'wb') as f:
                                    feather.write_feather(df_seq, f)
                            
                            if os.path.isdir(os.path.join(detector_dir, 'feathers')):
                                shutil.rmtree(os.path.join(detector_dir, 'feathers'))
                    
                        try:
                            seq_list = os.listdir(detector_dir)
                        except:
                            seq_list = list()
                    
                    if do_corr_clustering:
                        # average NMI
                        cluster_metric = [nmis]
                        logger.info(f'NMI: {cluster_metric[0]}')
                        # evaluate detection
                        _, detection_metric = eval_detection.eval_detection(
                            gt_folder=os.path.join(os.getcwd(), cfg.data.data_dir),
                            trackers_folder=detector_dir,
                            seq_to_eval=seq_list,
                            remove_far=True,#'80' in cfg.data.trajectory_dir,
                            remove_non_drive='non_drive' in cfg.data.trajectory_dir,
                            remove_non_move=cfg.data.remove_static_gt,
                            remove_non_move_strategy=cfg.data.remove_static_strategy,
                            remove_non_move_thresh=cfg.data.remove_static_thresh,
                            classes_to_eval='all',
                            debug=cfg.data.debug,
                            name=name)
                    
                        # log metrics
                        for_logs = {met: m for met, m in zip(['AP', 'ATE', 'ASE', 'AOE' ,'CDS'], detection_metric)}
                    
                        if cfg.wandb:
                            for met, m in for_logs.items():
                                wandb.log({met: m, "epoch": epoch})
                            wandb.log({'NMI': cluster_metric[0], "epoch": epoch})
                    
                    
                    if not is_neural_net and cfg.metric == 'acc':
                        metric = detection_metric
                    elif cfg.metric == 'acc':
                        metric = [edge_acc]
                    elif cfg.metric == 'cluster':
                        metric = cluster_metric
                    else:
                        metric = detection_metric

                    # store weights if neural net                
                    print(metric[0], best_metric)
                    if metric[0] >= best_metric:
                        best_metric = metric[0]
                        if is_neural_net and not cfg.just_eval:
                            savepath = str(checkpoints_dir) + name + '/best_model.pth'
                            logger.info('Saving at %s...' % savepath)
                            if do_corr_clustering:
                                state = {
                                    'epoch': epoch,
                                    'ACC': edge_acc,
                                    'best_metric': best_metric,
                                    'metric_mode': cfg.metric,
                                    'NMI': cluster_metric[0],
                                    'class_avg_iou': detection_metric[0],
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }
                            else:
                                state = {
                                    'epoch': epoch,
                                    'ACC': edge_acc,
                                    'best_metric': best_metric,
                                    'metric_mode': cfg.metric,
                                    'model_state_dict': model.state_dict(),
                                    'optimizer_state_dict': optimizer.state_dict(),
                                }
                            torch.save(state, savepath)
                    
                    if do_corr_clustering:
                        logger.info(f'Best {cfg.metric} metric: {best_metric}, acc: {edge_acc}, cluster metric: {cluster_metric}, detection metric: {for_logs}')
                    else:
                        logger.info(f'Best {cfg.metric} metric: {best_metric}, acc: {edge_acc}')


        if not is_neural_net or cfg.just_eval:
            break


if __name__ == '__main__':
    main()
