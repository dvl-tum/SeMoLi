from .data_utils.splits import get_seq_list_fixed_val
import os
import torch
import logging
from tqdm import tqdm 

import torch.nn as nn
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader as PyGDataLoader
import torch.distributed as dist

from SeMoLi.models import _model_factory, _loss_factory, Detector3D
from SeMoLi.data_utils.TrajectoryDataset import get_TrajectoryDataLoader
import wandb

# FOR DETECTION EVALUATION
from SeMoLi.evaluation import eval_detection
from SeMoLi.evaluation import calc_nmi
from pyarrow import feather
import shutil
import numpy as np
import random
from SeMoLi.utils.get_name import get_name


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


class Trainer():
    def __init__(self, cfg, rank, world_size):
        self.rank = rank
        self.cfg = cfg 

        if self.cfg.training.half_precision:
            self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None
        self.is_neural_net = self.cfg.models.model_name != 'DBSCAN' \
                        and self.cfg.models.model_name != 'DBSCAN_Intersection'

        if self.cfg.training.multi_gpu and self.rank != 'cpu':
            os.environ['MASTER_ADDR'] = 'localhost'
            os.environ['MASTER_PORT'] = '12355'
            os.environ['WORLD_SIZE'] = f'{world_size}'
            os.environ['RANK'] = f'{self.rank}'
            torch.cuda.set_device(self.rank)
            dist.init_process_group('nccl', rank=self.rank, world_size=world_size)
        
        #os.environ["CUDA_VISIBLE_DEVICES"] = self.cfg.training.gpu
        torch.manual_seed(1)
        # torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        np.random.seed(1)
        random.seed(1)

        # set up logger
        self.logger = logging.getLogger("Model")
        self.logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        self.logger.info(self.cfg)

        # get data
        self.get_data()
        if self.cfg.data.do_process:
            return

        # create experiment (dectections will be stored here) and checkpoint dirs
        out_path = os.path.join(self.cfg.root_dir, 'out/')
        os.makedirs(out_path, exist_ok=True)
        # experiments dir
        self.experiment_dir = os.path.join(out_path, f'detections_{self.cfg.data.detection_set}/')
        os.makedirs(self.experiment_dir, exist_ok=True)
        # checkpoints dir
        self.checkpoints_dir = os.path.join(out_path, 'checkpoints/')
        os.makedirs(self.checkpoints_dir, exist_ok=True)

        # get name of experiments and make save directories
        self.name = get_name(self.cfg)
        if self.rank == 0: 
            self.logger.info(f'Using this name: {self.name}')
        if self.rank == 0:
            self.make_dirs()

        # load model
        self.load_model()

    def train(self):
        # send model to data paralell
        if self.cfg.training.multi_gpu and self.is_neural_net:
            self.model = nn.parallel.DistributedDataParallel(self.model, find_unused_parameters=False)
        
        best_metric = 0
        if self.cfg.training.just_eval:
            self.start_epoch = 0

        # iterate over number of epochs
        for epoch in range(self.start_epoch, self.cfg.training.epochs):
            if not self.cfg.training.just_eval:
                '''Train on chopped scenes'''
                if self.rank == 0:
                    self.logger.info('**** Epoch (%d/%s) ****' % (
                        epoch + 1, self.cfg.training.epochs))
                
                if self.is_neural_net:
                    self.train_one_epoch(
                        epoch)
                if self.model is None:
                    self.logger.info("Terminating training due to nan values...")
                    self.on_return()
                    return
            
            # evaluate
            if epoch % self.cfg.training.eval_every_x == 0:
                # do corr clustering every eval_corr_every_x epochs if epoch not 0
                do_corr_clustering = epoch % self.cfg.training.eval_corr_every_x == 0 and epoch != 0
                # do corr clustering if only eval
                do_corr_clustering = do_corr_clustering or self.cfg.training.just_eval
                # do corr clustering if oracle node or edge
                do_corr_clustering = do_corr_clustering or self.cfg.models.hyperparams.oracle_node or self.cfg.models.hyperparams.oracle_edge
                # no corr clustering when hyper search
                do_corr_clustering = do_corr_clustering and not self.cfg.training.hypersearch
                best_metric = self.eval_one_epoch(
                    do_corr_clustering,
                    epoch,
                    best_metric)
            
            # if DBSCAN baselines or just eval return
            if not self.is_neural_net or self.cfg.training.just_eval:
                wandb.finish()
                self.on_return()
                return
    
        # final_evaluation with best model weights
        dist.barrier()
        if self.rank == 0:
            for d in os.listdir(self.experiment_dir + self.name):
                shutil.rmtree(self.experiment_dir + self.name + f'/{d}')
        self.final_evaluation(
                    epoch,
                    best_metric)
        wandb.finish()
        self.on_return()

    def on_return(self):
        if not self.cfg.training.just_eval and not self.cfg.evaluation.keep_checkpoint and self.rank == 0:
            shutil.rmtree(self.checkpoints_dir + self.name)
        if not self.cfg.evaluation.keep_detections and self.rank == 0:
            shutil.rmtree(self.experiment_dir + self.name)
        shutil.rmtree(f'{self.cfg.root_dir}/outputs', ignore_errors=True) 
    def make_dirs(self):
        if self.rank == 0 or not self.cfg.training.multi_gpu:
            if os.path.isdir(self.experiment_dir + self.name):
                shutil.rmtree(self.experiment_dir + self.name)
            if not self.cfg.training.just_eval and os.path.isdir(str(self.checkpoints_dir) + self.name):
                shutil.rmtree(str(self.checkpoints_dir) + self.name)
            os.makedirs(self.experiment_dir + self.name, exist_ok=True)
            self.logger.info(f'Detections are stored under {self.experiment_dir + self.name}...')
            os.makedirs(str(self.checkpoints_dir) + self.name, exist_ok=True)
            self.logger.info(f'Checkpoints are stored under {str(self.checkpoints_dir) + self.name}...')

    def get_optimizer(self):
        if self.cfg.training.optim.optimizer.o_class == 'Adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=self.cfg.training.optim.optimizer.params.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.cfg.training.optim.weight_decay
            )
        elif self.cfg.training.optim.optimizer.o_class == 'RAdam':
            optimizer = torch.optim.RAdam(
                self.model.parameters(),
                lr=self.cfg.training.optim.optimizer.params.lr,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.cfg.training.optim.weight_decay
            )
        else:
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.cfg.training.optim.optimizer.params.lr,
                momentum=0.9)
        return optimizer

    def load_model(self):
        # get model and criterions
        self.model = _model_factory[self.cfg.models.model_name](rank=self.rank, **self.cfg.models.hyperparams)
        if self.rank == 0:
            self.logger.info(self.model)
        self.criterion = _loss_factory[self.cfg.models.model_name]
        start_epoch = 0

        if 'DBSCAN' not in self.cfg.models.model_name:
            # get model and criterion
            self.model = self.model.to(self.rank)
            self.criterion = self.criterion(**self.cfg.models.loss_hyperparams, rank=self.rank).to(self.rank)
            os.makedirs(self.checkpoints_dir + self.name, exist_ok=True)
            
            # checkpoint loading
            try:
                # load chceckpoint
                checkpoint = torch.load(self.cfg.models.weight_path)
                chkpt_new = dict()
                for k, v in checkpoint['model_state_dict'].items():
                    if 'module' in k:
                        chkpt_new[k[7:]] = v
                    else:
                        chkpt_new[k] = v
                start_epoch = checkpoint['epoch'] if not self.cfg.training.just_eval else start_epoch
                self.model.load_state_dict(chkpt_new)
                met = checkpoint['best_metric'] if 'best_metric' in checkpoint.keys() else 0
                metric_mode = checkpoint['metric_mode'] if 'metric_mode' in checkpoint.keys() else None
                if self.rank == 0:
                    self.logger.info(f'Use pretrained model with {metric_mode}: {met}')
            except:
                # if could not find model path
                if self.cfg.models.weight_path != '':
                    if self.rank == 0:
                        self.logger.info(f'Did not find pretrained model with {self.cfg.models.weight_path}')
                    quit()
                # training from scratch
                else:
                    if self.rank == 0:
                        self.logger.info('No existing model, starting training from scratch...')
            
            # get optimizer
            self.optimizer = self.get_optimizer()
        self.start_epoch = start_epoch 
        # init wandb
        if self.cfg.training.wandb:
            wandb.login(key='3b716e6ab76d92ef92724aa37089b074ef19e29c')
            wandb.init(config=self.cfg, project=self.cfg.category, group=self.cfg.job_name, name=self.name + '_' + str(self.rank))

    def make_loader(self, data, shuffle=True):
        # make dataloader with distributed sampler depending if nessecary
        if data is not None:
            if not self.cfg.training.multi_gpu:
                data_loader = PyGDataLoader(
                    data,
                    batch_size=self.cfg.training.batch_size,
                    drop_last=False,
                    shuffle=shuffle,
                    num_workers=self.cfg.training.num_workers)
            else:
                sampler = DistributedSampler(
                        data,
                        num_replicas=torch.cuda.device_count(),
                        drop_last=False,
                        rank=self.rank, 
                        shuffle=shuffle,
                        seed=5)
                data_loader = PyGDataLoader(
                    data,
                    batch_size=self.cfg.training.batch_size,
                    sampler=sampler,
                    num_workers=self.cfg.training.num_workers)
        else:
            data_loader = None
        return data_loader

    def get_data(self):
        # get train, val and test data 
        if self.cfg.training.just_eval:
            train_data, val_data = \
                get_TrajectoryDataLoader(self.cfg, train=False)
        else:
            train_data, val_data = \
                get_TrajectoryDataLoader(self.cfg)
        
        # get dataloaders 
        self.av2_data_loader = val_data.loader
        self.train_loader = self.make_loader(train_data, shuffle=True)
        self.val_loader = self.make_loader(val_data, shuffle=False)
    
        if self.train_loader is not None and self.rank == 0:
            self.logger.info("The number of training data is: %d" % len(self.train_loader.dataset))
        if self.val_loader is not None and self.rank == 0:
            self.logger.info("The number of test data is: %d" % len(self.val_loader.dataset))

    def train_one_epoch(self, epoch):
        # Adapt learning rate
        lr = max(self.cfg.training.optim.optimizer.params.lr * (
            self.cfg.lr_scheduler.params.gamma ** (
                epoch // self.cfg.lr_scheduler.params.step_size)), self.cfg.lr_scheduler.params.clip)
        if self.rank == 0:
            self.logger.info('Learning rate:%f' % lr)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        # init logging
        self.model = self.model.train()
        if self.rank == 0:
            self.logger.info('---- EPOCH %03d TRAINING ----' % (epoch + 1))
        node_loss = torch.zeros(2).to(self.rank)
        node_acc = torch.zeros(6).to(self.rank)
        edge_loss = torch.zeros(2).to(self.rank)
        edge_acc = torch.zeros(6).to(self.rank)
        
        # iterate over dataset
        for batch, data in tqdm(enumerate(self.train_loader), total=len(self.train_loader), smoothing=0.9):
            if batch % 50 == 0:
                self.logger.info(f'Epoch {epoch}: batch {batch}/{len(self.train_loader)}')
            data = data.to(self.rank)
            self.optimizer.zero_grad()

            # if half precision use scaler
            if self.cfg.training.half_precision:
                with torch.cuda.amp.autocast():
                    logits, edge_index, batch_edge = self.model(data)
                    loss, log_dict = self.criterion(logits, data, edge_index)
                    
                    # means all points filtered
                    if loss is None:
                        continue

                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
            else:
                logits, edge_index, batch_edge = self.model(data)
                loss, log_dict = self.criterion(logits, data, edge_index)
                
                # means all points filtered
                if loss is None:
                        continue

                loss.backward()
                self.optimizer.step()
            
            # logging per iteration
            if self.cfg.training.wandb:
                for k, v in log_dict.items():
                    _type = ['all', 'neg', 'pos']
                    for i in range(int(v.shape[0]/2)):
                        if v[2*i+1]:
                            wandb.log({f'{k} {_type[i]}': v[2*i], "epoch": epoch})
            
            if 'train bce loss edge' in log_dict.keys():
                edge_loss += log_dict['train bce loss edge']
            if 'train bce loss node' in log_dict.keys():
                node_loss += log_dict['train bce loss node']
            if 'train accuracy edge' in log_dict.keys():
                edge_acc += log_dict['train accuracy edge']
            if 'train accuracy node' in log_dict.keys():
                node_acc = log_dict['train accuracy node']

        # reducing logs over gpus
        if self.cfg.training.multi_gpu:
            dist.all_reduce(node_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(edge_loss, op=dist.ReduceOp.SUM)
            dist.all_reduce(edge_acc, op=dist.ReduceOp.SUM)
            dist.all_reduce(node_acc, op=dist.ReduceOp.SUM)

        node_loss = round(float(node_loss[0] / node_loss[1]), 2)
        edge_loss = round(float(edge_loss[0] / edge_loss[1]), 2)
        edge_acc = {k: round(float(edge_acc[2*i] / edge_acc[2*i+1]), 2) \
                for i, k in enumerate(['all', 'neg', 'pos']) if edge_acc[2*i+1] != 0}
        node_acc = {k: round(float(node_acc[2*i] / node_acc[2*i+1]), 2) \
                for i, k in enumerate(['all', 'neg', 'pos']) if node_acc[2*i+1] != 0}

        # printing and saving model
        if self.rank == 0 or self.rank == 'cpu' or not self.cfg.training.multi_gpu:
            if 'train bce loss edge' in log_dict.keys():
                self.logger.info(f'train bce loss edge per epoch: {edge_loss}')
            if 'train bce loss node' in log_dict.keys():
                self.logger.info(f'train bce loss node per epoch: {node_loss}')
            if 'train accuracy edge' in log_dict.keys():
                self.logger.info(f'train accuracy edge per epoch (all / neg / pos): {edge_acc}')
            if 'train accuracy node' in log_dict.keys():
                self.logger.info(f'train accuracy node per epoch (all / neg / pos): {node_acc}')

            savepath = str(self.checkpoints_dir) + self.name + '/latest_model.pth'
            self.logger.info(f'Saving at {savepath}...')

            state = {
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
            }
            torch.save(state, savepath)

    def eval_one_epoch(self, do_corr_clustering, epoch, best_metric):
        
        node_loss = torch.zeros(2).to(self.rank)
        node_acc = torch.zeros(6).to(self.rank)
        edge_loss = torch.zeros(2).to(self.rank)
        edge_acc = torch.zeros(6).to(self.rank)
        nmis = torch.zeros(2).to(self.rank)
        # intialize detector
        if do_corr_clustering:
            detector = Detector3D(
                self.experiment_dir + self.name,
                split=self.cfg.data.detection_set,
                num_interior=self.cfg.detector.num_interior,
                av2_loader=self.av2_data_loader,
                rank=self.rank,
                kNN=self.cfg.detector.kNN,
                threshold=self.cfg.detector.threshold,
                median_flow=self.cfg.detector.median_flow,
                median_center=self.cfg.detector.median_center)
        
        _nmis = list()
        log_dict = dict()
        with torch.no_grad():
            if self.is_neural_net:
                self.model = self.model.eval()
            if self.rank == 0:
                self.logger.info('---- EPOCH %03d EVALUATION ----' % (epoch + 1))
                self.logger.info(f'Doing correlation clustering {do_corr_clustering}')
            # Iterate over validation set
            for batch, (data) in tqdm(enumerate(self.val_loader), total=len(self.val_loader), smoothing=0.9):
                if batch % 50 == 0:
                    self.logger.info(f'Epoch {epoch}: batch {batch}/{len(self.val_loader)}')
                # compute clusters

                logits, all_clusters, edge_index, _ = self.model(data, eval=True, name=self.name, corr_clustering=do_corr_clustering)

                batch_idx = data._slice_dict['pc_list']
                if do_corr_clustering:
                    for g, clusters in enumerate(all_clusters):
                        # continue if we didnt find clusters
                        if not len(clusters):
                            if batch+1 == len(self.val_loader) and g+1 == len(all_clusters):
                                found = detector.to_feather()
                                if not found:
                                    self.logger.info(f'No detections found in {data.log_id[g]}')
                            continue

                        # compute nmi
                        nmi = calc_nmi.calc_normalized_mutual_information(
                            data['point_instances'].cpu()[batch_idx[g]:batch_idx[g+1]], clusters)
                        _nmis.append(nmi)

                        # generate detections
                        _ = detector.get_detections(
                            data.pc_list[batch_idx[g]:batch_idx[g+1]],
                            data.traj[batch_idx[g]:batch_idx[g+1]],
                            clusters,
                            data.timestamps[g].unsqueeze(0),
                            data.log_id[g],
                            data['point_instances'][batch_idx[g]:batch_idx[g+1]],
                            data['point_categories'][batch_idx[g]:batch_idx[g+1]],
                            last= batch+1 == len(self.val_loader) and g+1 == len(all_clusters))

                # if not DBSCAN baselines compute acc and loss
                if self.is_neural_net and logits[0] is not None:
                    loss, log_dict = self.criterion(logits, data, edge_index, self.rank, mode='eval')
                    
                    if loss is None:
                        continue
                
                # logging
                if self.cfg.training.wandb:
                    for k, v in log_dict.items():
                        _type = ['all', 'neg', 'pos']
                        for i in range(int(v.shape[0]/2)):
                            if v[2*i+1]:
                                wandb.log({f'{k} {_type[i]}': v[2*i], "epoch": epoch})

                if 'eval bce loss edge' in log_dict.keys():
                    edge_loss += log_dict['eval bce loss edge']
                if 'eval bce loss node' in log_dict.keys():
                    node_loss += log_dict['eval bce loss node']
                if 'eval accuracy edge' in log_dict.keys():
                    edge_acc += log_dict['eval accuracy edge']
                if 'eval accuracy node' in log_dict.keys():
                    node_acc = log_dict['eval accuracy node']

                if do_corr_clustering:
                    nmi = sum(_nmis) / len(_nmis)
                    nmis[0] += float(nmi)
                    nmis[1] += 1
            
            # reducing after epoch
            if self.cfg.training.multi_gpu:
                if do_corr_clustering:
                    dist.all_reduce(nmis, op=dist.ReduceOp.SUM)
                if self.is_neural_net:
                    dist.all_reduce(node_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(edge_loss, op=dist.ReduceOp.SUM)
                    dist.all_reduce(edge_acc, op=dist.ReduceOp.SUM)
                    dist.all_reduce(node_acc, op=dist.ReduceOp.SUM)

            if self.is_neural_net:
                node_loss = round(float(node_loss[0] / node_loss[1]), 2)
                edge_loss = round(float(edge_loss[0] / edge_loss[1]), 2)
                edge_acc = {k: round(float(edge_acc[2*i] / edge_acc[2*i+1]), 2) \
                        for i, k in enumerate(['all', 'neg', 'pos']) if edge_acc[2*i+1] != 0}
                node_acc = {k: round(float(node_acc[2*i] / node_acc[2*i+1]), 2) \
                        for i, k in enumerate(['all', 'neg', 'pos']) if node_acc[2*i+1] != 0}
            
            if do_corr_clustering:
                nmis = round(float(nmis[0] / nmis[1]), 2)
            
            # log, combine outputs from different ranks and evaluatet
            if self.rank == 0 or self.rank == 'cpu' or not self.cfg.training.multi_gpu:
                if do_corr_clustering:
                    self.logger.info(f'nmi: {nmis}')
                
                if self.is_neural_net:
                    if 'eval bce loss edge' in log_dict.keys():
                        self.logger.info(f'eval bce loss edge per epoch: {edge_loss}')
                    if 'eval bce loss node' in log_dict.keys():
                        self.logger.info(f'eval bce loss node per epoch: {node_loss}')
                    if 'eval accuracy edge' in log_dict.keys():
                        self.logger.info(f'eval accuracy edge per epoch (all / neg / pos): {edge_acc}')
                    if 'eval accuracy node' in log_dict.keys():
                        self.logger.info(f'eval accuracy node per epoch (all / neg / pos): {node_acc}')

                if do_corr_clustering:
                    # combine output of different ranks
                    out = os.path.join(self.experiment_dir + self.name,  detector.split)
                    for _rank in os.listdir(self.experiment_dir + self.name):
                        rank_path = os.path.join(self.experiment_dir + self.name, _rank, detector.split)
                        if not os.path.isdir(rank_path):
                            continue
                        for log_id in os.listdir(rank_path):
                            df = feather.read_feather(os.path.join(rank_path, log_id, 'annotations.feather'))
                            write_path = os.path.join(out, log_id, 'annotations.feather')
                            if os.path.isfile(write_path):
                                df = df.append(feather.read_feather(write_path))
                            else:
                                os.makedirs(os.path.join(out, log_id), exist_ok=True)
                            feather.write_feather(df, write_path)
                        shutil.rmtree(rank_path)
                    
                    self.logger.info(f"Loading detections from {os.path.join(detector.out_path, detector.split)}...")
                    # get sequence list for evaluation
                    detector_dir = os.path.join(self.experiment_dir + self.name, detector.split)

                    seq_list = get_seq_list_fixed_val(
                        self.cfg.data.data_dir,
                        self.cfg.root_dir,
                        detection_set=self.cfg.data.detection_set,
                        percentage=self.cfg.data.percentage_data_val)
                    if self.cfg.data.debug:
                        seq_list = [seq_list[5]]

                    # average NMI
                    cluster_metric = [nmis]
                    self.logger.info(f'NMI: {cluster_metric[0]}')
                    self.logger.info(f'Evaluating detection performance...')

                    # evaluate detection
                    _, detection_metric, _ = eval_detection.eval_detection(
                        gt_folder=os.path.join(os.getcwd(), self.cfg.data.data_dir),
                        trackers_folder=detector_dir,
                        split='val' if 'evaluation' in self.cfg.data.detection_set else 'train',
                        seq_to_eval=seq_list,
                        remove_far=True,#'80' in self.cfg.data.trajectory_dir,
                        remove_non_drive='non_drive' in self.cfg.data.trajectory_dir,
                        remove_non_move=self.cfg.data.remove_static_gt,
                        remove_non_move_strategy=self.cfg.data.remove_static_strategy,
                        remove_non_move_thresh=self.cfg.data.remove_static_thresh,
                        filter_class=self.cfg.evaluation.filter_class,
                        filter_moving=self.cfg.evaluation.filter_moving,
                        use_matched_category=self.cfg.evaluation.use_matched_category,
                        heuristics=self.cfg.evaluation.heuristics,
                        debug=self.cfg.data.debug,
                        name=self.name,
                        inflate_bb=self.cfg.evaluation.inflate_bb, 
                        root_dir=self.cfg.root_dir,
                        filtered_file_path=self.cfg.data.filtered_file_path,
                        flow_path=self.cfg.evaluation.filtered_pc_path,
                        only_level_1=self.cfg.evaluation.only_level_1,
                        score_thresh=self.cfg.evaluation.score_thresh)

                    # log metrics
                    for_logs = {met: m for met, m in zip(['AP', 'ATE', 'ASE', 'AOE' ,'CDS'], detection_metric)}
                    self.logger.info(f'Detection metrics {for_logs}...')

                    if self.cfg.training.wandb:
                        for met, m in for_logs.items():
                            wandb.log({met: m, "epoch": epoch})
                        wandb.log({'NMI': cluster_metric[0], "epoch": epoch})
                
                # store weights if neural net
                if self.is_neural_net:
                    if edge_acc['all'] >= best_metric:
                        best_metric = edge_acc['all']
                        if self.is_neural_net and not self.cfg.training.just_eval:
                            savepath = str(self.checkpoints_dir) + self.name + '/best_model.pth'
                            self.logger.info('Saving at %s...' % savepath)
                            if do_corr_clustering:
                                state = {
                                    'epoch': epoch,
                                    'ACC': edge_acc['all'],
                                    'best_metric': best_metric,
                                    'metric_mode': 'edge_accuracy',
                                    'NMI': cluster_metric[0],
                                    'class_avg_iou': detection_metric[0],
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                }
                            else:
                                state = {
                                    'epoch': epoch,
                                    'ACC': edge_acc,
                                    'best_metric': best_metric,
                                    'metric_mode': 'edge_accuracy',
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict(),
                                }
                            torch.save(state, savepath)
                
                    if do_corr_clustering:
                        self.logger.info(f'Best acc: {best_metric}, acc: {edge_acc}, cluster metric: {cluster_metric}, detection metric: {for_logs}')
                    else:
                        self.logger.info(f'Best acc: {best_metric}, acc: {edge_acc}')
        
        return best_metric

    def final_evaluation(self, epoch, best_metric):
        
        # FINAL EVALUATION WITH BEST WEIGHTS
        if self.is_neural_net:
            if self.rank == 0:
                self.logger.info('**** FINAL EVALUATION ****')
            best_model_path = str(self.checkpoints_dir) + self.name + '/best_model.pth'
            checkpoint = torch.load(best_model_path)
            chkpt_new = dict()
            self.start_epoch = checkpoint['epoch'] if not self.cfg.training.just_eval else self.start_epoch
            self.model.load_state_dict(checkpoint['model_state_dict'])

        self.eval_one_epoch(True, epoch, best_metric)


