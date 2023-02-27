# CONFIGURATION HANDLING
import os
import hydra
from omegaconf import OmegaConf
import torch
import logging
from tqdm import tqdm

from models import _model_factory, _loss_factory, Tracker3D
from data_utils.TrajectoryDataset import get_TrajectoryDataLoader
from TrackEvalOpenWorld.scripts.run_av2_ow import evaluate_av2_ow_MOT
import wandb

# FOR DETECTION EVALUATION
from evaluation import eval_detection
from evaluation import calc_nmi


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR


'''import sys
import traceback

class TracePrints(object):
  def __init__(self):    
    self.stdout = sys.stdout
  def write(self, s):
    self.stdout.write("Writing %r\n" % s)
    traceback.print_stack(file=self.stdout)

sys.stdout = TracePrints()'''


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def initialize(cfg):
    '''HYPER PARAMETER'''
    #print(cfg.training.gpu)
    #os.environ["CUDA_VISIBLE_DEVICES"] = cfg.training.gpu

    '''CREATE DIR'''
    experiment_dir = cfg.job_name
    experiment_dir = "_".join([cfg.models.model_name, cfg.data.dataset_name])
    experiment_dir = os.path.join(cfg.out_path, experiment_dir)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoints_dir = '../../../checkpoints/'
    os.makedirs(checkpoints_dir, exist_ok=True)
    out_path = '../../../out/'
    os.makedirs(out_path, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(cfg)

    return logger, experiment_dir, checkpoints_dir, out_path

@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    logger, experiment_dir, checkpoints_dir, out_path = initialize(cfg)

    def log_string(str):
        logger.info(str)

    logger.info("start loading training data ...")
    
    train_loader, val_loader, test_loader = get_TrajectoryDataLoader(cfg)

    if train_loader is not None:
        log_string("The number of training data is: %d" % len(train_loader.dataset))
    if val_loader is not None:
        log_string("The number of test data is: %d" % len(val_loader.dataset))

    '''MODEL LOADING'''
    print(cfg.models.model_name)
    print(_model_factory[cfg.models.model_name])
    model = _model_factory[cfg.models.model_name](**cfg.models.hyperparams)
    criterion = _loss_factory[cfg.models.model_name]
    start_epoch = 0
    if 'DBSCAN' not in cfg.models.model_name and cfg.models.model_name != 'SpectralClustering':
        model = model.cuda()
        criterion = criterion(**cfg.models.loss_hyperparams).cuda()

        if cfg.models.model_name != 'SimpleGraph':
            node = '_nodescore' if cfg.models.hyperparams.use_node_score else ''
            cluster = '_' + cfg.models.hyperparams.clustering
            my_graph = "_mygraph" if cfg.models.hyperparams.my_graph else '_torchgraph'

            name = cfg.models.hyperparams.edge_attr + "_" + cfg.models.hyperparams.node_attr + node + cluster + my_graph
            print(f'Using this name: {name}')
            os.makedirs(checkpoints_dir + name, exist_ok=True)
            if cfg.wandb:
                wandb.init(config=cfg, project=cfg.job_name, name=name)
            try:
                checkpoint = torch.load(cfg.models.weight_path)
                start_epoch = checkpoint['epoch'] if not cfg.just_eval else start_epoch
                model.load_state_dict(checkpoint['model_state_dict'])
                met = checkpoint['class_avg_iou']
                log_string(f'Use pretrain model {met}')
            except:
                log_string('No existing model, starting training from scratch...')

            if cfg.training.optim.optimizer.o_class == 'Adam':
                optimizer = torch.optim.Adam(
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

        def bn_momentum_adjust(m, momentum):
            if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
                m.momentum = momentum
    elif cfg.models.model_name == 'DBSCAN':
        name = cfg.models.hyperparams.input + "_" + str(cfg.models.hyperparams.thresh) + "_" + str(cfg.models.hyperparams.min_samples)
    elif cfg.models.model_name == 'DBSCAN_Intersection':
        name = cfg.models.hyperparams.input_traj + "_" + str(cfg.models.hyperparams.thresh_traj) + "_" + str(cfg.models.hyperparams.min_samples_traj) + "_" + str(cfg.models.hyperparams.thresh_pos) + "_" + str(cfg.models.hyperparams.min_samples_pos) + "_" + str(cfg.models.hyperparams.flow_thresh)

    global_epoch = 0
    best_metric = 0
    for epoch in range(start_epoch, cfg.training.epochs):
        if not cfg.just_eval:
            '''Train on chopped scenes'''
            log_string('**** Epoch %d (%d/%s) ****' % (
                global_epoch + 1, epoch + 1, cfg.training.epochs))

            if cfg.models.model_name != 'DBSCAN' \
                and cfg.models.model_name != 'SpectralClustering'\
                    and cfg.models.model_name != 'SimpleGraph':

                lr = max(cfg.training.optim.optimizer.params.lr * (
                    cfg.lr_scheduler.params.gamma ** (
                        epoch // cfg.lr_scheduler.params.step_size)), cfg.lr_scheduler.params.clip)
                log_string('Learning rate:%f' % lr)
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
                momentum = max(cfg.training.optim.bn_scheduler.params.bn_momentum * (
                    cfg.training.optim.bn_scheduler.params.bn_decay ** (
                        epoch // cfg.training.optim.bn_scheduler.params.decay_step)), \
                            cfg.training.optim.bn_scheduler.params.bn_clip)
                print('BN momentum updated to: %f' % momentum)
                model = model.apply(lambda x: bn_momentum_adjust(x, momentum))
                num_batches = len(train_loader)
                loss_sum = 0
                model = model.train()
                log_string('---- EPOCH %03d TRAINING ----' % (global_epoch + 1))
                for i, data in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
                    data = data.cuda()
                    optimizer.zero_grad()
                    logits, edge_index, batch_edge = model(data)
                    loss, log_dict = criterion(logits, data, edge_index)
                    loss.backward()
                    optimizer.step()

                    loss_sum += loss
                    if cfg.wandb:
                        for k, v in log_dict.items():
                            wandb.log({k: v, "epoch": epoch, "batch": i})
                    else:
                        logger.info(log_dict)
                log_string('Training mean loss: %f' % (loss_sum / num_batches))

                savepath = str(checkpoints_dir) + name + '/latest_model.pth'
                log_string(f'Saving at {savepath}...')

                state = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)

        if epoch % cfg.training.eval_per_seq == 0: # and (epoch != 0 or cfg.just_eval):
            num_batches = len(val_loader)
            eval_loss = 0
            nmis = list()
            tracker = Tracker3D(
                out_path + name,
                split='val',
                a_threshold=cfg.tracker_options.a_threshold,
                i_threshold=cfg.tracker_options.i_threshold,
                every_x_frame=cfg.data.every_x_frame,
                num_interior=cfg.tracker_options.num_interior,
                overlap=cfg.tracker_options.overlap,
                av2_loader=val_loader.dataset.loader)
            with torch.no_grad():
                if 'DBSCAN' not in cfg.models.model_name and \
                    cfg.models.model_name != 'SpectralClustering':
                    model = model.eval()
                log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
                import time
                for i, (data) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                    
                    logits, clusters, edge_index, batch_edge = model(data, eval=True, name=name)
                    data = data.cpu()
                    non_background = data['point_instances'] != 0
                    if not len(clusters):
                        continue

                    # nmi = calc_nmi.calc_normalized_mutual_information(data['point_instances'][non_background], torch.from_numpy(clusters)[non_background])
                    # nmi = calc_nmi.calc_normalized_mutual_information(data['point_instances'], clusters)
                    nmi = calc_nmi.calc_normalized_mutual_information(data['point_instances'][clusters!=-1], clusters[clusters!=-1])
                    
                    nmis.append(nmi)

                    detections = tracker.get_detections(
                        data.pc_list,
                        data.traj,
                        clusters,
                        data.timestamps,
                        data.log_id[0],
                        data['point_instances'],
                        last=i+1 == len(val_loader))
                    
                    if 'DBSCAN' not in cfg.models.model_name \
                        and cfg.models.model_name != 'SpectralClustering' \
                            and cfg.models.model_name != 'SimpleGraph':
                        loss, log_dict = criterion.eval(logits, data, edge_index)
                        eval_loss += loss
                        if cfg.wandb:
                            for k, v in log_dict.items():
                                wandb.log({k: v, "epoch": epoch, "batch": i})
                        else:
                            logger.info(log_dict)

                tracker_dir = os.path.join(tracker.out_path, tracker.split)
                # tracker_dir = os.path.join(tracker_dir, 'first_training', tracker.split)
                try:
                    seq_list = os.listdir(tracker_dir)
                except:
                    seq_list = list()
                
                log_string('NMI: %f' % (sum(nmis) / len(nmis)))

                cluster_metric = [sum(nmis) / len(nmis)]
                
                _, detection_metric = eval_detection.eval_detection(
                    gt_folder=os.path.join(os.getcwd(), cfg.data.data_dir),
                    trackers_folder=tracker_dir,
                    seq_to_eval=seq_list,
                    remove_far='80' in cfg.data.trajectory_dir,
                    remove_non_drive='non_drive' in cfg.data.trajectory_dir,
                    remove_non_move=cfg.data.remove_static,
                    remove_non_move_strategy=cfg.data.remove_static_strategy,
                    remove_non_move_thresh=3000/3600,
                    classes_to_eval='all',
                    debug=cfg.data.debug,
                    name=name)
                

                if 'DBSCAN' not in cfg.models.model_name \
                    and cfg.models.model_name != 'SpectralClustering' \
                        and cfg.models.model_name != 'SimpleGraph':
                    log_string('eval mean loss: %f' % (eval_loss / float(num_batches)))
                for_logs = {met: m for met, m in zip(['AP', 'ATE', 'ASE', 'AOE' ,'CDS'], detection_metric)}
                if cfg.wandb:
                    for met, m in for_logs.items():
                        wandb.log({met: m, "epoch": epoch})
                    wandb.log({'NMI': cluster_metric[0], "epoch": epoch})

                if cfg.metric == 'cluster':
                    metric = cluster_metric
                else:
                    metric = detection_metric
                
                metric = metric[0]
                if metric >= best_metric:
                    best_metric = metric
                    if 'DBSCAN' not in cfg.models.model_name \
                        and cfg.models.model_name != 'SpectralClustering' \
                            and cfg.models.model_name != 'SimpleGraph' \
                                and not cfg.just_eval:
                        savepath = str(checkpoints_dir) + name + '/best_model.pth'
                        log_string('Saving at %s...' % savepath)
                        state = {
                            'epoch': epoch,
                            'NMI': cluster_metric[0],
                            'class_avg_iou': detection_metric[0],
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                        }
                        torch.save(state, savepath)
                
                log_string(f'Best {cfg.metric} metric: {best_metric}, cluster metric: {cluster_metric}, detection metric: {for_logs}')
                
        if 'DBSCAN' not in cfg.models.model_name \
            or cfg.models.model_name == 'SpectralClustering' \
                or cfg.models.model_name == 'SimpleGraph' \
                    or cfg.just_eval:
            break
                    
        global_epoch += 1


if __name__ == '__main__':
    main()
