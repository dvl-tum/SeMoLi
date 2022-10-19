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
    checkpoints_dir = os.path.join(experiment_dir, 'checkpoints/')
    os.makedirs(checkpoints_dir, exist_ok=True)

    '''LOG'''
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.info(cfg)

    return logger, experiment_dir, checkpoints_dir

@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    logger, experiment_dir, checkpoints_dir = initialize(cfg)

    def log_string(str):
        logger.info(str)

    logger.info("start loading training data ...")
    
    train_loader, val_loader, test_loader = get_TrajectoryDataLoader(cfg)

    if train_loader is not None:
        log_string("The number of training data is: %d" % len(train_loader.dataset))
    if val_loader is not None:
        log_string("The number of test data is: %d" % len(val_loader.dataset))

    '''MODEL LOADING'''
    print(_model_factory[cfg.models.model_name])
    model = _model_factory[cfg.models.model_name](**cfg.models.hyperparams)
    criterion = _loss_factory[cfg.models.model_name]
    start_epoch = 0
    if cfg.models.model_name != 'DBSCAN' and cfg.models.model_name != 'SpectralClustering':
        model = model.cuda()
        criterion = criterion(**cfg.models.loss_hyperparams).cuda()

        if cfg.models.model_name != 'SimpleGraph':
            try:
                checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
                start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['model_state_dict'])
                log_string('Use pretrain model')
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

    global_epoch = 0
    best_metric = 0
    for epoch in range(start_epoch, cfg.training.epochs):
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
                loss = criterion(logits, data, edge_index)
                loss.backward()
                optimizer.step()

                loss_sum += loss
            log_string('Training mean loss: %f' % (loss_sum / num_batches))

            savepath = str(checkpoints_dir) + '/latest_model.pth'
            log_string('Saving at %s...' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)

        if epoch % cfg.training.eval_per_seq == 0:
            num_batches = len(val_loader)
            eval_loss = 0
            tracker = Tracker3D(
                os.path.join(cfg.out_path, 'tracker_out', cfg.job_name),
                split='val',
                inact_thresh=cfg.tracker_options.inact_thresh,
                a_threshold=cfg.tracker_options.a_threshold,
                i_threshold=cfg.tracker_options.i_threshold,
                every_x_frame=cfg.data.every_x_frame,
                num_interior=cfg.tracker_options.num_interior,
                overlap=cfg.tracker_options.overlap)
            with torch.no_grad():
                if cfg.models.model_name != 'DBSCAN' and \
                    cfg.models.model_name != 'SpectralClustering':
                    model = model.eval()
                log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
                for i, (data) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                    
                    logits, clusters, edge_index, batch_edge = model(data, eval=True)
                    data = data.cpu()

                    detections = tracker.get_detections(
                        data.pc_list,
                        data.traj,
                        clusters,
                        data.flow[:, 0, :],
                        data.timestamps[0],
                        data.log_id[0])
                    tracker.associate(detections, i+1 == len(val_loader))
                    
                    if cfg.models.model_name != 'DBSCAN' \
                        and cfg.models.model_name != 'SpectralClustering' \
                            and cfg.models.model_name != 'SimpleGraph':
                        loss = criterion.eval(logits, data, edge_index)
                        eval_loss += loss

                tracker_dir = os.path.join(os.getcwd(), f"{os.sep}".join(tracker.out_path.split(os.sep)[:-1]))
                seq_list = os.listdir(os.path.join(tracker.out_path, tracker.split))

                output_res, _ = evaluate_av2_ow_MOT(
                    gt_folder=os.path.join(os.getcwd(), cfg.data.data_dir),
                    trackers_folder=tracker_dir,
                    seq_to_eval=seq_list,
                    remove_far='80' in cfg.data.trajectory_dir,
                    remove_non_drive='non_drive' in cfg.data.trajectory_dir,
                    remove_non_move=0, #cfg.data.remove_static,
                    remove_non_move_thresh=cfg.data.static_thresh,
                    do_print=False
                    )
                metric = 100*(sum(output_res['AV2_OW'][cfg.job_name]['COMBINED_SEQ'][
                    'cls_comb_cls_av']['HOTA']['RHOTA'])/len(output_res['AV2_OW'][
                        cfg.job_name]['COMBINED_SEQ']['cls_comb_cls_av']['HOTA']['RHOTA']))

                log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))

                if metric >= best_metric and cfg.models.model_name != 'DBSCAN' \
                    and cfg.models.model_name != 'SpectralClustering' \
                        and cfg.models.model_name != 'SimpleGraph':
                    best_metric = metric
                    savepath = str(checkpoints_dir) + '/best_model.pth'
                    log_string('Saving at %s...' % savepath)
                    state = {
                        'epoch': epoch,
                        'class_avg_iou': metric,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }
                    torch.save(state, savepath)
                
                log_string('Best metric: %f' % best_metric)
                
                if cfg.models.model_name == 'DBSCAN' \
                    or cfg.models.model_name == 'SpectralClustering' \
                        or cfg.models.model_name == 'SimpleGraph':
                    quit()
                    
        global_epoch += 1


if __name__ == '__main__':
    main()
