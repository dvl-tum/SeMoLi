# CONFIGURATION HANDLING
import os
import hydra
from omegaconf import OmegaConf
import torch
import logging
from tqdm import tqdm
import numpy as np

from models import _model_factory, _loss_factory
from data_utils.PointCloudDataset import data_loaders
from data_utils.helper_tool import DataProcessing as DP
from data_utils import augmentation


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

    '''CREATE DIR'''
    experiment_dir = cfg.job_name
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

    experiment_dir = "_".join([cfg.models.model_name, cfg.data.dataset_name])
    experiment_dir = os.path.join(cfg.out_path, experiment_dir)

    return logger, experiment_dir, checkpoints_dir

@hydra.main(config_path="conf", config_name="conf")   
def main(cfg):
    OmegaConf.set_struct(cfg, False)  # This allows getattr and hasattr methods to function correctly

    logger, experiment_dir, checkpoints_dir = initialize(cfg)

    def log_string(str):
        logger.info(str)

    logger.info("start loading training data ...")
    
    train_loader, val_loader = data_loaders(cfg)

    weights = torch.Tensor(DP.get_class_weights(cfg.data.dataset_name)).cuda()

    log_string("The number of training data is: %d" % len(train_loader.dataset))
    log_string("The number of test data is: %d" % len(val_loader.dataset))

    '''MODEL LOADING'''
    seg_label_to_cat = {}
    for i, cat in enumerate(range(cfg.data.num_classes)):
        seg_label_to_cat[i] = cat

    classifier = _model_factory[cfg.models.model_name](cfg.data.num_classes, cfg.models).cuda()
    criterion = _loss_factory[cfg.models.model_name]().cuda()
    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)
    

    print(cfg.training.optim)
    if cfg.training.optim.optimizer.o_class == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=cfg.training.optim.optimizer.params.lr,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=cfg.training.optim.weight_decay
        )
    else:
        optimizer = torch.optim.SGD(
            classifier.parameters(),
            lr=cfg.training.optim.optimizer.params.lr,
            momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, cfg.training.epochs):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (
            global_epoch + 1, epoch + 1, cfg.training.epochs))

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
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(train_loader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(train_loader), total=len(train_loader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            points[:, :, :3] = augmentation.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, cfg.data.num_classes)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (cfg.training.batch_size * cfg.data.npoints)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/latest_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(val_loader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(cfg.data.num_classes)
            total_seen_class = [0 for _ in range(cfg.data.num_classes)]
            total_correct_class = [0 for _ in range(cfg.data.num_classes)]
            total_iou_deno_class = [0 for _ in range(cfg.data.num_classes)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(val_loader), total=len(val_loader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, cfg.data.num_classes)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (cfg.training.batch_size * cfg.data.npoints)
                tmp, _ = np.histogram(batch_label, range(cfg.data.num_classes + 1))
                labelweights += tmp

                for l in range(cfg.data.num_classes):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=np.float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=np.float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(cfg.data.num_classes):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l - 1],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    main()
