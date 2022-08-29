from .pointnet2_sem_seg import PointNet2, SemSegLoss

_model_factory = {'pointnet2_sem_seg': PointNet2}
_loss_factory = {'pointnet2_sem_seg': SemSegLoss}