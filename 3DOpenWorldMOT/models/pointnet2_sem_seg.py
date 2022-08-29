from statistics import mode
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation

_layer_factory = {'PointNetSetAbstraction': PointNetSetAbstraction,
                'PointNetFeaturePropagation': PointNetFeaturePropagation}


class PointNet2(nn.Module):
    def __init__(self, num_classes, model_conv):
        super(PointNet2, self).__init__()
        down_conv = model_conv.down_conv
        if 'sa1' in down_conv.keys():
            sa_1 = down_conv.sa1
            self.sa1 = _layer_factory[sa_1.module_name](**sa_1)
        if 'sa2' in down_conv.keys():
            sa_2 = down_conv.sa2
            self.sa2 = _layer_factory[sa_2.module_name](**sa_2)
        if 'sa3' in down_conv.keys():
            sa_3 = down_conv.sa3
            self.sa2 = _layer_factory[sa_3.module_name](**sa_3)
        if 'sa4' in down_conv.keys():
            sa_4 = down_conv.sa4
            self.sa4 = _layer_factory[sa_4.module_name](**sa_4)
        if 'sa5' in down_conv.keys():
            sa_5 = down_conv.sa5
            self.sa5 = _layer_factory[sa_5.module_name](**sa_5)

        up_conv = model_conv.up_conv
        if 'fp5' in up_conv.keys():
            fp5 = up_conv.fp5
            self.fp5 = _layer_factory[fp5.module_name](**fp5)
        if 'fp4' in up_conv.keys():
            fp4 = up_conv.fp4
            self.fp4 = _layer_factory[fp4.module_name](**fp4)
        if 'fp3' in up_conv.keys():
            fp3 = up_conv.fp3
            self.fp3 = _layer_factory[fp3.module_name](**fp3)
        if 'fp2' in up_conv.keys():
            fp2 = up_conv.fp2
            self.fp2 = _layer_factory[fp2.module_name](**fp2)
        if 'fp1' in up_conv.keys():
            fp1 = up_conv.fp1
            self.fp1 = _layer_factory[fp1.module_name](**fp1)

        mlp_cls = model_conv.mlp_cls
        self.conv1 = nn.Conv1d(fp1.mlp[-1], mlp_cls.mlp1, 1)
        self.bn1 = nn.BatchNorm1d(mlp_cls.mlp1)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(mlp_cls.mlp1, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


class SemSegLoss(nn.Module):
    def __init__(self):
        super(SemSegLoss, self).__init__()
    def forward(self, pred, target, trans_feat, weight):
        total_loss = F.nll_loss(pred, target, weight=weight)

        return total_loss

if __name__ == '__main__':
    import  torch
    model = PointNet2(13)
    xyz = torch.rand(6, 9, 2048)
    (model(xyz))