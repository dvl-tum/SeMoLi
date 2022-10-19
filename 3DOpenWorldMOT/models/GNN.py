from turtle import forward
import torch
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.nn import knn_graph, radius_graph
import torch.nn.functional as F
from torch_geometric.utils import softmax
import torch.nn as nn
import numpy as np


class ClusterLayer(MessagePassing):
    def __init__(self, traj_channels, pos_channels, edge_attr, use_batchnorm=False, use_drop=False, drop_p=0.4):
        super().__init__(aggr='mean')
        self.edge_attr = edge_attr
        if self.edge_attr == 'diffpos':
            in_channels_edge = pos_channels
        elif self.edge_attr == 'difftraj':
            in_channels_edge = traj_channels
        elif self.edge_attr == 'difftraj_diffpos':
            in_channels_edge = traj_channels + pos_channels
        self.edge_mlp = torch.nn.Linear(in_channels_edge, in_channels_edge)
        self.relu = nn.ReLU(inplace=True)
        self.batchnorm = nn.BatchNorm1d(in_channels_edge) \
            if use_batchnorm else use_batchnorm
        self.drop = nn.Dropout(p=drop_p) if use_drop else use_drop
        self.mlp = torch.nn.Linear(traj_channels * 2 + in_channels_edge, traj_channels)

    def edge_updater(self, edge_attr):
        edge_attr = self.edge_mlp(edge_attr)
        return edge_attr

    def forward(self, x1, x2, edge_index, edge_attr):
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        edge_attr = self.edge_updater(edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        x1 = self.propagate(edge_index, x1=x1, edge_attr=edge_attr)
        return x1, edge_attr
    
    def propagate(self, edge_index, x1, edge_attr):
        x1 = self.message(x1[edge_index[0, :]], x1[edge_index[1, :]], edge_attr)
        x1 = self.aggregate(x1, edge_index[1, :])
        return x1

    def message(self, x1_i, x1_j, edge_attr):
        x1_i = x1_i.view(x1_i.shape[0], -1)
        x1_j = x1_j.view(x1_j.shape[0], -1)
        tmp = torch.cat([x1_i, x1_j, edge_attr], dim=1)
        x = self.relu(self.mlp(tmp))
        if self.batchnorm:
            x = self.batchnorm(x)
        if self.drop:
            x = self.drop(x)
        return x


class ClusterGNN(MessagePassing):
    def __init__(self, traj_channels, pos_channels, k=32, r=0.5, graph='radius', edge_attr='diffpos'):
        super().__init__(aggr='mean')
        self.k = k
        self.r = r
        self.graph = graph
        self.edge_attr = edge_attr
        if self.edge_attr == 'diffpos':
            edge_dim = pos_channels
        elif self.edge_attr == 'difftraj':
            edge_dim = traj_channels
        elif self.edge_attr == 'difftraj_diffpos':
            edge_dim = pos_channels + traj_channels
        
        self.layer1 = ClusterLayer(traj_channels=traj_channels, pos_channels=pos_channels, edge_attr=edge_attr)
        self.layer2 = ClusterLayer(traj_channels=traj_channels, pos_channels=pos_channels, edge_attr=edge_attr)
        self.final = nn.Linear(edge_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()

    
    def initial_edge_attributes(self, x1, x2, edge_index):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if self.edge_attr == 'diffpos':
            edge_attr = x2[edge_index[0]] - x2[edge_index[1]]
        elif self.edge_attr == 'difftraj':
            edge_attr = x1[edge_index[0]] - x1[edge_index[1]]
        elif self.edge_attr == 'difftraj_diffpos':
            edge_attr = torch.hstack([x1[edge_index[0]] - x1[edge_index[1]], x2[edge_index[0]] - x2[edge_index[1]]])

        return edge_attr
    
    def forward(self, data, eval=False, use_edge_att=True):
        data = data.cuda()
        traj = data['traj']
        traj = traj.view(traj.shape[0], -1)
        batch = [torch.tensor([i] * (data._slice_dict['pc_list'][i+1]-data._slice_dict['pc_list'][i]).item()) for i in range(0, data._slice_dict['pc_list'].shape[0]-1)]
        batch = torch.cat(batch).cuda()
        batch_idx = data._slice_dict['pc_list']
        pc = data['pc_list']

        # get edges using knn graph (for computational feasibility)
        if self.graph == 'knn':
            edge_index = knn_graph(x=traj, k=self.k, batch=batch)
        elif self.graph == 'radius':
            edge_index = radius_graph(traj, self.r, batch)

        # get which edges belong to which batch
        batch_edge = torch.zeros(edge_index.shape[1]).cuda()
        for i in range(1, batch_idx.shape[0]-1):
            mask = (edge_index.T < batch_idx[i+1]) & (edge_index.T >= batch_idx[i])
            mask = mask[:, 0] * i
            batch_edge += mask
        
        edge_attr = self.initial_edge_attributes(traj, pc, edge_index)
        
        x, edge_attr = self.layer1(traj, pc, edge_index, edge_attr)
        x, edge_attr = self.layer2(traj, pc, edge_index, edge_attr)

        src, dst = edge_index
        # computes per edge index by computing dot product between node features
        if not use_edge_att:
            score = (x[src] * x[dst]).sum(dim=-1)
        # directly uses edge attirbutes
        else:
            score = self.final(edge_attr)

        score = self.sigmoid(score)

        if eval:
            score = score.cpu()
            data = data.cpu()

            # original score
            score_orig = torch.zeros((data['pc_list'].shape[0], data['pc_list'].shape[0]))
            score_orig[src, dst] = score

            score_orig = self.make_symmetric(score_orig, mode='minimum')
            cluster_id = 0
            cluster_dict = dict()
            for i, scores in enumerate(score_orig):
                idxs = (scores > 0).nonzero(as_tuple=True)[0]
                if i not in cluster_dict.keys():
                    cluster_dict[i] = cluster_id
                    cluster_id += 1
                greedy_id = cluster_dict[i]
                for idx in idxs:
                    if score_orig[idx, i] == 0:
                        continue
                    if idx.item() not in cluster_dict.keys() or torch.max(score_orig[idx]) < scores[idx]:
                        cluster_dict[idx.item()] = greedy_id
            clusters = np.array([v for v in cluster_dict.values()])

            return score, clusters, edge_index, batch_edge

        return score, edge_index, batch_edge

    def make_symmetric(self, score_orig, mode='minimum'):
        for i in range(score_orig.shape[0]):
            for j in range(i, score_orig.shape[0]):
                if score_orig[i, j] == 0 and score_orig[j, i] == 0:
                    continue
                if mode == 'minimum':
                    score_orig[i, j] = score_orig[j, i] = min(score_orig[i, j], score_orig[j, i])
                elif mode == 'maximum':
                    score_orig[i, j] = score_orig[j, i] = max(score_orig[i, j], score_orig[j, i])
        return score_orig

class GNNLoss(nn.Module):
    def __init__(self, bce_loss=True) -> None:
        super().__init__()
        
        self.bce_loss = nn.BCELoss().cuda() if bce_loss else bce_loss
        self.max_iter = 2000

    def forward(self, logits, data, edge_index):
        loss = 0
        if self.bce_loss:
            y_categories = data.point_categories.unsqueeze(
                0) == data.point_categories.unsqueeze(0).T
            y_categories = y_categories.cuda()
            edge_index = edge_index.cuda()
            y_categories = y_categories[
                edge_index[0, :], edge_index[1, :]].type(torch.FloatTensor).cuda()
            loss += self.bce_loss(logits.squeeze(), y_categories.squeeze())
        return loss
    
    def eval(self, logits, data, edge_index):
        loss = 0
        if self.bce_loss:
            y_categories = data.point_categories.unsqueeze(
                0) == data.point_categories.unsqueeze(0).T
            y_categories = y_categories[
                edge_index[0, :], edge_index[1, :]].type(torch.FloatTensor)
            loss += torch.nn.functional.binary_cross_entropy(
                logits.squeeze(), y_categories.squeeze())
        return loss
