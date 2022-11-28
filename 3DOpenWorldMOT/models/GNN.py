from turtle import forward
import torch
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.nn import knn_graph, radius_graph
import torch.nn.functional as F
from torch_geometric.utils import softmax
import torch.nn as nn
import numpy as np
from collections import defaultdict


class ClusterLayer(MessagePassing):
    def __init__(self, traj_channels, pos_channels, edge_attr, node_attr, use_batchnorm=False, use_drop=False, drop_p=0.4):
        super().__init__(aggr='mean')
        # get edge mlp
        self.edge_attr = edge_attr
        if self.edge_attr == 'diffpos':
            in_channels_edge = pos_channels
        elif self.edge_attr == 'difftraj' or self.edge_attr == 'diffpostraj':
            in_channels_edge = traj_channels
        elif self.edge_attr == 'difftraj_diffpos':
            in_channels_edge = traj_channels + pos_channels
        self.edge_mlp = torch.nn.Linear(in_channels_edge, in_channels_edge)

        # get edge relu, bn, drop
        self.edge_relu = nn.ReLU(inplace=True)
        self.edge_batchnorm = nn.BatchNorm1d(in_channels_edge) \
            if use_batchnorm else use_batchnorm
        self.edge_drop = nn.Dropout(p=drop_p) if use_drop else use_drop

        # get node mlp
        self.node_attr = node_attr
        if self.node_attr == 'pos':
            in_channels_node = pos_channels
        elif self.node_attr == 'traj' or self.node_attr == 'postraj':
            in_channels_node = traj_channels
        elif self.node_attr == 'traj_pos':
            in_channels_node = traj_channels + pos_channels
        self.mlp = torch.nn.Linear(in_channels_node * 2 + in_channels_edge, in_channels_node)

        # get node relu, bn, drop
        self.node_relu = nn.ReLU(inplace=True)
        self.node_batchnorm = nn.BatchNorm1d(in_channels_node) \
            if use_batchnorm else use_batchnorm
        self.node_drop = nn.Dropout(p=drop_p) if use_drop else use_drop

    def edge_updater(self, edge_attr):
        edge_attr = self.edge_relu(self.edge_mlp(edge_attr))
        if self.edge_batchnorm:
            edge_attr = self.edge_batchnorm(edge_attr)
        if self.edge_drop:
            edge_attr = self.edge_drop(edge_attr)
        return edge_attr

    def forward(self, node_attr, edge_index, edge_attr):
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        edge_attr = self.edge_updater(edge_attr)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        node_attr = self.propagate(edge_index, node_attr=node_attr, edge_attr=edge_attr)
        return node_attr, edge_attr
    
    def propagate(self, edge_index, node_attr, edge_attr):
        node_attr = self.message(node_attr[edge_index[0, :]], node_attr[edge_index[1, :]], edge_attr)
        node_attr = self.aggregate(node_attr, edge_index[1, :])
        return node_attr

    def message(self, x1_i, x1_j, edge_attr):
        x1_i = x1_i.view(x1_i.shape[0], -1)
        x1_j = x1_j.view(x1_j.shape[0], -1)
        tmp = torch.cat([x1_i, x1_j, edge_attr], dim=1)
        x = self.node_relu(self.mlp(tmp))
        if self.node_batchnorm:
            x = self.node_batchnorm(x)
        if self.node_drop:
            x = self.node_drop(x)
        return x


class ClusterGNN(MessagePassing):
    def __init__(self, traj_channels, pos_channels, k=32, r=0.5, graph='radius', edge_attr='diffpos', node_attr='traj', cut_edges=0.5, min_samples=20):
        super().__init__(aggr='mean')
        self.k = k
        self.r = r
        self.graph = graph
        self.edge_attr = edge_attr
        self.node_attr = node_attr
        if self.edge_attr == 'diffpos':
            edge_dim = pos_channels
        elif self.edge_attr == 'difftraj':
            edge_dim = traj_channels
        elif self.edge_attr == 'difftraj_diffpos':
            edge_dim = pos_channels + traj_channels
        
        self.layer1 = ClusterLayer(traj_channels=traj_channels, pos_channels=pos_channels, edge_attr=edge_attr, node_attr=node_attr)
        self.layer2 = ClusterLayer(traj_channels=traj_channels, pos_channels=pos_channels, edge_attr=edge_attr, node_attr=node_attr)
        self.final = nn.Linear(edge_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        self.cut_edges = cut_edges
        self.min_samples = min_samples

    def initial_edge_attributes(self, x1, x2, edge_index, distance='euclidean'):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if self.edge_attr == 'diffpos':
            a = x2[edge_index[0]]
            b = x2[edge_index[1]]
        elif self.edge_attr == 'difftraj':
            a = x1[edge_index[0]]
            b = x1[edge_index[1]]
        elif self.edge_attr == 'difftraj_diffpos':
            a = torch.stack([x2[edge_index[0]], x1[edge_index[0]]])
            b = torch.stack([x2[edge_index[1]], x1[edge_index[1]]])
        elif self.edge_attr == 'diffpostraj':
            a = x2[edge_index[0]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[0]]
            b = x2[edge_index[1]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[1]]
        
        edge_attr = b - a

        return edge_attr
    
    def initial_node_attributes(self, x1, x2):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if self.node_attr == 'pos':
            node_attr = x2
        elif self.node_attr == 'traj':
            node_attr = x1
        elif self.node_attr == 'difftraj_diffpos':
            node_attr = torch.stack([x2, x1])
        elif self.node_attr == 'diffpostraj':
            node_attr = x2.repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1

        return node_attr

    def forward(self, data, eval=False, use_edge_att=True):
        data = data.cuda()
        traj = data['traj']
        traj = traj.view(traj.shape[0], -1)
        batch = [torch.tensor([i] * (data._slice_dict['pc_list'][i+1]-data._slice_dict['pc_list'][i]).item()) for i in range(0, data._slice_dict['pc_list'].shape[0]-1)]
        batch = torch.cat(batch).cuda()
        batch_idx = data._slice_dict['pc_list']
        pc = data['pc_list']

        node_attr = self.initial_node_attributes(traj, pc)
        
        # get edges using knn graph (for computational feasibility)
        if self.graph == 'knn':
            edge_index = knn_graph(x=node_attr, k=self.k, batch=batch)
        elif self.graph == 'radius':
            edge_index = radius_graph(node_attr, self.r, batch)

        # get which edges belong to which batch
        batch_edge = torch.zeros(edge_index.shape[1]).cuda()
        for i in range(1, batch_idx.shape[0]-1):
            mask = (edge_index.T < batch_idx[i+1]) & (edge_index.T >= batch_idx[i])
            mask = mask[:, 0] * i
            batch_edge += mask
        
        edge_attr = self.initial_edge_attributes(traj, pc, edge_index)
        
        x, edge_attr = self.layer1(node_attr, edge_index, edge_attr)
        x, edge_attr = self.layer2(node_attr, edge_index, edge_attr)

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

            edges_filtered = list()
            scores_filtered = list()
            for i, e in enumerate(edge_index.T):
                if score[i] < self.cut_edges:
                    edges_filtered.append(e)
                    scores_filtered.append(score[i])

            edge_index = torch.stack(edges_filtered).T
            score = torch.stack(scores_filtered).T
            src, dst = edge_index

            # original score
            score_orig = torch.zeros((data['pc_list'].shape[0], data['pc_list'].shape[0]))
            score_orig[src, dst] = score.float()

            clusters = defaultdict(list)
            cluster_assignment = dict()
            id_count = 0
            import copy
            for i, scores in enumerate(score_orig):
                # get edges
                idxs = (scores > 0).nonzero(as_tuple=True)[0]
                other_ids = list()

                # check if node A in any cluster yet
                to_add = [i] if i not in cluster_assignment.keys() else []
                _id = None if i not in cluster_assignment.keys() else cluster_assignment[i]
                
                # iterate over edges and find clusters that need to be merged
                for idx in idxs:
                    if idx in cluster_assignment.keys() and _id is None:
                        _id = cluster_assignment[idx.item()]
                    elif idx.item() in cluster_assignment.keys():
                        if _id != cluster_assignment[idx.item()]:
                            other_ids.append(cluster_assignment[idx.item()])
                    else:
                        to_add.append(idx.item())

                # if no connected node as well as node i is not in cluster yet
                if _id is None:
                    _id = id_count
                    id_count += 1

                # change cluster ids and merge clusters
                for change_id in set(other_ids):
                    for node in clusters[change_id]:
                        cluster_assignment[node] = _id
                    clusters[_id].extend(copy.deepcopy(clusters[change_id]))
                    del clusters[change_id]

                # add nodes that where in no cluster yet
                clusters[_id].extend(to_add)
                for node in to_add:
                    cluster_assignment[node] = _id

            clusters_new = defaultdict(list)
            cluster_assignment_new = dict()
            for c, node_list in clusters.items():
                if len(node_list) < self.min_samples:
                    clusters_new[-1].extend(node_list)
                    for n in node_list:
                        cluster_assignment_new[n] = -1
                else:
                    clusters_new[c] = node_list
                    for n in node_list:
                        cluster_assignment_new[n] = c

            clusters = clusters_new
            cluster_assignment = cluster_assignment_new
            # self.visualize(torch.arange(pc.shape[0]), edge_index, pc[:, :-1], clusters, data.timestamps[0])

            clusters = np.array([cluster_assignment[k] for k in sorted(cluster_assignment.keys())])

            return score, clusters, edge_index, batch_edge

        return score, edge_index, batch_edge


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
