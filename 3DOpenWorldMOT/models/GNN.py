import torch
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.nn import knn_graph, radius_graph
import torch.nn.functional as F
from torch_geometric.utils import softmax
import torch.nn as nn
import numpy as np
from collections import defaultdict
import rama_py
import random
import matplotlib
import os
import logging
import networkx as nx
import models.losses
import math
import sklearn.metrics


rgb_colors = {}
for name, hex in matplotlib.colors.cnames.items():
    rgb_colors[name] = matplotlib.colors.to_rgb(hex)
rgb_colors = list(rgb_colors.values())
rgb_colors = rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
random.shuffle(rgb_colors)

logger = logging.getLogger("Model.GNN")

class ClusterLayer(MessagePassing):
    def __init__(self, traj_channels, pos_channels, edge_attr, node_attr, use_batchnorm=False, use_drop=False, drop_p=0.4):
        super().__init__(aggr='mean')
        # get edge dim
        self.edge_attr = edge_attr
        if self.edge_attr == 'diffpos':
            in_channels_edge = pos_channels
        elif self.edge_attr == 'difftraj' or self.edge_attr == 'diffpostraj':
            in_channels_edge = traj_channels
        elif self.edge_attr == 'difftraj_diffpos':
            in_channels_edge = traj_channels + pos_channels
        elif self.edge_attr == 'pertime_diffpostraj':
            in_channels_edge = int(traj_channels/3)
        elif self.edge_attr == 'min_mean_max_diffpostrajtime':
            in_channels_edge = 3

        # get node dim
        self.node_attr = node_attr
        if self.node_attr == 'pos':
            in_channels_node = pos_channels
        elif self.node_attr == 'traj' or self.node_attr == 'postraj':
            in_channels_node = traj_channels
        elif self.node_attr == 'traj_pos':
            in_channels_node = traj_channels + pos_channels
        elif self.node_attr == 'min_mean_max_vel':
            in_channels_node = 3

        # get edge mlp
        self.edge_mlp = torch.nn.Linear(in_channels_node * 2 + in_channels_edge, in_channels_edge)

        # get edge relu, bn, drop
        self.edge_relu = nn.ReLU(inplace=True)
        self.edge_batchnorm = nn.BatchNorm1d(in_channels_edge) \
            if use_batchnorm else use_batchnorm
        self.edge_drop = nn.Dropout(p=drop_p) if use_drop else use_drop

        #get node mlp
        self.mlp = torch.nn.Linear(in_channels_node + in_channels_edge, in_channels_node)

        # get node relu, bn, drop
        self.node_relu = nn.ReLU(inplace=True)
        self.node_batchnorm = nn.BatchNorm1d(in_channels_node) \
            if use_batchnorm else use_batchnorm
        self.node_drop = nn.Dropout(p=drop_p) if use_drop else use_drop

    def edge_updater(self, edge_attr, node_attr, edge_index):
        x1_i = node_attr[edge_index[0, :]]
        x1_i = x1_i.view(x1_i.shape[0], -1)
        # receiving
        x1_j = node_attr[edge_index[1, :]]
        x1_j = x1_j.view(x1_j.shape[0], -1)
        update_input = torch.cat([x1_i, x1_j, edge_attr], dim=1)

        edge_attr = self.edge_relu(self.edge_mlp(update_input))
        if self.edge_batchnorm:
            edge_attr = self.edge_batchnorm(edge_attr)
        if self.edge_drop:
            edge_attr = self.edge_drop(edge_attr)
        return edge_attr

    def forward(self, node_attr, edge_index, edge_attr):
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        edge_attr = self.edge_updater(edge_attr, node_attr, edge_index)

        # propagate_type: (x: OptPairTensor, alpha: Tensor)
        node_attr = self.propagate(edge_index, node_attr=node_attr, edge_attr=edge_attr)
        return node_attr, edge_attr
    
    def propagate(self, edge_index, node_attr, edge_attr):
        dim_size = node_attr.shape[0]
        node_attr = self.message(node_attr[edge_index[0, :]], node_attr[edge_index[1, :]], edge_attr)
        node_attr = self.aggregate(node_attr, edge_index[1, :], dim_size=dim_size)
        return node_attr

    def message(self, x1_i, x1_j, edge_attr):
        # sending
        x1_i = x1_i.view(x1_i.shape[0], -1)
        # receiving
        x1_j = x1_j.view(x1_j.shape[0], -1)
        # only use sending and edge_attr
        tmp = torch.cat([x1_i, edge_attr], dim=1)
        x = self.node_relu(self.mlp(tmp))
        if self.node_batchnorm:
            x = self.node_batchnorm(x)
        if self.node_drop:
            x = self.node_drop(x)
        return x


def simplediff(a, b):
    return b-a


class ClusterGNN(MessagePassing):
    def __init__(
            self,
            traj_channels,
            pos_channels,
            k=32,
            k_eval=64,
            r=0.5,
            graph='radius',
            edge_attr='diffpos',
            graph_construction='pos',
            node_attr='traj',
            cut_edges=0.5,
            min_samples=20,
            use_node_score=False,
            clustering='correlation',
            do_visualize=True,
            my_graph=True,
            oracle_node=False,
            oracle_edge=False,
            dataset='waymo'):
        super().__init__(aggr='mean')
        self.k = k
        self.k_eval = k_eval
        self.r = r
        self.graph = graph
        self.edge_attr = edge_attr
        self.node_attr = node_attr
        self.graph_construction = graph_construction
        self.use_node_score = use_node_score
        self.clustering = clustering
        if self.edge_attr == 'diffpos':
            edge_dim = pos_channels
        elif self.edge_attr == 'difftraj' or self.edge_attr == 'diffpostraj':
            edge_dim = traj_channels
        elif self.edge_attr == 'difftraj_diffpos':
            edge_dim = pos_channels + traj_channels
        elif self.edge_attr == 'pertime_diffpostraj':
            edge_dim = int(traj_channels/3)
        elif self.edge_attr == 'min_mean_max_diffpostrajtime':
            edge_dim = 3
        
        # get node mlp
        self.node_attr = node_attr
        if self.node_attr == 'pos':
            node_dim = pos_channels
        elif self.node_attr == 'traj' or self.node_attr == 'postraj':
            node_dim = traj_channels
        elif self.node_attr == 'traj_pos':
            node_dim = traj_channels + pos_channels
        elif self.node_attr == 'min_mean_max_vel':
            node_dim = 3

        self.layer1 = ClusterLayer(traj_channels=traj_channels, pos_channels=pos_channels, edge_attr=edge_attr, node_attr=node_attr)
        self.layer2 = ClusterLayer(traj_channels=traj_channels, pos_channels=pos_channels, edge_attr=edge_attr, node_attr=node_attr)
        self.final = nn.Linear(edge_dim, 1)
        self.final_node = nn.Linear(node_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # self.sigmoid = torch.nn.Tanh()
        self.cut_edges = cut_edges
        self.min_samples = min_samples
        self.do_visualize = do_visualize
        self.my_graph = my_graph
        self.oracle_node = oracle_node
        self.oracle_edge = oracle_edge
        self.dataset = dataset

        self.opts = rama_py.multicut_solver_options("PD")
        self.opts.sanitize_graph = True
        self.opts.verbose = False

    def initial_edge_attributes(self, x1, x2, edge_index, distance='euclidean'):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if self.edge_attr == 'diffpos':
            a = x2[edge_index[0]]
            b = x2[edge_index[1]]
            d = simplediff
        elif self.edge_attr == 'difftraj':
            a = x1[edge_index[0]]
            b = x1[edge_index[1]]
            d = simplediff
        elif self.edge_attr == 'difftraj_diffpos':
            a = torch.stack([x2[edge_index[0]], x1[edge_index[0]]])
            b = torch.stack([x2[edge_index[1]], x1[edge_index[1]]])
            d = simplediff
        elif self.edge_attr == 'diffpostraj':
            a = x2[edge_index[0]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[0]]
            b = x2[edge_index[1]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[1]]
            d = simplediff
        elif self.edge_attr == 'pertime_diffpostraj' or 'min_mean_max_diffpostrajtime':
            a = x1.view(x1.shape[0], -1, 3)[edge_index[0]]+x2[edge_index[0]].unsqueeze(1)
            a = a.view(edge_index.shape[1], -1)

            b = x1.view(x1.shape[0], -1, 3)[edge_index[1]]+x2[edge_index[1]].unsqueeze(1)
            b = b.view(edge_index.shape[1], -1)

            a_shape = a.shape
            a = a.view(-1, 3)
            b = b.view(-1, 3)
            d = torch.nn.PairwiseDistance(p=2)
        
        edge_attr = d(a, b)
        if self.edge_attr == 'pertime_diffpostraj':
            edge_attr = edge_attr.view(a_shape[0], -1)
        elif self.edge_attr == 'min_mean_max_diffpostrajtime':
            edge_attr = edge_attr.view(a_shape[0], -1)
            edge_attr = torch.vstack([
                edge_attr.min(dim=-1).values,
                edge_attr.max(dim=-1).values,
                edge_attr.mean(dim=-1)]).T

        return edge_attr
    
    def initial_node_attributes(self, x1, x2, _type, timestamps=None, batch=None):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if _type == 'pos':
            node_attr = x2
        elif _type == 'traj':
            node_attr = x1
        elif _type == 'traj_pos':
            node_attr = torch.stack([x2, x1])
        elif _type == 'postraj' or _type =='mean_dist_over_time':
            node_attr = x1.view(x1.shape[0], -1, 3)+x2.unsqueeze(1)
            if _type == 'postraj':
                node_attr = node_attr.view(node_attr.shape[0], -1)
        elif _type == 'min_mean_max_vel':
            time = timestamps[batch, :]
            node_attr = x1.view(x1.shape[0], -1, 3)
            diff_time = timestamps[batch, 1:] - timestamps[batch, :-1]
            if 'argo' in self.dataset:
                diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
            else:
                diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)
            node_attr = node_attr[:, 1:, :] - node_attr[:, :-1, :]
            node_attr = torch.linalg.norm(node_attr, dim=-1)
            node_attr = node_attr / diff_time
            node_attr = torch.vstack([
                node_attr.min(dim=-1).values,
                node_attr.max(dim=-1).values,
                node_attr.mean(dim=-1)]).T

        return node_attr
    
    def get_graph(self, node_attr, r=5, max_num_neighbors=16, batch_idx=None, type='radius', metric='euclidean', batch=None):
        # my graph
        _idxs_0, _idxs_1 = list(), list()
        for start, end in zip(batch_idx[:-1], batch_idx[1:]):
            # iterate over frames in batch
            X = node_attr[start:end]

            # get distances between nodes
            if self.edge_attr == 'diffpos':
                dist = torch.from_numpy(sklearn.metrics.pairwise_distances(X.cpu().numpy(), metric=metric)).cuda()
            else:                
                # following two lines are faster but cuda oom
                # dist = torch.cdist(X_time, X_time)
                # dist = dist.mean(dim=0)
                dist = torch.zeros(X.shape[0], X.shape[0]).cuda()
                for t in range(X.shape[1]):
                    dist += torch.cdist(X[:, t, :].unsqueeze(0),X[:, t, :].unsqueeze(0)).squeeze()
                dist = dist / X.shape[1]

            # set diagonal elements to 0to have no self-loops
            dist.fill_diagonal_(100)

            # get indices up to max_num_neighbors per node --> knn neighbors
            num_neighbors = min(max_num_neighbors, dist.shape[0])
            idxs_0 = torch.tile(torch.arange(dist.shape[0]).unsqueeze(1).cuda(), (1, num_neighbors)).flatten()
            idxs_1 = dist.topk(k=num_neighbors, dim=1, largest=False).indices.flatten()

            # if radius graph, filter nodes that are within radius 
            # but don't exceed max num neighbors
            if type == 'radius':
                dist = dist[idxs_0, idxs_1]
                idx = torch.where(dist<r)[0]
                idxs_0, idxs_1 = idxs_0[idx], idxs_1[idx]
            
            _idxs_0.append(idxs_0)
            _idxs_1.append(idxs_1)

        _idxs_0 = torch.hstack(_idxs_0)
        _idxs_1 = torch.hstack(_idxs_1)

        edge_index = torch.vstack([_idxs_0, _idxs_1])

        return edge_index

    def forward(self, data, eval=False, use_edge_att=True, augment=True, visualize=False, name='General'):
        '''
        clustering: 'heuristic' / 'correlation'
        '''
        data = data.cuda()
        traj = data['traj']
        if traj.shape[0] == 0:
            return [None, None], list(), None, None
        
        traj = traj.view(traj.shape[0], -1)
        pc = data['pc_list']

        node_attr = self.initial_node_attributes(traj, pc, self.node_attr, data['timestamps'], data['batch'])
        graph_attr = self.initial_node_attributes(traj, pc, self.graph_construction)

        # get edges using knn graph (for computational feasibility)
        k = self.k if not eval else self.k_eval
        if self.graph == 'knn':
            if self.my_graph:
                edge_index = self.get_graph(
                    graph_attr, self.r, max_num_neighbors=k, batch_idx=data._slice_dict['pc_list'], type='knn', batch=data['batch'])
            else:
                edge_index = knn_graph(x=graph_attr, k=k, batch=data['batch'])
        elif self.graph == 'radius':
            if self.my_graph:
                edge_index = self.get_graph(
                    graph_attr, self.r, max_num_neighbors=k, batch_idx=data._slice_dict['pc_list'], type='radius', batch=data['batch'])
            else:
                edge_index = radius_graph(graph_attr, self.r, data['batch'], max_num_neighbors=k)

        # add negative edges to edge_index
        if not eval and augment:
            point_instances = data.point_instances.unsqueeze(
                0) == data.point_instances.unsqueeze(0).T
            # setting edges that do not belong to object to zero
            point_instances[data.point_instances == 0, :] = False
            num_pos = edge_index[:, point_instances[
                edge_index[0, :], edge_index[1, :]]].shape[1]
            num_neg = edge_index.shape[1] - num_pos

            # fast version
            missing_neg = int((num_pos - num_neg))
            a, b = list(), list()
            if missing_neg > 0 and point_instances.shape[0]:
                for _ in range(max(math.ceil(missing_neg/point_instances.shape[0])*2, 1)):
                    a.append(torch.randperm(point_instances.shape[0]))
                    b.append(torch.randperm(point_instances.shape[0]))
                a = torch.cat(a)
                b = torch.cat(b)
                a, b = a[~point_instances[a, b]], b[~point_instances[a, b]]
                a, b = a[:missing_neg], b[:missing_neg]
            elif point_instances.shape[0]:
                missing_pos = -missing_neg
                for _ in range(max(math.ceil(missing_pos/point_instances.shape[0])*2, 1)):
                    a.append(torch.randperm(point_instances.shape[0]))
                    b.append(torch.randperm(point_instances.shape[0]))
                a = torch.cat(a)
                b = torch.cat(b)
                a, b = a[point_instances[a, b]], b[point_instances[a, b]]
                a, b = a[:missing_pos], b[:missing_pos]
            add_idxs = torch.stack([a, b]).cuda()
            edge_index = torch.cat([edge_index.T, add_idxs.T]).T

        if edge_index.shape[1] == 0:
            return [None, None], torch.tensor(list(range(pc.shape[0]))), None, None
        
        edge_attr = self.initial_edge_attributes(traj, pc, edge_index)

        edge_attr = edge_attr.float()
        node_attr = node_attr.float()

        node_attr, edge_attr = self.layer1(node_attr, edge_index, edge_attr)
        node_attr, edge_attr = self.layer2(node_attr, edge_index, edge_attr)

        src, dst = edge_index
        # computes per edge index by computing dot product between node features
        if not use_edge_att:
            score = (node_attr[src] * node_attr[dst]).sum(dim=-1)
        # directly uses edge attirbutes
        else:
            score = self.final(edge_attr)
        node_score = self.final_node(node_attr)
                   
        if eval:
            score = self.sigmoid(score)
            node_score = self.sigmoid(node_score)
            if self.oracle_edge:
                score[data['point_instances'][src] == data['point_instances'][dst]] = 1
                score[data['point_instances'][src] != data['point_instances'][dst]] = 0
                score[data['point_instances'][src] <= 0] = 0
                score[data['point_instances'][dst] <= 0] = 0
            if self.oracle_node:
                node_score[data['point_categories']>0] = 1
                node_score[data['point_categories']<=0] = 1

            if self.clustering == 'correlation':
                if self.do_visualize:
                    point_instances = data.point_instances.unsqueeze(
                        0) == data.point_instances.unsqueeze(0).T
                    # setting edges that do not belong to object to zero
                    # --> instance 0 is no object
                    point_instances[data.point_instances == 0, :] = False
                    point_instances = point_instances.cuda()
                    point_instances = point_instances[
                        edge_index[0, :], edge_index[1, :]]
                                        
                    gt_edges = edge_index.T[point_instances].T
                    gt_clusters = data.point_instances != 0
                    gt_clusters = gt_clusters.type(torch.FloatTensor).cuda().cpu().numpy().tolist()
                    # gt_clusters = data.point_categories.cpu().numpy().tolist()

                    self.visualize(
                        torch.arange(pc.shape[0]),
                        gt_edges,
                        pc,
                        gt_clusters,
                        data.timestamps[0,0],
                        mode='groundtruth',
                        name=name)

                    # visualize with all the same cluster
                    self.visualize(
                        torch.arange(pc.shape[0]),
                        edge_index,
                        pc,
                        np.ones(pc.shape[0]),
                        data.timestamps[0,0],
                        mode='before',
                        name=name)
                    
                    edges_filtered = list()
                    for i, e in enumerate(edge_index.T):
                        if self.use_node_score:
                            valid_nodes = node_score[e[0]] > 0.5 or node_score[e[1]] > 0.5
                        else:
                            valid_nodes = True
                        if score[i] > self.cut_edges and valid_nodes:
                            edges_filtered.append(e)
                    if len(edges_filtered):
                        edges_filtered = torch.stack(edges_filtered).T
                    else:
                        edges_filtered = torch.empty((2, ))

                    self.visualize(
                        torch.arange(pc.shape[0]),
                        edges_filtered,
                        pc,
                        np.ones(pc.shape[0]),
                        data.timestamps[0,0],
                        mode='filtered',
                        name=name)

                # add edges to nodes not in edge set with dummy score
                edges = set(
                    edge_index.cpu().numpy()[0, :].tolist() +
                    edge_index.cpu().numpy()[1, :].tolist())
                diff = set(list(range(pc.shape[0]))).difference(edges)
                if len(diff):
                    _edge_index = torch.tensor([[d, 0] for d in diff]).T
                    edge_index = torch.cat([edge_index.T, _edge_index.cuda().T]).T
                    score = torch.cat([score, torch.tensor([0]*len(diff)).cuda().unsqueeze(1)])

                try:
                    rama_out = rama_py.rama_cuda(
                        [e[0] for e in edge_index.T.cpu().numpy()],
                        [e[1] for e in edge_index.T.cpu().numpy()], 
                        (score.cpu().numpy()*2)-1,
                        self.opts)
                    clusters = rama_out[0]
                except:
                    clusters = np.arange(node_score.shape[0])

                # filter out nodes thatare classified as non-objects
                if self.use_node_score:
                    clusters = torch.tensor(clusters)
                    clusters[(node_score.cpu() < 0.5).squeeze()] = -1
                    clusters = clusters.numpy()

                _clusters = defaultdict(list)
                for i, c in enumerate(clusters):
                    _clusters[c].append(i)

                cluster_assignment_new = dict()
                for c, node_list in _clusters.items():
                    if len(node_list) < self.min_samples:
                        for n in node_list:
                            cluster_assignment_new[n] = -1
                    else:
                        for n in node_list:
                            cluster_assignment_new[n] = c

                clusters = np.array([cluster_assignment_new[k] for k in sorted(cluster_assignment_new.keys())])

                if self.do_visualize:
                    # visualize with all the with predicted cluster
                    self.visualize(
                        torch.arange(pc.shape[0]),
                        edge_index,
                        pc,
                        clusters,
                        data.timestamps[0,0],
                        mode='after',
                        name=name)
            
            elif self.clustering == 'DBSCAN':
                model = sklearn.cluster.DBSCAN(min_samples=self.min_samples, eps=0.5)
                clustering = model.fit(node_attr.cpu().numpy()) # only flow 0.0015
                clusters = clustering.labels_
                if self.do_visualize:
                    # visualize with all the with predicted cluster
                    self.visualize(
                        torch.arange(pc.shape[0]),
                        edge_index,
                        pc,
                        clusters,
                        data.timestamps[0,0],
                        mode='after',
                        name=name)

            elif self.clustering == 'girvan_newman':
                edges_filtered = list()
                scores_filtered = list()
                for i, e in enumerate(edge_index.T):
                    if self.use_node_score:
                        valid_nodes = node_score[e[0]] > 0.5 or node_score[e[1]] > 0.5
                    else:
                        valid_nodes = True

                    if score[i] > self.cut_edges and valid_nodes:
                        edges_filtered.append(e)
                        scores_filtered.append(score[i])

                edge_index = torch.stack(edges_filtered).T
                score = torch.stack(scores_filtered).T

                G = nx.Graph()
                G.add_nodes_from(torch.arange(pc.shape[0]).numpy())
                G.add_edges_from(edge_index.T.cpu().numpy())
                comp = nx.algorithms.community.centrality.girvan_newman(G)
                print(comp)
                clusters = dict()
                for i, com in enumerate(comp):
                    for c in com:
                        clusters[c] = i
                clusters = [clusters[c] for c in sorted(clusters.keys())]

            else:
                print('Invalid clustering choice')
                quit()

            return [score, node_score], clusters, edge_index, None

        return [score, node_score], edge_index, None

    def visualize(self, nodes, edge_indices, pos, clusters, timestamp, mode='before', name='General'): 
        os.makedirs(f'../../../vis_graph/{name}', exist_ok=True)
        import networkx as nx
        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        clusters_dict = defaultdict(list)
        for i, c in enumerate(clusters):
            clusters_dict[c].append(i)
        clusters = clusters_dict

        # adapt edges to predicted clusters
        if mode == 'after':
            edge_indices = list()
            for c, nodelist in clusters.items():
                if c == -1:
                    continue
                for i, node1 in enumerate(nodelist):
                    for j, node2 in enumerate(nodelist):
                        if j <= i:
                            continue
                        edge_indices.append([node1, node2])
            edge_indices = torch.tensor(edge_indices).cuda().T

        # take only x and y position
        pos = pos[:, :-1]

        # make graph
        pos = {n.item(): p for n, p in zip(nodes, pos.cpu().numpy())}
        G = nx.Graph()
        G.add_nodes_from(nodes.numpy())
        G.add_edges_from(edge_indices.T.cpu().numpy())

        colors = [(255, 255, 255)] * nodes.shape[0]
        col_dict = dict()
        for i, (c, node_list) in enumerate(clusters.items()):
            for node in node_list:
                colors[node] = rgb_colors[i]
            col_dict[c] = rgb_colors[i]

        # save graph
        labels = {n.item(): str(n.item()) for n in nodes}
        plt.figure(figsize=(50, 50))
        nx.draw_networkx_edges(G, pos, width=3)
        nx.draw_networkx_nodes(G, pos, node_size=2, node_color=colors)
        # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_color='red')
        plt.axis("off")
        plt.savefig(f'../../../vis_graph/{name}/{timestamp}_{mode}.png', bbox_inches='tight')
        plt.close()
        logger.info(f'Stored to ../../../vis_graph/{name}/{timestamp}_{mode}.png...')


class GNNLoss(nn.Module):
    def __init__(self, bce_loss=False, node_loss=False, focal_loss=True) -> None:
        super().__init__()
        
        self.bce_loss = bce_loss
        self.node_loss = node_loss
        self.focal_loss = focal_loss
        self.max_iter = 2000

    def forward(self, logits, data, edge_index, weight=False, weight_node=False):
        edge_logits, node_logits = logits
        loss = 0
        log_dict = dict()
        if self.bce_loss or self.focal_loss:
            # get all positive egdes
            point_instances = data.point_instances.unsqueeze(
                0) == data.point_instances.unsqueeze(0).T
            # setting edges that do not belong to object to zero
            # --> instance 0 is no object
            point_instances[data.point_instances == 0, :] = False
            point_instances = point_instances.cuda()
            edge_index = edge_index.cuda()
            point_instances = point_instances[
                edge_index[0, :], edge_index[1, :]].type(torch.FloatTensor).cuda()        

            # compute loss
            if self.bce_loss:
                # weight pos and neg samples
                if weight:
                    num_pos = torch.sum((point_instances==1).float())
                    num_neg = torch.sum((point_instances==0).float())
                    pos_weight = num_neg/num_pos
                    pos_weight = pos_weight.cpu()
                    nbatch = edge_logits.squeeze().shape[0]
                    pos_weight = torch.ones(nbatch) * pos_weight
                    bce_loss_e = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
                else:
                    bce_loss_e = nn.BCEWithLogitsLoss().cuda()

                # compute loss
                bce_loss_edge = bce_loss_e(
                    edge_logits.squeeze(), point_instances.squeeze())
                loss += bce_loss_edge
                log_dict['train bce loss edge'] = bce_loss_edge.item()

                # get accuracy
                logits_rounded = edge_logits.clone().detach().squeeze()
                logits_rounded[logits_rounded>0.5] = 1
                logits_rounded[logits_rounded<=0.5] = 0
                correct = torch.sum(logits_rounded == point_instances.squeeze())
                edge_accuracy = correct/logits_rounded.shape[0]
                log_dict['train edge accuracy'] = edge_accuracy.item()

            if self.focal_loss:
                focal_loss = models.losses.FocalLoss()
                loss += focal_loss(
                    edge_logits.squeeze(), point_instances.squeeze())
        
        if self.node_loss:
            is_object = data.point_instances != 0
            is_object = is_object.type(torch.FloatTensor).cuda()
            # weight pos and neg samples
            if weight_node:
                num_pos = torch.sum((is_object==1).float())
                num_neg = torch.sum((is_object==0).float())
                pos_weight = num_neg/num_pos
                nbatch = node_logits.squeeze().shape[0]
                pos_weight = torch.ones(nbatch) * pos_weight.cpu()
                bce_loss_n = nn.BCEWithLogitsLoss(pos_weight=pos_weight).cuda()
            else:
                bce_loss_n = nn.BCEWithLogitsLoss().cuda()

            # compute loss
            bce_loss_node = bce_loss_n(node_logits.squeeze(), is_object.squeeze())
            log_dict['train bce loss node'] = bce_loss_node.item()
            loss += bce_loss_node

            # get accuracy
            logits_rounded = node_logits.clone().detach().squeeze()
            logits_rounded[logits_rounded>0.5] = 1
            logits_rounded[logits_rounded<=0.5] = 0
            correct = torch.sum(logits_rounded == is_object.squeeze())
            node_accuracy = correct/logits_rounded.shape[0]
            log_dict['train node accuracy'] = node_accuracy.item()

        return loss, log_dict
    
    def eval(self, logits, data, edge_index):
        edge_logits, node_logits = logits
        loss = 0
        log_dict = dict()
        if self.bce_loss:
            edge_logits = edge_logits.cuda()
            point_instances = data.point_instances.unsqueeze(
                0) == data.point_instances.unsqueeze(0).T
            point_instances[data.point_instances == 0, :] = False
            point_instances = point_instances.cuda()
            point_instances = point_instances[
                edge_index[0, :], edge_index[1, :]].type(torch.FloatTensor).cuda()
            bce_loss_edge = torch.nn.functional.binary_cross_entropy(
                edge_logits.squeeze(), point_instances.squeeze())
            log_dict['eval bce loss edge'] = bce_loss_edge.item()
            loss += bce_loss_edge

            # get accuracy
            logits_rounded = edge_logits.squeeze()
            logits_rounded[logits_rounded>0.5] = 1
            logits_rounded[logits_rounded<=0.5] = 0
            correct = torch.sum(logits_rounded == point_instances.squeeze())
            edge_accuracy = correct/logits_rounded.shape[0]
            log_dict['eval edge accuracy'] = edge_accuracy.item()
        
        if self.node_loss:
            is_object = data.point_instances != 0
            is_object = is_object.type(torch.FloatTensor).cuda()
            # compute loss
            bce_loss_node = torch.nn.functional.binary_cross_entropy(
                node_logits.squeeze(), is_object.squeeze())
            log_dict['eval bce loss node'] = bce_loss_node.item()

            # get accuracy
            logits_rounded = node_logits.squeeze()
            logits_rounded[logits_rounded>0.5] = 1
            logits_rounded[logits_rounded<=0.5] = 0
            correct = torch.sum(logits_rounded == is_object.squeeze())
            node_accuracy = correct/logits_rounded.shape[0]
            log_dict['eval node accuracy'] = node_accuracy.item()

        return loss, log_dict
