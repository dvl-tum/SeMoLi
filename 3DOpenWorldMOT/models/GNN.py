from torch_geometric.data import Data as PyGData
import os
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
import models.losses
import math
import sklearn.metrics
# import torchvision 
from .losses import sigmoid_focal_loss
from torch import multiprocessing as mp
import pickle
import wandb
import copy
import torch.utils.checkpoint as checkpoint


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
rgb_colors[0] = (0, 0, 1)


logger = logging.getLogger("Model.GNN")


def _addindent(s_, numSpaces):
    s = s_.split('\n')
    # don't do anything for single-line stuff
    if len(s) == 1:
        return s_
    first = s.pop(0)
    s = [(numSpaces * ' ') + line for line in s]
    s = '\n'.join(s)
    s = first + '\n' + s
    return s


class ClusterLayer(MessagePassing):
    def __init__(self, in_channel_node, in_channel_edge, out_channel_node, out_channel_edge, use_batchnorm=True, use_layernorm=False, use_drop=False, drop_p=0.4, skip_node_update=False):
        super().__init__(aggr='mean')
        # get edge mlp
        self.edge_mlp = torch.nn.Linear(in_channel_node * 2 + in_channel_edge, out_channel_edge)
        # get edge relu, bn, drop
        self.edge_relu = nn.ReLU(inplace=True)
        self.edge_batchnorm = nn.BatchNorm1d(out_channel_edge) \
            if use_batchnorm else use_batchnorm
        self.edge_layernorm = nn.LayerNorm(out_channel_edge) \
            if use_layernorm else use_layernorm
        self.edge_drop = nn.Dropout(p=drop_p) if use_drop else use_drop
        
        self.skip_node_update = skip_node_update
        if not self.skip_node_update:
            #get node mlp
            self.node_mlp = torch.nn.Linear(in_channel_node + out_channel_edge, out_channel_node)

            # get node relu, bn, drop
            self.node_relu = nn.ReLU(inplace=True)
            self.node_batchnorm = nn.BatchNorm1d(out_channel_node) \
                if use_batchnorm else use_batchnorm
            self.node_layernorm = nn.LayerNorm(out_channel_node) \
                if use_layernorm else use_layernorm
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
        if self.edge_layernorm:
            edge_attr = self.edge_layernorm(edge_attr)
        if self.edge_drop:
            edge_attr = self.edge_drop(edge_attr)
        return edge_attr

    def forward(self, node_attr, edge_index, edge_attr):
        # edge_updater_type: (alpha: OptPairTensor, edge_attr: OptTensor)
        edge_attr = self.edge_updater(edge_attr, node_attr, edge_index)
        
        if not self.skip_node_update:
            # propagate_type: (x: OptPairTensor, alpha: Tensor)
            node_attr = self.propagate(edge_index, node_attr=node_attr, edge_attr=edge_attr)
        return node_attr, edge_index, edge_attr
    
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
        x = self.node_relu(self.node_mlp(tmp))
        if self.node_batchnorm:
            x = self.node_batchnorm(x)
        if self.node_layernorm:
            x = self.node_layernorm(x)
        if self.node_drop:
            x = self.node_drop(x)
        return x

    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'
        main_str += ')'
        return main_str


def simplediff(a, b):
    return b-a


def get_mean_min_max(_in, num_edges):
    _in = _in.view(num_edges, -1)
    _in = torch.vstack([
        _in.min(dim=-1).values,
        _in.max(dim=-1).values,
        _in.mean(dim=-1)]).T
    return _in

def get_per_time_traj_diff(traj, edge_index, pos_dim):
    a = traj.view(traj.shape[0], -1, pos_dim)[edge_index[0]]
    a = a.view(edge_index.shape[1], -1)

    b = traj.view(traj.shape[0], -1, pos_dim)[edge_index[1]]
    b = b.view(edge_index.shape[1], -1)

    a = a.view(-1, pos_dim)
    b = b.view(-1, pos_dim)
    return torch.nn.PairwiseDistance(p=2)(a, b)

def get_per_time_pos_traj_diff(traj, pos, edge_index, pos_dim):
    a = traj.view(traj.shape[0], -1, pos_dim)[edge_index[0]]+pos[edge_index[0]].unsqueeze(1)
    a = a.view(edge_index.shape[1], -1)
    b = traj.view(traj.shape[0], -1, pos_dim)[edge_index[1]]+pos[edge_index[1]].unsqueeze(1)
    b = b.view(edge_index.shape[1], -1)
    a = a.view(-1, pos_dim)
    b = b.view(-1, pos_dim)
    return torch.nn.PairwiseDistance(p=2)(a, b)

def get_per_time_vel_diff(traj, time, _batch, dataset, edge_index, pos_dim):
    a = traj.view(traj.shape[0], -1, pos_dim)
    diff_time = time[_batch, 1:] - time[_batch, :-1]
    if 'argo' in dataset:
        diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
    else:
        diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)
    
    # get dx/dt of all nodes
    a = (a[:, 1:, :] - a[:, :-1, :])/diff_time
    # get dx/dt for sending and receiving
    a = a[edge_index[0]]
    b = b[edge_index[1]]
    # get diff between and compute norm to get magnitude
    return torch.linalg.norm(simplediff(a - b), dim=-1)


class SevInpSequential(nn.Sequential):

    def __init__(self, gradient_checkpointing, layers):
        super().__init__(*layers)
        self.gradient_checkpointing = gradient_checkpointing

    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                if self.gradient_checkpointing:
                    inputs = checkpoint.checkpoint(module, *inputs, use_reentrant=False)
                else:
                    inputs = module(*inputs)
            else:
                if self.gradient_checkpointing:
                    inputs = checkpoint.checkpoint(module, inputs, use_reentrant=False)
                else:
                    module(inputs)
        return inputs


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
            dataset='waymo',
            layer_sizes_edge=None,
            layer_sizes_node=None,
            ignore_stat_edges=0,
            ignore_stat_nodes=0,
            filter_edges=-1,
            node_loss=True,
            layer_norm=False,
            batch_norm=False,
            drop_out=False,
            augment=False,
            rank=0,
            gradient_checkpointing=False):
        super().__init__(aggr='mean')
        self.k = k
        self.k_eval = k_eval
        self.r = r
        self.graph = graph
        self.edge_attr = edge_attr
        self.node_attr = node_attr
        self.graph_construction = graph_construction
        self.use_node_score = use_node_score * node_loss
        self.clustering = clustering
        edge_dim = 0
        if '_DP_' in self.edge_attr:
            edge_dim += pos_channels
        if '_DT_' in self.edge_attr:
            edge_dim += pos_channels
        if '_DTDP_' in self.edge_attr:
            edge_dim += pos_channels + traj_channels
        if '_DPT_' in self.edge_attr:
            edge_dim += traj_channels
        if '_PTDT_' in self.edge_attr:
            edge_dim += int(traj_channels / pos_channels)
        if '_MMMDTT_' in self.edge_attr:
            edge_dim += 3
        if '_PTDPT_' in self.edge_attr:
            edge_dim += int(traj_channels / pos_channels)
        if '_MMMDPTT_' in self.edge_attr:
            edge_dim += 3
        if '_MMMDV_' in self.edge_attr:
            edge_dim += 3
        
        # get node mlp
        self.node_attr = node_attr
        node_dim = 0
        if '_P_' in self.node_attr:
            node_dim += pos_channels
        if '_T_' in self.node_attr:
            node_dim += traj_channels
        if '_PT_' in self.node_attr:
            node_dim += traj_channels
        if '_MMMV_' in self.node_attr:
            node_dim += 3
        
        layers = list()

        _node_dim = node_dim
        _edge_dim = edge_dim
        if layer_sizes_node is None:
            layer_sizes_node = {'l_1': node_dim}
            layer_sizes_edge = {'l_1': edge_dim}
        for j, (node_dim_hid, edge_dim_hid) in enumerate(zip(layer_sizes_node.values(), layer_sizes_edge.values())):
            if j == len(layer_sizes_node) -1 and not self.use_node_score:
                skip_node_update = True
            else:
                skip_node_update = False
            layers.append(ClusterLayer(
                in_channel_node=_node_dim,
                in_channel_edge=_edge_dim,
                out_channel_node=node_dim_hid,
                out_channel_edge=edge_dim_hid,
                use_batchnorm=batch_norm,
                use_layernorm=layer_norm,
                use_drop=drop_out,
                skip_node_update=skip_node_update))
            _node_dim = node_dim_hid
            _edge_dim = edge_dim_hid
        '''layers.append(ClusterLayer(
            in_channel_node=_node_dim,
            in_channel_edge=_edge_dim,
            out_channel_node=node_dim,
            out_channel_edge=edge_dim))'''
        self.layers = SevInpSequential(gradient_checkpointing, layers)

        self.final = nn.Linear(_edge_dim, 1)
        if self.use_node_score:
            self.final_node = nn.Linear(_node_dim, 1)
        self.sigmoid = torch.nn.Sigmoid()
        # self.sigmoid = torch.nn.Tanh()
        self.cut_edges = cut_edges
        self.augment = augment
        self.min_samples = min_samples
        self.do_visualize = do_visualize
        self.my_graph = my_graph
        self.oracle_node = oracle_node
        self.oracle_edge = oracle_edge
        self.ignore_stat_edges = ignore_stat_edges
        self.ignore_stat_nodes = ignore_stat_nodes
        self.filter_edges = filter_edges
        self.dataset = dataset
        self.gradient_checkpointing = gradient_checkpointing

        self.opts = rama_py.multicut_solver_options("PD")
        self.opts.sanitize_graph = True
        self.opts.verbose = False

        self.rank = rank

    def initial_edge_attributes(self, x1, x2, edge_index, point_normals=None, distance='euclidean', timestamps=None, batch=None):
        """
        DP = diffpos
        DT = difftraj
        DTDP = difftraj_diffpos
        DPT = diffpostraj
        PTDT = pertime_difftraj
        MMMDTT = min_mean_max_difftrajtime
        PTDPT = pertime_diffpostraj
        MMMDPTT = min_mean_max_diffpostrajtime
        MMMDV = min_mean_max_diffvelocity
        """
        num_edges = edge_index.shape[1]
        num_objexts = x2.shape[0]
        pos_dim = x2.shape[1]
        traj_dim = x1.shape[1]

        edge_attr = list()
        edge_dim = 0
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if '_DP_' in self.edge_attr:
            a = x2[edge_index[0]]
            b = x2[edge_index[1]]
            edge_attr.append(simplediff(a, b))
            edge_dim += pos_dim

        if '_DT_' in self.edge_attr:
            a = x1[edge_index[0]]
            b = x1[edge_index[1]]
            edge_attr.append(simplediff(a, b))
            edge_dim += traj_dim

        if '_DTDP_' in self.edge_attr:
            a = torch.stack([x2[edge_index[0]], x1[edge_index[0]]])
            b = torch.stack([x2[edge_index[1]], x1[edge_index[1]]])
            edge_attr.append(simplediff(a, b))
            edge_dim += pos_dim + traj_dim

        if '_DPT_' in self.edge_attr:
            a = x2[edge_index[0]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[0]]
            b = x2[edge_index[1]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[1]]
            edge_attr.append(simplediff(a, b))
            edge_dim += traj_dim

        if '_PTDT_' in self.edge_attr:
            edge_attr.append(get_per_time_traj_diff(x1, edge_index, pos_dim))
            edge_dim += int(traj_dim / pos_dim)
        
        if '_MMMDTT_' in self.edge_attr:
            edge_attr.append(get_mean_min_max(get_per_time_traj_diff(x1, edge_index, pos_dim), num_edges))
            edge_dim += 3

        if '_PTDPT_' in self.edge_attr:
            edge_attr.append(get_per_time_pos_traj_diff(x1, x2, edge_index, pos_dim))
            edge_dim += int(traj_dim / 3)
        
        if '_MMMDPTT_' in self.edge_attr:
            edge_attr.append(get_mean_min_max(get_per_time_pos_traj_diff(x1, x2, edge_index, pos_dim), num_edges))
            edge_dim += 3
        
        if '_MMMDV_' in self.edge_attr:
            edge_attr.append(get_mean_min_max(get_per_time_vel_diff(x1, timestamps, batch, self.dataset, edge_index, pos_dim), num_edges))
            edge_dim += 3
        
        edge_attr = torch.stack(edge_attr).squeeze()

        return edge_attr, edge_dim
    
    def initial_node_attributes(self, x1, x2, _type, point_normals=None, timestamps=None, batch=None):
        """
        x1 = N x 3
        x2 = N x 3*T

        ----

        P = pos
        T = traj
        MTOT = mean_traj_over_time
        PTS = pos_traj_stacked
        PT = postraj
        MDOT = mean_dist_over_time
        MMMV = min_mean_max_vel
        """
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        node_attr = list()
        num_objects = x1.shape[0]
        traj_dim = x1.shape[1]
        pos_dim = x2.shape[1]
        node_dim = 0
        # JUST FOR GRAPH CONST
        if '_MTOT_' in _type:
            node_attr.append(x1.view(traj_dim, -1, pos_dim))
        if  '_MDOT_' in _type:
            node_attr.append(x1.view(traj_dim, -1, pos_dim)+x2.unsqueeze(1))
        
        # FOR GRAPH CONST AND NODES
        if '_P_' in _type:
            node_attr.append(x2)
            node_dim += pos_dim
        if '_T_' in _type:
            node_attr.append(x1)
            node_dim += traj_dim
        if '_PT_' in _type:
            node_attr.append((x1.view(traj_dim, -1, pos_dim)+x2.unsqueeze(1)).view(traj_dim, -1))
            node_dim += traj_dim
        if '_MMMV_' in _type:
            _node_attr = x1.view(num_objects, -1, pos_dim)
            diff_time = timestamps[batch, 1:] - timestamps[batch, :-1]
            if 'argo' in self.dataset:
                diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
            else:
                diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)
            _node_attr = _node_attr[:, 1:, :] - _node_attr[:, :-1, :]
            _node_attr = torch.linalg.norm(_node_attr, dim=-1)
            _node_attr = _node_attr / diff_time
            _node_attr = torch.vstack([
                _node_attr.min(dim=-1).values,
                _node_attr.max(dim=-1).values,
                _node_attr.mean(dim=-1)]).T
            node_attr.append(_node_attr)
            node_dim += 3
        
        node_attr = torch.stack(node_attr).squeeze()

        return node_attr, node_dim
    
    def get_graph(self, node_attr, r=5, max_num_neighbors=16, batch_idx=None, type='radius', metric='euclidean', batch=None, data=None):
        # my graph
        _idxs_0, _idxs_1 = list(), list()
        for ith, (start, end) in enumerate(zip(batch_idx[:-1], batch_idx[1:])):
            # iterate over frames in batch
            X = node_attr[start:end]
            # check edge_idx
            name = data['path'][ith].split('_')[-1]
            p = f'/workspace/result/all_egocomp_margin0.6_width25_{self.graph_construction}/{name}' 
            recompute = True
            if os.path.isfile(p):
                edge_idx = torch.load(p, map_location='cpu')['x'].to(self.rank)
                try:
                    # print(f'got it {p}')
                    has_len = len(edge_idx)
                    data['edge_index'] = edge_idx
                    if edge_idx.min() != 0:
                        edge_idx = edge_idx - edge_idx.min()
                        # d = PyGData(edge_idx)
                        # torch.save(d, p)
                    idxs_0 = edge_idx[0, :]
                    idxs_1 = edge_idx[1, :]
                    
                    idxs_0 += start
                    idxs_1 += start
                    _idxs_0.append(idxs_0)
                    _idxs_1.append(idxs_1)
                    
                    recompute = False
                except:
                    recompute = True
            
            if not recompute:
                continue

            '''
            # get indices up to max_num_neighbors per node --> knn neighbors
            num_neighbors = min(int(max_num_neighbors*1.5), dist.shape[0])
            idxs_0 = torch.tile(torch.arange(dist.shape[0]).unsqueeze(1).to(self.rank), (1, num_neighbors)).flatten()
            dist = torch.nn.PairwiseDistance(X[:, 0, :])
            idxs_1 = dist.topk(k=num_neighbors, dim=1, largest=False).indices.flatten()
            X_time_0 = X[idxs_0]
            X_time_1 = X[idxs_1]
            '''

            # get distances between nodes
            if self.graph_construction == 'traj' or self.graph_construction == 'pos' or self.graph_construction == 'MMMV' or self.graph_construction == 'postraj':
                dist = torch.from_numpy(sklearn.metrics.pairwise_distances(X.cpu().numpy(), metric=metric)).to(self.rank)
            
            else:
                '''
                # get indices up to max_num_neighbors per node --> knn neighbors
                num_neighbors = min(int(max_num_neighbors*1.5), dist.shape[0])
                idxs_0 = torch.tile(torch.arange(dist.shape[0]).unsqueeze(1).to(self.rank), (1, num_neighbors)).flatten()
                dist = torch.from_numpy(sklearn.metrics.pairwise_distances(X[:, 0, :].cpu().numpy(), metric=metric)).to(self.rank)
                idxs_1 = dist.topk(k=num_neighbors, dim=1, largest=False).indices.flatten()
                X_time_0 = X[idxs_0]
                X_time_1 = X[idxs_1]
                
                # following two lines are faster but cuda oom
                dist = torch.cdist(X_time_0, X_time_1)
                dist = dist.mean(dim=0)
                '''
                dist = torch.zeros(X.shape[0], X.shape[0]).to(self.rank)
                for t in range(X.shape[1]):
                    dist += torch.cdist(X[:, t, :].unsqueeze(0),X[:, t, :].unsqueeze(0)).squeeze()
                dist = dist / X.shape[1]

            # set diagonal elements to 0to have no self-loops
            dist.fill_diagonal_(100)

            # get indices up to max_num_neighbors per node --> knn neighbors
            num_neighbors = min(max_num_neighbors, dist.shape[0])
            idxs_0 = torch.tile(torch.arange(dist.shape[0]).unsqueeze(1).to(self.rank), (1, num_neighbors)).flatten()
            idxs_1 = dist.topk(k=num_neighbors, dim=1, largest=False).indices.flatten()

            # if radius graph, filter nodes that are within radius 
            # but don't exceed max num neighbors
            if type == 'radius':
                dist = dist[idxs_0, idxs_1]
                idx = torch.where(dist<r)[0]
                idxs_0, idxs_1 = idxs_0[idx], idxs_1[idx]
            
            # if not os.path.isfile(p):
            #     d = PyGData(torch.vstack([idxs_0, idxs_1]))
            #     os.makedirs(os.path.dirname(p), exist_ok=True)
            #     torch.save(d, p)
            idxs_0 += start
            idxs_1 += start            
            _idxs_0.append(idxs_0)
            _idxs_1.append(idxs_1)

        _idxs_0 = torch.hstack(_idxs_0)
        _idxs_1 = torch.hstack(_idxs_1)

        edge_index = torch.vstack([_idxs_0, _idxs_1])

        return edge_index

    def forward(self, data, eval=False, use_edge_att=True, name='General', corr_clustering=False):
        '''
        clustering: 'heuristic' / 'correlation'
        '''
        data = data.to(self.rank)
        batch_idx = data._slice_dict['pc_list']
        traj = data['traj']
        if traj.shape[0] == 0:
            return [None, None], list(), None, None
        
        traj = traj.view(traj.shape[0], -1)
        pc = data['pc_list']
        if 'pc_normals' in [k for k in data.keys]:
            point_normals = data['pc_normals']
        else:
            point_normals = None
        
        node_attr, node_dim = self.initial_node_attributes(traj, pc, self.node_attr, point_normals, data['timestamps'], data['batch'])
        if self.node_attr == self.graph_construction:
            graph_attr = node_attr
        else:
            graph_attr, _ = self.initial_node_attributes(traj, pc, self.graph_construction, point_normals, data['timestamps'], data['batch'])
        
        # get edges using knn graph (for computational feasibility)
        k = self.k if not eval else self.k_eval
        if 'edge_index' not in data.keys:
            if self.graph == 'knn':
                if self.my_graph and len(graph_attr.shape) != 2:
                    edge_index = self.get_graph(
                        graph_attr, self.r, max_num_neighbors=k, batch_idx=batch_idx, type='knn', batch=data['batch'], data=data)
                else:
                    edge_index = knn_graph(x=graph_attr, k=k, batch=data['batch'])
            elif self.graph == 'radius':
                if self.my_graph and len(graph_attr.shape) != 2:
                    edge_index = self.get_graph(
                        graph_attr, self.r, max_num_neighbors=k, batch_idx=batch_idx, type='radius', batch=data['batch'], data=data)
                else:
                    edge_index = radius_graph(graph_attr, self.r, data['batch'], max_num_neighbors=k)
            '''
            if self.my_graph and len(graph_attr.shape) == 2:
                from torch_geometric.data import Data as PyGData
                import os
                # DataBatch(pc_list=[40615, 3], traj=[40615, 25, 3], timestamps=[4, 25], point_categories_mov=[40615], point_instances_mov=[40615], point_categories=[40615], point_instances=[40615], log_id=[4], batch=[40615], path=[4], ptr=[5])
                for i, (start, end) in enumerate(zip(batch_idx[:-1], batch_idx[1:])):
                    name = data['path'][i].split('_')[-1]
                    p = f'/workspace/result/all_egocomp_margin0.6_width25_{self.graph_construction}/{name}'
                    if os.path.isfile(p):
                        continue
                    print('save')
                    sample_edge = edge_index[:, torch.logical_and(
                                edge_index[0, :] >= start,
                                edge_index[1, :] < end)]
                    d = PyGData(
                            sample_edge - sample_edge.min())
                    os.makedirs(os.path.dirname(p), exist_ok=True)
                    torch.save(d, p)
                '''
        else:
            edge_index = data['edge_index']
        
        # add negative edges to edge_index
        if not eval and self.augment:
            point_instances = data.point_instances.unsqueeze(
                0) == data.point_instances.unsqueeze(0).T
            same_graph = data['batch'].unsqueeze(0) == data['batch'].unsqueeze(0).T
            point_instances = torch.logical_and(point_instances, same_graph)
            # setting edges that do not belong to object to zero
            point_instances[data.point_instances == 0, :] = False
            point_instances[:, data.point_instances == 0] = False
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
                a = torch.cat(a).to(self.rank)
                b = torch.cat(b).to(self.rank)
                a, b = a[~point_instances[a, b]], b[~point_instances[a, b]]
                a, b = a[:missing_neg], b[:missing_neg]
            elif point_instances.shape[0]:
                missing_pos = -missing_neg
                for _ in range(max(math.ceil(missing_pos/point_instances.shape[0])*2, 1)):
                    a.append(torch.randperm(point_instances.shape[0]))
                    b.append(torch.randperm(point_instances.shape[0]))
                a = torch.cat(a).to(self.rank)
                b = torch.cat(b).to(self.rank)
                a, b = a[point_instances[a, b]], b[point_instances[a, b]]
                a, b = a[:missing_pos], b[:missing_pos]
            add_idxs = torch.stack([a, b]).to(self.rank)
            edge_index = torch.cat([edge_index.T, add_idxs.T]).T

        if edge_index.shape[1] == 0:
            print("EEEMPTY")
            return [None, None], torch.tensor(list(range(pc.shape[0]))), None
        
        edge_attr, edge_dim = self.initial_edge_attributes(traj, pc, edge_index, point_normals)

        edge_attr = edge_attr.float()
        node_attr = node_attr.float()
        
        node_attr, edge_index, edge_attr = self.layers(node_attr, edge_index, edge_attr)

        src, dst = edge_index
        # computes per edge index by computing dot product between node features
        if not use_edge_att:
            score = (node_attr[src] * node_attr[dst]).sum(dim=-1)
        # directly uses edge attirbutes
        else:
            score = self.final(edge_attr)
        if self.use_node_score:
            node_score = self.final_node(node_attr)
        else:
            node_score = None

        if torch.any(torch.isnan(score)):
            print('Having nan during forward pass...')
            return [torch.nan, torch.nan], edge_index, None
        '''
        pc_vis = pc[:batch_idx[1]]
        edge_mask = torch.logical_or(
            torch.logical_and(edge_index[0] >= batch_idx[0], edge_index[1] < batch_idx[1]),
            torch.logical_and(edge_index[1] >= batch_idx[0], edge_index[0] < batch_idx[1]))
        self.visualize(torch.arange(pc_vis.shape[0]), edge_index[:, edge_mask], pc_vis, torch.ones(pc_vis.shape[0]), '00000', mode='before', name='General')
        quit() 
        '''
        if eval and corr_clustering:
            _score = self.sigmoid(score)
            if self.use_node_score:
                _node_score = self.sigmoid(node_score)
            else:
                _node_score = None

            if self.clustering == 'correlation':
                multiprocessing = False
                data_loader = enumerate(zip(batch_idx[:-1], batch_idx[1:]))
                rama_cuda = rama_py.rama_cuda
                all_clusters = list()
                if multiprocessing:
                    pickle.dumps(rama_cuda)
                    self.args = edge_index, _node_score, _score, data, score, node_score, pc, rama_cuda, name
                    with mp.Pool() as pool:
                        clusters = pool.map(self.corr_clustering, data_loader, chunksize=None)
                        all_clusters.append(clusters)
                else:
                    self.args = edge_index, _node_score, _score, data, score, node_score, pc, rama_cuda, name
                    for iter_data in data_loader:
                        clusters = self.corr_clustering(iter_data)
                        all_clusters.append(clusters)
            else:
                print('Invalid clustering choice')
                quit()

            return [score, node_score], all_clusters, edge_index, None
        elif eval:
            return [score, node_score], [[]*len(batch_idx[:-1])], edge_index, None
        
        return [score, node_score], edge_index, None
    
    def corr_clustering(self, iter_data):
        i, (start, end) = iter_data
        edge_index, _node_score, _score, data, score, node_score, pc, rama_cuda, name = self.args

        edge_mask = torch.logical_or(
            torch.logical_and(edge_index[0] >= start, edge_index[1] < end),
            torch.logical_and(edge_index[1] >= start, edge_index[0] < end))
        graph_edge_index = edge_index[:, edge_mask]
        src, dst = graph_edge_index
        # graph_edge_index = graph_edge_index - start
        if self.use_node_score:
            graph_node_score = _node_score[start:end]
        graph_edge_score = _score[edge_mask]

        if self.oracle_edge:
            graph_edge_score[data['point_instances'][src] == data['point_instances'][dst]] = 1
            graph_edge_score[data['point_instances'][src] != data['point_instances'][dst]] = 0
            graph_edge_score[data['point_instances'][src] <= 0] = 0
            graph_edge_score[data['point_instances'][dst] <= 0] = 0

            score[edge_mask] = graph_edge_score
            score[score == 0] = -10
            score[score == 1] = 10

        if self.oracle_node and self.use_node_score:
            graph_node_score[data['point_categories'][start:end]>0] = 1
            graph_node_score[data['point_categories'][start:end]<=0] = 0

            node_score[start:end] = graph_node_score
            node_score[node_score == 0] = -10
            node_score[node_score == 1] = 10

        # filter out edges with very low score already
        if self.filter_edges > 0:
            graph_edge_index = graph_edge_index[:, (graph_edge_score > self.filter_edges).squeeze()]
            graph_edge_score = graph_edge_score[(graph_edge_score > self.filter_edges).squeeze()]
        
        graph_edge_index = graph_edge_index - start.item()

        if self.use_node_score:
            graph_edge_score = graph_edge_score[torch.logical_and(
                graph_node_score[graph_edge_index[0]] > self.use_node_score, 
                graph_node_score[graph_edge_index[1]] > self.use_node_score).squeeze()]
            graph_edge_index = graph_edge_index[:, torch.logical_and(
                graph_node_score[graph_edge_index[0]] > self.use_node_score, 
                graph_node_score[graph_edge_index[1]] > self.use_node_score).squeeze()]

        # map nodes
        edges = torch.unique(graph_edge_index)
        mapping = torch.ones(end.item()-start.item()) * - 1
        mapping = mapping.int()
        mapping[edges] = torch.arange(edges.shape[0]).int()
        mapping = mapping.to(self.rank)
        _edge_index = graph_edge_index
        _edge_index[0, :] = mapping[graph_edge_index[0, :]]
        _edge_index[1, :] = mapping[graph_edge_index[1, :]]

        try:
            rama_out = rama_cuda(
                [e[0] for e in _edge_index.T.cpu().numpy()],
                [e[1] for e in _edge_index.T.cpu().numpy()], 
                (graph_edge_score.cpu().numpy()*2)-1,
                self.opts)
            mapped_clusters = torch.tensor(rama_out[0]).to(self.rank).int()
        except:
            mapped_clusters = torch.arange(edges.shape[0]).to(self.rank).int()

        # map back 
        _edge_index[0, :] = edges[_edge_index[0, :]]
        _edge_index[1, :] = edges[_edge_index[1, :]]
        clusters = torch.ones(end.item()-start.item()) * - 1
        clusters = clusters.int().to(self.rank)
        clusters[edges] = mapped_clusters
        clusters = clusters.cpu().numpy().tolist()

        # filter out nodes thatare classified as non-objects
        '''if self.use_node_score > 0:
            clusters = torch.tensor(clusters)
            clusters[(graph_node_score.cpu() < self.use_node_score).squeeze()] = -1
            clusters = clusters.numpy()'''

        _clusters = defaultdict(list)
        for iter, c in enumerate(clusters):
            _clusters[c].append(iter)

        cluster_assignment_new = dict()
        for c, node_list in _clusters.items():
            if len(node_list) < self.min_samples:
                for n in node_list:
                    cluster_assignment_new[n] = -1
            else:
                for n in node_list:
                    cluster_assignment_new[n] = c

        clusters = np.array([cluster_assignment_new[k] for k in sorted(cluster_assignment_new.keys())])
        
        return clusters

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
            edge_indices = torch.tensor(edge_indices).to(self.rank).T

        # take only x and y position
        pos = pos[:, :-1]

        # make graph
        pos = {n.item(): p for n, p in zip(nodes, pos.cpu().numpy())}
        G = nx.Graph()
        G.add_nodes_from(nodes.numpy())
        G.add_edges_from(edge_indices.T.cpu().numpy())

        colors = [(0.999, 0.999, 0.999)] * nodes.shape[0]
        '''
        for i, (c, node_list) in enumerate(clusters.items()):
            for node in node_list:
                colors[node] = rgb_colors[i]
            col_dict[c] = rgb_colors[i]
        '''

        # save graph
        labels = {n.item(): str(n.item()) for n in nodes}
        plt.figure(figsize=(50, 50))
        nx.draw_networkx_edges(G, pos, width=3)
        nx.draw_networkx_nodes(G, pos, node_size=2, node_color=colors)
        # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_color='red')
        plt.axis("off")
        plt.savefig(f'../../../vis_graph/{name}/{timestamp}_{mode}.png', bbox_inches='tight', dpi=300)
        plt.close()
         
    def __repr__(self):
        # We treat the extra repr like the sub-module, one item per line
        extra_lines = []
        extra_repr = self.extra_repr()
        # empty string will be split into list ['']
        if extra_repr:
            extra_lines = extra_repr.split('\n')
        child_lines = []
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = _addindent(mod_str, 2)
            child_lines.append('(' + key + '): ' + mod_str)
        lines = extra_lines + child_lines

        main_str = self._get_name() + '('
        if lines:
            # simple one-liner info, which most builtin Modules will use
            if len(extra_lines) == 1 and not child_lines:
                main_str += extra_lines[0]
            else:
                main_str += '\n  ' + '\n  '.join(lines) + '\n'

        main_str += ')'
        return main_str


class GNNLoss(nn.Module):
    def __init__(
            self,
            bce_loss=False,
            node_loss=False,
            focal_loss_node=True,
            focal_loss_edge=True,
            alpha_node=0.25,
            alpha_edge=0.25,
            gamma_node=2,
            gamma_edge=2,
            rank=0,
            edge_weight=1,
            node_weight=1,
            ignore_stat_edges=0,
            ignore_stat_nodes=0,
            ignore_edges_between_background=0,
            classification_is_moving_node=0,
            use_node_score=False) -> None:
        super().__init__()
        
        self.bce_loss = bce_loss
        self.node_loss = node_loss
        self.focal_loss_node = focal_loss_node
        self.focal_loss_edge = focal_loss_edge
        self.edge_weight = edge_weight
        self.node_weight = node_weight
        self.alpha_edge = alpha_edge
        self.alpha_node = alpha_node
        self.gamma_node = gamma_node
        self.gamma_edge = gamma_edge
        self.max_iter = 2000
        self.rank = rank
        self.ignore_stat_edges = ignore_stat_edges
        self.ignore_stat_nodes = ignore_stat_nodes
        self.ignore_edges_between_background = ignore_edges_between_background
        self.classification_is_moving_node = classification_is_moving_node
        assert self.classification_is_moving_node != self.ignore_stat_nodes, "Can only either ignore static objects or classify as moving object or not a moving object"
        self.sigmoid = torch.nn.Sigmoid()
        self.use_node_score = use_node_score

        if not self.focal_loss_node:
            self._node_loss = nn.BCEWithLogitsLoss().to(self.rank)
        else:
            self._node_loss = sigmoid_focal_loss
        
        if not self.focal_loss_edge:
            self._edge_loss = nn.BCEWithLogitsLoss().to(self.rank)
        else:
            self._edge_loss = sigmoid_focal_loss
        
        self.test = nn.BCEWithLogitsLoss().to(self.rank)

    def forward(self, logits, data, edge_index, weight=False, weight_node=True, mode='train'):
        hist_node, hist_edge = None, None
        edge_logits, node_logits = logits
        loss = 0
        log_dict = dict()
        same_graph = data['batch'][edge_index[0, :]] == data['batch'][edge_index[1, :]]
        
        batch_idx = data._slice_dict['pc_list']
        idxs = list()
        for i in range(batch_idx.shape[0]-1):
            sample = data['batch'][batch_idx[i]:batch_idx[i+1]]
            idxs.append(torch.stack(torch.where(sample == sample.unsqueeze(1))) + batch_idx[i])
        idxs = torch.cat(idxs, dim=1)
        
        all_prediction = torch.sparse_coo_tensor(
                idxs,
                torch.ones(idxs.shape[1]).to(idxs.device)*-1,
                (data['batch'].shape[0], data['batch'].shape[0])) 
        
        if self.bce_loss:
            point_instances = data.point_instances
            point_categories = data.point_categories[edge_index[0, :]]
            point_categories1 = data.point_categories[edge_index[1, :]]

            # get bool edge mask
            point_instances = point_instances[edge_index[0, :]] == point_instances[edge_index[1, :]]
            
            # keep only edges that belong to same graph (for batching opteration)
            point_instances = torch.logical_and(point_instances, same_graph).bool()
            
            # setting edges that do not belong to object to zero
            # --> instance 0 is no object
            point_instances[data.point_instances[edge_index[0, :]] == 0] = False
            point_instances = point_instances.to(self.rank)
            
            point_mask = torch.zeros(point_instances.shape[0], dtype=bool).to(self.rank)
            # if ignoring predictions for static edges in loss, get static edge filter
            if (self.ignore_stat_edges and mode == 'train') or (self.use_node_score and mode != 'train'):
                point_mask = torch.logical_or(
                    torch.logical_and(
                        torch.logical_and(
                            ~(data.point_instances_mov[edge_index[0, :]] != 0), 
                            data.point_instances[edge_index[0, :]] != 0),
                        torch.logical_and(
                            ~(data.point_instances_mov[edge_index[1, :]] != 0), 
                            data.point_instances[edge_index[1, :]] != 0)),
                    point_mask)
                
                # filter moving objects from all predictions
                # all_prediction = all_prediction.coalesce()
                sparse_idx = all_prediction._indices()
                idx_mask = torch.logical_and(
                        ~(data.point_instances_mov[sparse_idx[0, :]] != 0), 
                        data.point_instances[sparse_idx[0, :]] != 0)
                idx_mask = torch.logical_and(
                        idx_mask, torch.logical_and(
                            ~(data.point_instances_mov[sparse_idx[1  , :]] != 0), 
                            data.point_instances[sparse_idx[1, :]] != 0))
                all_prediction = torch.sparse_coo_tensor(
                        sparse_idx[:, ~idx_mask],
                        all_prediction._values()[~idx_mask],
                        all_prediction.size())

            if (self.ignore_edges_between_background and mode == 'train') or (self.use_node_score and mode != 'train'):
                # setting edges that do not belong to object to zero
                # --> instance 0 is no object
                point_mask = torch.logical_or(
                    torch.logical_and(
                        data.point_instances[edge_index[0, :]] == 0,
                        data.point_instances[edge_index[1, :]] == 0),
                    point_mask)
                
                # filter background edges from all predictions
                # all_prediction = all_prediction.coalesce()
                sparse_idx = all_prediction._indices()
                idx_mask = torch.logical_and(
                        data.point_instances[sparse_idx[0, :]] == 0,
                        data.point_instances[sparse_idx[1, :]] == 0)
                all_prediction = torch.sparse_coo_tensor(
                        sparse_idx[:, ~idx_mask],
                        all_prediction._values()[~idx_mask],
                        all_prediction.size())
            
            if self.use_node_score and self.node_loss and mode != 'train':
                logits_rounded_node = self.sigmoid(node_logits.clone().detach()).squeeze()
                point_mask = torch.logical_or(
                    torch.logical_or(
                        logits_rounded_node[edge_index[0, :]] < self.use_node_score,
                        logits_rounded_node[edge_index[1, :]] < self.use_node_score),
                    point_mask)
                
                # filter edges that would be filtered out by node score
                # all_prediction = all_prediction.coalesce()
                sparse_idx = all_prediction._indices()
                idx_mask = torch.logical_or(
                        logits_rounded_node[sparse_idx[0, :]] < self.use_node_score,
                        logits_rounded_node[sparse_idx[1, :]] < self.use_node_score)
                all_prediction = torch.sparse_coo_tensor(
                        sparse_idx[:, ~idx_mask],
                        all_prediction._values()[~idx_mask],
                        all_prediction.size())

            # filter edge logits, point instances and point categories
            edge_logits = edge_logits[~point_mask]
            point_instances = point_instances[~point_mask].float()
            point_categories = point_categories[~point_mask]
            point_categories1 = point_categories1[~point_mask]
            edge_index = edge_index[:, ~point_mask]
            
            if edge_logits.shape[0] == 0:
                return None, None, None, None

            num_edge_pos, num_edge_neg = point_instances.sum(), (point_instances==0).sum()

            # COMPUTE LOSS
            if weight and not self.focal_loss_edge:
                # weight pos and neg samples
                num_pos = torch.sum((point_instances==1).float())
                num_neg = torch.sum((point_instances==0).float())
                pos_weight = num_neg/num_pos
                pos_weight = pos_weight.cpu()
                self._edge_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.rank)
            
            if not self.focal_loss_edge:
                bce_loss_edge = self._edge_loss(
                    edge_logits.squeeze(), point_instances.squeeze())
            else:
                bce_loss_edge = self._edge_loss(
                    edge_logits.squeeze(),
                    point_instances.squeeze(),
                    alpha=self.alpha_edge,
                    gamma=self.gamma_edge,
                    reduction="mean",)
            
            # LOG LOSS
            loss += self.edge_weight * bce_loss_edge
            #print(f'{mode} bce loss edge', bce_loss_edge.detach().item())
            log_dict[f'{mode} bce loss edge'] = torch.zeros(2).to(self.rank)
            log_dict[f'{mode} bce loss edge'][0] = bce_loss_edge.detach().item()
            log_dict[f'{mode} bce loss edge'][1] = 1

            # get accuracy
            logits_rounded = self.sigmoid(edge_logits.clone().detach()).squeeze()
            hist_edge = np.histogram(logits_rounded.cpu().numpy(), bins=10, range=(0., 1.))
            logits_rounded[logits_rounded>0.5] = 1
            logits_rounded[logits_rounded<=0.5] = 0
            # correct = logits_rounded == point_instances.squeeze()
            
            # comment out if not all
            print(all_prediction.shape)
            # all_prediction = all_prediction.coalesce()
            all_size = all_prediction.size()
            all_edges = all_prediction._indices()
            all_prediction = all_prediction._values()
            point_instances = data.point_instances[all_edges[0, :]] == data.point_instances[all_edges[1, :]]
            point_categories = data.point_categories[all_edges[0, :]]
            point_categories1 = data.point_categories[all_edges[1, :]]
            
            # set predictions to rounded logits
            logits_rounded[logits_rounded == 1] = 2
            all_prediction = torch.sparse_coo_tensor(
                    torch.hstack([all_edges, edge_index]),
                    torch.cat([all_prediction, logits_rounded]),
                    all_size)
            all_prediction = all_prediction.coalesce()
            all_prediction = all_prediction.values()
            all_prediction[all_prediction == -1] = 0

            # get correct mask
            correct = all_prediction == point_instances
            
            # overall
            if correct.shape[0]:
                log_dict[f'{mode} accuracy edge'] = torch.zeros(6).to(self.rank) 
                log_dict[f'{mode} accuracy edge'][0] = torch.sum(correct)/logits_rounded.shape[0]
                log_dict[f'{mode} accuracy edge'][1] = 1
            # negative edges
            if correct[point_instances==0].shape[0]:
                log_dict[f'{mode} accuracy edge'][2] = torch.sum(
                        correct[point_instances==0])/correct[point_instances==0].shape[0] 
                log_dict[f'{mode} accuracy edge'][3] = 1
            # positive edges
            if correct[point_instances==1].shape[0]:
                log_dict[f'{mode} accuracy edge'][4] = torch.sum(
                        correct[point_instances==1])/correct[point_instances==1].shape[0] 
                log_dict[f'{mode} accuracy edge'][5] = 1

            # per class accuracy:
            log_dict[f'{mode} accuracy edges connected to class'] = torch.zeros(65).to(self.rank)
            for c in torch.unique(point_categories):
                if correct[point_categories==c].shape[0]:
                    log_dict[f'{mode} accuracy edges connected to class'][2*c] = (torch.sum(
                            correct[point_categories==c])+torch.sum(
                            correct[point_categories1==c]))/(correct[point_categories1==c].shape[0]+correct[point_categories==c].shape[0])
                    log_dict[f'{mode} accuracy edges connected to class'][2*c+1] = 1
            log_dict[f'{mode} num edge pos'] = num_edge_pos
            log_dict[f'{mode} num edge neg'] = num_edge_neg

        if self.node_loss:
            # get if point is object
            is_object = data.point_instances != 0
            is_object = is_object.type(torch.FloatTensor).to(self.rank)
            object_class = data.point_categories

            # classify moving objects only or ignore static nodes for loss computation get filter
            if self.classification_is_moving_node or self.ignore_stat_nodes:
                is_object_stat = torch.logical_and(
                    ~(data.point_instances_mov != 0), data.point_instances != 0)
                is_object_stat = is_object_stat.to(self.rank)
                
                if self.ignore_stat_nodes:
                    # filter logits and object ground truth
                    is_object = is_object[~is_object_stat]
                    node_logits = node_logits[~is_object_stat]
                    object_class = object_class[~is_object_stat]
                if self.classification_is_moving_node:
                    is_object[is_object_stat] = False

            # weight pos and neg samples
            if weight_node and not self.focal_loss_node:
                num_pos = torch.sum((is_object==1).float())
                num_neg = torch.sum((is_object==0).float())
                pos_weight = num_neg/num_pos
                self._node_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.rank)
            
            # compute loss            
            if not self.focal_loss_node:
                bce_loss_node = self._node_loss(
                    node_logits.squeeze(), is_object.squeeze())
            else:
                bce_loss_node = self._node_loss(
                    node_logits.squeeze(),
                    is_object.squeeze(),
                    alpha=self.alpha_node,
                    gamma=self.gamma_node,
                    reduction="mean",)

            # log loss
            log_dict[f'{mode} bce loss node'] = torch.zeros(2).to(self.rank)
            log_dict[f'{mode} bce loss node'][0] = bce_loss_node.detach().item()
            log_dict[f'{mode} bce loss node'][1] = 1
            loss += self.node_weight * bce_loss_node

            # get accuracy
            logits_rounded_node = self.sigmoid(node_logits.clone().detach()).squeeze()
            hist_node = np.histogram(logits_rounded_node.cpu().numpy(), bins=10, range=(0., 1.))
            logits_rounded_node[logits_rounded_node>0.5] = 1
            logits_rounded_node[logits_rounded_node<=0.5] = 0
            correct = logits_rounded_node == is_object.squeeze()

            # Overall accuracy
            log_dict[f'{mode} accuracy node'] = torch.zeros(6).to(self.rank)
            log_dict[f'{mode} accuracy node'][0] = torch.sum(correct)/logits_rounded_node.shape[0]
            log_dict[f'{mode} accuracy node'][1] = 1
            # negative nodes
            if correct[is_object==0].shape[0]:
                log_dict[f'{mode} accuracy node'][2] = torch.sum(
                        correct[is_object==0])/correct[is_object==0].shape[0] 
                log_dict[f'{mode} accuracy node'][3] = 1
            # negative nodes
            if correct[is_object==1].shape[0]:
                log_dict[f'{mode} accuracy node'][4] = torch.sum(
                        correct[is_object==1])/correct[is_object==1].shape[0]
                log_dict[f'{mode} accuracy node'][5] = 1
            
            # per class
            log_dict[f'{mode} accuracy nodes of class'] = torch.zeros(65).to(self.rank)
            for c in torch.unique(object_class):
                if correct[object_class==c].shape[0]:
                    log_dict[f'{mode} accuracy nodes of class'][2*c] = torch.sum(
                            correct[object_class==c])/correct[object_class==c].shape[0]
                    log_dict[f'{mode} accuracy nodes of class'][2*c+1] = 1

            num_node_pos, num_node_neg = is_object.sum(), (is_object==0).sum()
            log_dict[f'{mode} num node pos'] = num_node_pos
            log_dict[f'{mode} num node neg'] = num_node_neg
        return loss, log_dict, hist_node, hist_edge
    
