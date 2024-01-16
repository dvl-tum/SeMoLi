import os
import torch
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.nn import knn_graph, radius_graph
import torch.nn as nn
import numpy as np
from collections import defaultdict
import rama_py
import random
import matplotlib
import os
import logging
from PseudoDetection3D.models.losses import sigmoid_focal_loss
from torch import multiprocessing as mp
import pickle
import torch.utils.checkpoint as checkpoint
import torch_cluster
from scipy.sparse.csgraph import connected_components


rgb_colors_dict = {}
for name, hex in matplotlib.colors.cnames.items():
    rgb_colors_dict[name] = matplotlib.colors.to_rgb(hex)
rgb_colors = list(rgb_colors_dict.values())
rgb_colors = rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
rgb_colors = rgb_colors + rgb_colors
random.seed(10)
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
    a = a.view(-1, pos_dim)
    b = traj.view(traj.shape[0], -1, pos_dim)[edge_index[1]]
    b = b.view(-1, pos_dim)

    return torch.nn.PairwiseDistance(p=2)(a, b).view(edge_index.shape[1], -1)

def get_per_time_pos_traj_diff(traj, pos, edge_index, pos_dim):
    a = traj.view(traj.shape[0], -1, pos_dim)[edge_index[0]]+pos[edge_index[0]].unsqueeze(1)  
    a = a.view(-1, pos_dim)
    b = traj.view(traj.shape[0], -1, pos_dim)[edge_index[1]]+pos[edge_index[1]].unsqueeze(1)
    b = b.view(-1, pos_dim)
    
    return torch.nn.PairwiseDistance(p=2)(a, b).view(edge_index.shape[1], -1)

def get_per_time_vel_diff(traj, time, _batch, dataset, edge_index, pos_dim):
    a = traj.view(traj.shape[0], -1, pos_dim)
    diff_time = time[_batch, 1:] - time[_batch, :-1]
    if 'argo' in dataset:
        diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
    else:
        diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)

    # get dx/dt of all nodes
    a = (a[:, 1:, :] - a[:, :-1, :])/diff_time.unsqueeze(2)
    # get dx/dt for sending and receiving
    b = a[edge_index[1]]
    a = a[edge_index[0]]
    # get diff between and compute norm to get magnitude
    return torch.linalg.norm(simplediff(a, b), dim=-1)


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
            graph_eval='radius',
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
            oracle_cluster=False,
            dataset='waymo',
            layers_node=None,
            layers_edge=None,
            ignore_stat_edges=0,
            ignore_stat_nodes=0,
            filter_edges=-1,
            node_loss=True,
            layer_norm=False,
            batch_norm=False,
            drop_out=False,
            augment=False,
            rank=0,
            gradient_checkpointing=False, 
            remove_non_move_thresh=1.0,
            classification_is_moving_edge=False,
            classification_is_moving_node=False,
            set_all_pos=False,
            deep_supervision=False,
            initial_edge_as_input=False, 
            inflation_layer_edge=None,
            inflation_layer_node=None):
        super().__init__(aggr='mean')
        self.k = k
        self.k_eval = k_eval
        self.r = r
        self.graph = graph
        self.graph_eval = graph_eval
        self.edge_attr = edge_attr
        self.node_attr = node_attr
        self.graph_construction = graph_construction
        self.use_node_score = use_node_score * node_loss
        self.clustering = clustering
        self.deep_supervision = deep_supervision
        self.initial_edge_as_input = initial_edge_as_input
        traj_channels = traj_channels * pos_channels
        edge_dim = 0
        if '_DP_' in self.edge_attr:
            edge_dim += pos_channels
        if '_DT_' in self.edge_attr:
            edge_dim += traj_channels
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
        if '_DV_' in self.edge_attr:
            edge_dim += int(traj_channels/pos_channels) - 1
        
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
        if '_V0_' in self.node_attr:
            node_dim += 1
        if '_V_' in self.node_attr:
            node_dim += int(traj_channels/pos_channels) - 1
        
        layers = list()
        final = list()
        if self.use_node_score:
            final_node = list()
        self.reuse = layers_node.reuse
        self.num_layers = layers_node.num_layers
        layer_sizes_node = [layers_node.size for _ in range(0, layers_node.num_layers)]
        layer_sizes_edge = [layers_edge.size for _ in range(0, layers_edge.num_layers)]
        
        self.inflation_layer_edge = inflation_layer_edge
        self.inflation_layer_node = inflation_layer_node
        if self.inflation_layer_edge.use:
            self.encode_edge = torch.nn.Linear(edge_dim, self.inflation_layer_edge.dim)
            edge_dim = self.inflation_layer_edge.dim
        if self.inflation_layer_node.use:
            self.encode_node = torch.nn.Linear(node_dim, self.inflation_layer_node.dim)
            node_dim = self.inflation_layer_node.dim

        self.encode_layer = ClusterLayer(
                    in_channel_node=node_dim,
                    in_channel_edge=edge_dim,
                    out_channel_node=layer_sizes_node[0],
                    out_channel_edge=layer_sizes_edge[0],
                    use_batchnorm=batch_norm,
                    use_layernorm=layer_norm,
                    use_drop=drop_out,
                    skip_node_update=False)
        
        if self.deep_supervision:
            final.append(nn.Linear(layer_sizes_node[0], 1))
            if self.use_node_score:
                final_node.append(nn.Linear(layer_sizes_edge[0], 1))
        self.layers = None
        if self.reuse:
            node_dim = layers_node.size
            edge_dim = layers_edge.size
            self.layers = ClusterLayer(
                    in_channel_node=layers_node.size,
                    in_channel_edge=layers_edge.size,
                    out_channel_node=layers_node.size,
                    out_channel_edge=layers_edge.size,
                    use_batchnorm=batch_norm,
                    use_layernorm=layer_norm,
                    use_drop=drop_out,
                    skip_node_update=False)
            for j in range(layers_edge.num_layers):
                if self.deep_supervision or j == len(layer_sizes_node)-1 :
                    final.append(nn.Linear(layers_edge.size, 1))
                    if self.use_node_score:
                        final_node.append(nn.Linear(layers_node.size, 1))
        else:
            node_dim = layer_sizes_node[0]
            edge_dim = layer_sizes_edge[0]
            for j, (node_dim_hid, edge_dim_hid) in enumerate(zip(layer_sizes_node, layer_sizes_edge)):
                if j == len(layer_sizes_node) -1 and not self.use_node_score:
                    skip_node_update = True
                else:
                    skip_node_update = False

                layers.append(ClusterLayer(
                    in_channel_node=node_dim,
                    in_channel_edge=edge_dim,
                    out_channel_node=node_dim_hid,
                    out_channel_edge=edge_dim_hid,
                    use_batchnorm=batch_norm,
                    use_layernorm=layer_norm,
                    use_drop=drop_out,
                    skip_node_update=skip_node_update))
                
                node_dim = node_dim_hid
                edge_dim = edge_dim_hid
                if self.deep_supervision or j == len(layer_sizes_node) -1 :
                    final.append(nn.Linear(edge_dim, 1))
                    if self.use_node_score:
                        final_node.append(nn.Linear(node_dim, 1))

            self.layers = SevInpSequential(gradient_checkpointing, layers)

        self.final = SevInpSequential(gradient_checkpointing, final)
        if self.use_node_score:
            self.final_node = SevInpSequential(gradient_checkpointing, final_node)

        self.sigmoid = torch.nn.Sigmoid()
        # self.sigmoid = torch.nn.Tanh()
        self.cut_edges = cut_edges
        self.augment = augment
        self.min_samples = min_samples
        self.do_visualize = do_visualize
        self.my_graph = my_graph
        self.oracle_cluster = oracle_cluster
        self.oracle_node = oracle_node
        self.oracle_edge = oracle_edge
        self.set_all_pos = set_all_pos
        self.ignore_stat_edges = ignore_stat_edges
        self.ignore_stat_nodes = ignore_stat_nodes
        self.filter_edges = filter_edges
        self.dataset = dataset
        self.remove_non_move_thresh = remove_non_move_thresh
        self.gradient_checkpointing = gradient_checkpointing
        self.classification_is_moving_node = classification_is_moving_node
        self.classification_is_moving_edge = classification_is_moving_edge

        self.opts = rama_py.multicut_solver_options("PD")
        self.opts.sanitize_graph = True
        self.opts.verbose = False

        self.rank = rank
        self.pdist = torch.nn.PairwiseDistance()

    def initial_edge_attributes(self, x1, x2, edge_index, timestamps=None, batch=None):
        """
        x1 = N x 3
        x2 = N x 3*T

        ----

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
            a = torch.hstack([x2[edge_index[0]], x1[edge_index[0]]])
            b = torch.hstack([x2[edge_index[1]], x1[edge_index[1]]])
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
        
        if '_DV_' in self.edge_attr:
            edge_attr.append(get_per_time_vel_diff(x1, timestamps, batch, self.dataset, edge_index, pos_dim))
            edge_dim += int(traj_dim / 3) -1
        
        edge_attr = torch.hstack(edge_attr).squeeze().float()
        
        return edge_attr, edge_dim
    
    def initial_node_attributes(self, x1, x2, _type, timestamps=None, batch=None):
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
        V0 = initial_vel
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
            node_attr.append(x1.view(num_objects, -1, pos_dim))
        if  '_MDOT_' in _type:
            node_attr.append(x1.view(num_objects, -1, pos_dim)+x2.unsqueeze(1))

        # FOR GRAPH CONST AND NODES
        if '_P_' in _type:
            node_attr.append(x2)
            node_dim += pos_dim
        if '_T_' in _type:
            node_attr.append(x1)
            node_dim += traj_dim
        if '_PT_' in _type:
            node_attr.append((x1.view(num_objects, -1, pos_dim)+x2.unsqueeze(1)).view(num_objects, -1))
            node_dim += traj_dim
        if '_MMMV_' in _type or '_V_' in _type or '_V0_' in _type:
            _node_attr = x1.view(num_objects, -1, pos_dim)
            diff_time = timestamps[batch, 1:] - timestamps[batch, :-1]
            if 'argo' in self.dataset:
                diff_time = diff_time / torch.pow(torch.tensor(10), 9.0) 
            else:
                diff_time = diff_time / torch.pow(torch.tensor(10), 6.0)
            _node_attr = _node_attr[:, 1:, :] - _node_attr[:, :-1, :]
            _node_attr = torch.linalg.norm(_node_attr, dim=-1)
            _node_attr = _node_attr / diff_time
            if '_V_' in _type:
                node_attr.append(_node_attr)
                node_dim += int(traj_dim/pos_dim) - 1
            elif '_V0_' in _type:
                node_attr.append(_node_attr[:, 0].unsqueeze(1))
                node_dim += 1
            else:
                _node_attr = torch.vstack([
                    _node_attr.min(dim=-1).values,
                    _node_attr.max(dim=-1).values,
                    _node_attr.mean(dim=-1)]).T
                node_attr.append(_node_attr)
                node_dim += 3
        node_attr = torch.hstack(node_attr).squeeze().float()

        return node_attr, node_dim
    
    def get_graph(self, node_attr, r=5, max_num_neighbors=16, batch_idx=None, type='radius', metric='euclidean', fast=False):
        # my graph
        _idxs_0, _idxs_1 = list(), list()
        for ith, (start, end) in enumerate(zip(batch_idx[:-1], batch_idx[1:])):
            # iterate over frames in batch
            X = node_attr[start:end]

            # get distances between nodes
            if fast:
                x_shape = X.shape
                num_neighbors = min(65, x_shape[0])
                knn_0 = torch_cluster.knn(X[:, 0, :], X[:, 0, :], k=num_neighbors)
                knn_0 = knn_0.view(2, x_shape[0], -1)[:, :, 1:]
                idx = knn_0.reshape(2, -1)
                dist = self.pdist(
                    X[idx[0], :, :].view(-1, x_shape[2]), X[idx[1], :, :].view(-1, x_shape[2]))
                dist = dist.view(-1, x_shape[1])
                dist = dist.mean(dim=1).view(x_shape[0], -1)
            else:
                try:
                    dist = torch.cdist(X ,X[:, t, :]).mean(dim=1)
                except:
                    dist = torch.zeros(X.shape[0], X.shape[0]).to(self.rank)
                    for t in range(X.shape[1]):
                        dist += torch.cdist(X[:, t, :].unsqueeze(0),X[:, t, :].unsqueeze(0)).squeeze()
                    dist = dist / X.shape[1]

            # get indices up to max_num_neighbors per node --> knn neighbors
            num_neighbors = min(max_num_neighbors, dist.shape[0])
            if fast:
                idxs_0 = torch.tile(torch.arange(dist.shape[0]).unsqueeze(1).to(self.rank), (1, num_neighbors)).flatten()
                idxs_1_for_rad = dist.topk(k=num_neighbors, dim=1, largest=False).indices.flatten()
                idxs_1 = knn_0[1, idxs_0, idxs_1_for_rad]
            else:
                # set diagonal elements to 0to have no self-loops
                dist.fill_diagonal_(100)
                idxs_0 = torch.tile(torch.arange(dist.shape[0]).unsqueeze(1).to(self.rank), (1, num_neighbors)).flatten()
                idxs_1 = dist.topk(k=num_neighbors, dim=1, largest=False).indices.flatten()
                idxs_1_for_rad = idxs_1

            # if radius graph, filter nodes that are within radius 
            # but don't exceed max num neighbors
            if type == 'radius':
                dist = dist[idxs_0, idxs_1_for_rad]
                idx = torch.where(dist<r)[0]
                idxs_0, idxs_1 = idxs_0[idx], idxs_1[idx]
            
            # store precomputed edges
            # name = data['path'][ith].split('_')[-1]
            # p = f'/workspace/result/all_egocomp_margin0.6_width25_{self.graph_construction}/{name}' 
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

    def get_graph_VEL_POS(self, node_attr, pos, r=5, max_num_neighbors=16, batch_idx=None):
        # my graph
        _idxs_0, _idxs_1 = list(), list()
        for ith, (start, end) in enumerate(zip(batch_idx[:-1], batch_idx[1:])):
            # iterate over frames in batch
            X = pos[start:end]
            P = node_attr[start:end]
            # get distances between nodes
            x_shape = X.shape
            num_neighbors = min(max_num_neighbors, x_shape[0])
            knn_0 = torch_cluster.knn(X, X, k=num_neighbors)
            knn_0 = knn_0.view(2, x_shape[0], -1)[:, :, 1:]
            idx = knn_0.reshape(2, -1)
            idxs_0 = idx[0]
            idxs_1 = idx[1]

            # get indices up to max_num_neighbors per node --> knn neighbors
            dist = self.pdist(P[idxs_0, :], P[idxs_1, :])

            # if radius graph, filter nodes that are within radius 
            # but don't exceed max num neighbors
            idx = torch.where(dist<r)[0]
            idxs_0, idxs_1 = idxs_0[idx], idxs_1[idx]
            
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
        # if no points return
        if data['traj'].shape[0] == 0:
            return [None, None], list(), None, None
        
        # extract data
        data = data.to(self.rank)
        batch_idx = data._slice_dict['pc_list']
        traj = data['traj'].view(data['traj'].shape[0], -1)
        pc = data['pc_list']
        
        # get initial node attributed ans attirbutes to construct the grap
        node_attr, _ = self.initial_node_attributes(
            traj, pc, self.node_attr, data['timestamps'], data['batch'])
        if self.node_attr == self.graph_construction:
            graph_attr = node_attr
        else:
            graph_attr, _ = self.initial_node_attributes(
                traj, pc, self.graph_construction, data['timestamps'], data['batch'])
        
        # get edges using knn graph (for computational feasibility)
        k = self.k if not eval else self.k_eval
        graph = self.graph if not eval else self.graph_eval
        if 'edge_index' not in data.keys:
            if graph == 'knn':
                if self.my_graph and len(graph_attr.shape) != 2:
                    edge_index = self.get_graph(
                        graph_attr, self.r, max_num_neighbors=k, batch_idx=batch_idx, type='knn')
                else:
                    edge_index = knn_graph(x=graph_attr, k=k, batch=data['batch'])
            elif graph == 'radius':
                if self.my_graph and len(graph_attr.shape) != 2:
                    edge_index = self.get_graph(
                        graph_attr, self.r, max_num_neighbors=k, batch_idx=batch_idx, type='radius')
                else:
                    edge_index = radius_graph(graph_attr, self.r, data['batch'], max_num_neighbors=k)
            elif graph == 'VELPOS':
                edge_index = self.get_graph_VEL_POS(graph_attr, pc, r=self.r, max_num_neighbors=k, batch_idx=batch_idx)
        else:
            edge_index = data['edge_index']

        # if there are no edges in pc --> very sparse?!
        if edge_index.shape[1] == 0:
            return [None, None], torch.tensor(list(range(pc.shape[0]))), None
        
        # get initial edge attributes
        edge_attr, _ = self.initial_edge_attributes(traj, pc, edge_index, timestamps=data['timestamps'], batch=data['batch'])
        
        if self.inflation_layer_edge.use:
            edge_attr = self.encode_edge(edge_attr)
        if self.inflation_layer_node.use:
            node_attr = self.encode_node(node_attr)
        
        if not self.initial_edge_as_input:
            # forward pass thourgh layers
            final, score = list(), None
            final_node, node_score = list(), None
            node_attr, edge_index, edge_attr = self.encode_layer(node_attr, edge_index, edge_attr)
            if self.deep_supervision or self.layers is None:
                score = self.final[0](edge_attr)
                final.append(score)
                if self.use_node_score:
                    node_score = self.final_node[0](node_attr)
                    final_node.append(node_score)
            for i in range(self.num_layers):
                if self.reuse:
                    if self.gradient_checkpointing:
                        inputs = (node_attr, edge_index, edge_attr)
                        inputs = checkpoint.checkpoint(self.layers, *inputs, use_reentrant=False)
                        node_attr, edge_index, edge_attr = inputs
                    else:
                        node_attr, edge_index, edge_attr = self.layers(node_attr, edge_index, edge_attr)
                else:
                    if self.gradient_checkpointing:
                        inputs = (node_attr, edge_index, edge_attr)
                        inputs = checkpoint.checkpoint(self.layers[i], *inputs, use_reentrant=False)
                        node_attr, edge_index, edge_attr = inputs
                    else:
                        node_attr, edge_index, edge_attr = self.layers[i](node_attr, edge_index, edge_attr)
                if self.deep_supervision:
                    score = self.final[i+1](edge_attr)
                    final.append(score)
                    if self.use_node_score:
                        node_score = self.final_node[i+1](node_attr)
                        final_node.append(node_score)
                elif i == self.num_layers-1:
                    score = self.final[-1](edge_attr)
                    final.append(score)
                    if self.use_node_score:
                        node_score = self.final_node[-1](node_attr)
                        final_node.append(node_score)
        else:
            score = 1 - torch.linalg.norm(edge_attr, dim=1)
            node_score = None
            final = [score]
            final_node = list()
        # evaluate correlation clustering
        if eval and corr_clustering:
            _score = self.sigmoid(score)
            if self.use_node_score:
                _node_score = self.sigmoid(node_score)
            else:
                _node_score = None
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
            return [final, final_node], all_clusters, edge_index, None
        elif eval:
            return [final, final_node], [[]*len(batch_idx[:-1])], edge_index, None
        
        return [final, final_node], edge_index, None
    
    def corr_clustering(self, iter_data, method='connected_components'):
        i, (start, end) = iter_data
        edge_index, _node_score, _score, data, score, node_score, pc, rama_cuda, name = self.args
        
        if self.oracle_cluster:
            clusters = data['point_instances'][start:end]
            clusters[clusters == 0] = -1
            return clusters.cpu().numpy()

        # get edge index and scores for current graph
        edge_mask = torch.logical_or(
            torch.logical_and(edge_index[0] >= start, edge_index[1] < end),
            torch.logical_and(edge_index[1] >= start, edge_index[0] < end))
        graph_edge_index = edge_index[:, edge_mask]
        src, dst = graph_edge_index
        if self.use_node_score:
            graph_node_score = _node_score[start:end]
        graph_edge_score = _score[edge_mask]
        
        # set oracle scores edge
        if self.oracle_edge:
            if self.set_all_pos: 
                graph_edge_score = torch.ones(graph_edge_score.shape[0]).unsqueeze(1).to(graph_edge_score.device)
            else:
                graph_edge_score[data['point_instances'][src] == data['point_instances'][dst]] = 1
                graph_edge_score[data['point_instances'][src] != data['point_instances'][dst]] = 0
                graph_edge_score[data['point_instances'][src] <= 0] = 0
                if self.classification_is_moving_edge:
                    graph_edge_score[~data['point_instances_mov'][src]] = 0
            
            score[edge_mask] = graph_edge_score
            score[score == 0] = -10
            score[score == 1] = 10
        
        # set oracle scores node
        if self.oracle_node and self.use_node_score:
            graph_node_score[data['point_categories'][start:end]>0] = 1
            graph_node_score[data['point_categories'][start:end]<=0] = 0
            if self.classification_is_moving_node:
                graph_node_score[~data['point_instances_mov'][start:end]] = 0

            node_score[start:end] = graph_node_score
            node_score[node_score == 0] = -10
            node_score[node_score == 1] = 10

        # filter out edges with very low score already
        if self.filter_edges > 0:
            graph_edge_index = graph_edge_index[:, (graph_edge_score > self.filter_edges).squeeze()]
            graph_edge_score = graph_edge_score[(graph_edge_score > self.filter_edges).squeeze()]
        
        # if not os.path.isfile(os.path.join('/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/vis_graph', 'filtered', data['log_id'][0] + '.png')):
        #     self.visualize(torch.arange(end-start), graph_edge_index-start.item(), pc[start:end], torch.ones(end-start), data.timestamps[i, 0], name='filtered',data=data)
        
        # filter egdes using node score to make problem smaller
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
        
        # if not os.path.isfile(os.path.join('/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/vis_graph', 'mapped', data['log_id'][0] + '.png')):
        #     self.visualize(torch.arange(edges.shape[0]), _edge_index, pc[edges], torch.ones(edges.shape[0]), data.timestamps[i, 0], name='mapped', data=data)
        clusters = torch.ones(end.item()-start.item()) * - 1
        clusters = clusters.int().to(self.rank)
        if edges.shape[0]:

            if self.clustering == 'correlation':
                # solve correlation clustering
                '''
                i = _edge_index[0, :].contiguous().to(torch.int32) # Can only be of dtype int32!
                j = _edge_index[1, :].contiguous().to(torch.int32) # Can only be of dtype int32!
                costs = ((graph_edge_score*2)-1).contiguous().to(torch.float32) # Can only be of dtype float32!
                num_nodes = edges.shape[0]
                node_labels = torch.ones(num_nodes, device = i.device).to(torch.int32)
                num_edges = i.numel()
                self.opts.dump_timeline = True # Set to true to get intermediate results.
                timeline = rama_py.rama_cuda_gpu_pointers(i.data_ptr(), j.data_ptr(), costs.data_ptr(), node_labels.data_ptr(), num_nodes, num_edges, i.device.index, self.opts)
                mapped_clusters = node_labels
                '''
                rama_out = rama_cuda(
                    [e[0] for e in _edge_index.T.cpu().numpy()],
                    [e[1] for e in _edge_index.T.cpu().numpy()],
                    (graph_edge_score.cpu().numpy()*2)-1,
                    self.opts)
                mapped_clusters = torch.tensor(rama_out[0]).to(self.rank).int()
            else:
                from scipy.sparse import csr_matrix
                input_graph = csr_matrix((np.ones(_edge_index.shape[1]), (_edge_index[0, :].cpu().numpy(), _edge_index[1, :].cpu().numpy())), shape=(edges.shape[0], edges.shape[0]))
                _, mapped_clusters = connected_components(csgraph=input_graph, directed=False, return_labels=True)
                mapped_clusters = torch.from_numpy(mapped_clusters).to(_edge_index.device)

            # map back 
            _edge_index[0, :] = edges[_edge_index[0, :]]
            _edge_index[1, :] = edges[_edge_index[1, :]]
            clusters[edges] = mapped_clusters
        
        clusters = clusters.cpu().numpy()

        # if clusters are < min_samples set to rubbish
        _clusters = defaultdict(list)
        for iter, c in enumerate(clusters):
            _clusters[c].append(iter)
        for c, node_list in _clusters.items():
            if len(node_list) < self.min_samples:
                for n in node_list:
                    clusters[n] = -1
        
        # if not os.path.isfile(os.path.join('/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/vis_graph', 'after', data['log_id'][0] + '.png')):
        #     self.visualize(torch.arange(end-start), graph_edge_index, pc[start:end], clusters, data.timestamps[i, 0], name='after', data=data)
        
        return clusters

    def visualize(self, nodes, edge_indices, pos, clusters, timestamp, mode='before', name='General', data=None, colors=None):
        os.makedirs(f'../../../vis_graph/{name}', exist_ok=True)
        print('made dir ', f'../../../vis_graph/{name}')
        import networkx as nx
        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        clusters_dict = defaultdict(list)
        for i, c in enumerate(clusters):
            clusters_dict[c].append(i)
        clusters = clusters_dict

        # adapt edges to predicted clusters
        if colors is None:
            colors = [rgb_colors[0] for _ in range(nodes.shape[0])]
            if name == 'after':
                edge_indices = list()
                for c, nodelist in clusters.items():
                    if c == -1:
                        continue
                    for i, node1 in enumerate(nodelist):
                        colors[node1] = rgb_colors[c]
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

        # save graph
        labels = {n.item(): str(n.item()) for n in nodes}
        plt.figure(figsize=(50, 50))
        print(type(pos), pos[0])
        nx.draw_networkx_edges(G, pos, width=3)
        print(type(pos), pos[0])
        nx.draw_networkx_nodes(G, pos, node_color=colors)
        # nx.draw_networkx_labels(G, pos, labels=labels, font_size=6, font_color='red')
        plt.axis("off")
        p = os.path.join('/workspace/3DOpenWorldMOT_motion_patterns/3DOpenWorldMOT/3DOpenWorldMOT/vis_graph', name, data['log_id'][0] + '_{timestamp}' + '.png')
        plt.savefig(p, bbox_inches='tight', dpi=300)
        # plt.savefig(f'../../../vis_graph/{name}/{timestamp}_{mode}.png', bbox_inches='tight', dpi=300)
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
            classification_is_moving_edge=0,
            use_node_score=False,
            set_3_to_false=True) -> None:
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
        self.set_3_to_false = set_3_to_false
        self.max_iter = 2000
        self.rank = rank
        self.ignore_stat_edges = ignore_stat_edges
        self.ignore_stat_nodes = ignore_stat_nodes
        self.ignore_edges_between_background = ignore_edges_between_background
        self.classification_is_moving_node = classification_is_moving_node
        self.classification_is_moving_edge = classification_is_moving_edge
        assert ~self.classification_is_moving_node or (self.classification_is_moving_node != self.ignore_stat_nodes), "Can only either ignore static objects or classify as moving object or not a moving object"
        assert ~self.classification_is_moving_edge or (self.classification_is_moving_edge != self.ignore_edges_between_background), "Can not ignore background when classifying moving vs static edges"
        assert ~self.classification_is_moving_edge or (self.classification_is_moving_edge != self.ignore_stat_edges), "Can not ignore static edges when classifying moving vs static edges"
        
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
        edge_logits, node_logits = logits
        loss = 0
        log_dict = dict()
        same_graph = data['batch'][edge_index[0, :]] == data['batch'][edge_index[1, :]]
                
        if self.bce_loss:
            point_instances = data.point_instances
            point_categories = data.point_categories[edge_index[0, :]]
            point_categories1 = data.point_categories[edge_index[1, :]]

            # get bool edge mask
            point_instances = point_instances[edge_index[0, :]] == point_instances[edge_index[1, :]]
            
            # keep only edges that belong to same graph (for batching opteration)
            point_instances = torch.logical_and(point_instances, same_graph).bool()
            
            # setting edges that do not belong to object to zero as well as sign edges
            # --> instance 0 is no object
            point_instances[data.point_instances[edge_index[0, :]] == 0] = False
            if self.set_3_to_false:
                point_instances[data.point_categories[edge_index[0, :]] == 3] = False
            point_instances = point_instances.to(self.rank)

            # if using moving vs non-moving / background as training objective
            if self.classification_is_moving_edge:
                point_instances[torch.logical_and(
                    ~(data.point_instances_mov[edge_index[0, :]] != 0), 
                    data.point_instances[edge_index[0, :]] != 0)] = False
                
            # if ignoring edges between static object or edges between background during training 
            # --> requires any kind of node classification!!
            point_mask = torch.zeros(point_instances.shape[0], dtype=bool).to(self.rank)
            if self.ignore_stat_edges:
                point_mask = torch.logical_or(
                    torch.logical_and(
                        torch.logical_and(
                            ~(data.point_instances_mov[edge_index[0, :]] != 0), 
                            data.point_instances[edge_index[0, :]] != 0),
                        torch.logical_and(
                            ~(data.point_instances_mov[edge_index[1, :]] != 0), 
                            data.point_instances[edge_index[1, :]] != 0)),
                    point_mask)

            if self.ignore_edges_between_background:
                point_mask = torch.logical_or(
                    torch.logical_and(
                        data.point_instances[edge_index[0, :]] == 0,
                        data.point_instances[edge_index[1, :]] == 0),
                    point_mask)
            
            # filter edge logits, point instances and point categories
            edge_logits = [e[~point_mask] for e in edge_logits]
            point_instances = point_instances[~point_mask].float()
            point_categories = point_categories[~point_mask]
            point_categories1 = point_categories1[~point_mask]
            edge_index = edge_index[:, ~point_mask]
            
            if edge_logits[0].shape[0] == 0:
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
            
            bce_loss_edge = 0
            for _edge_logits in edge_logits:
                if not self.focal_loss_edge:
                    bce_loss_edge += self._edge_loss(
                        _edge_logits.squeeze(), point_instances.squeeze())
                else:
                    bce_loss_edge += self._edge_loss(
                        _edge_logits.squeeze(),
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
            logits_rounded = self.sigmoid(edge_logits[-1].clone().detach()).squeeze()
            hist_edge = np.histogram(logits_rounded.cpu().numpy(), bins=10, range=(0., 1.))
            logits_rounded[logits_rounded>0.5] = 1
            logits_rounded[logits_rounded<=0.5] = 0
            correct = logits_rounded == point_instances.squeeze()

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

        if self.node_loss:
            # get if point is object (sign is considered as no object)
            is_object = data.point_instances != 0
            if self.set_3_to_false:
                is_object[data.point_categories == 3] = False
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
                    node_logits = [n[~is_object_stat] for n in node_logits]
                    object_class = object_class[~is_object_stat]
                if self.classification_is_moving_node:
                    is_object[is_object_stat] = False

            # weight pos and neg samples
            if weight_node and not self.focal_loss_node:
                num_pos = torch.sum((is_object==1).float())
                num_neg = torch.sum((is_object==0).float())
                pos_weight = num_neg/num_pos
                self._node_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(self.rank)
            
            bce_loss_node = 0
            for _node_logits in node_logits:
                # compute loss            
                if not self.focal_loss_node:
                    bce_loss_node += self._node_loss(
                        _node_logits.squeeze(), is_object.squeeze())
                else:
                    bce_loss_node += self._node_loss(
                        _node_logits.squeeze(),
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
            logits_rounded_node = self.sigmoid(node_logits[-1].clone().detach()).squeeze()
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

        return loss, log_dict
    
