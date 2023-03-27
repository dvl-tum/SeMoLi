from collections import defaultdict
from sklearn import cluster
import torch
from torch_geometric.nn.conv.message_passing import MessagePassing
from torch_geometric.nn import knn_graph, radius_graph
import torch.nn.functional as F
from torch_geometric.utils import softmax
import torch.nn as nn
import numpy as np
import matplotlib
import random

rgb_colors = {}
for name, hex in matplotlib.colors.cnames.items():
    rgb_colors[name] = matplotlib.colors.to_rgb(hex)
rgb_colors = list(rgb_colors.values())
random.shuffle(rgb_colors)


class SimpleGraph(MessagePassing):
    def __init__(self, k=32, r=0.5, graph='radius', edge_attr='diffpos', node_attr='diffpos', cut_edges=0.2, min_samples=5):
        super().__init__(aggr='mean')
        self.k = k
        self.r = r
        self.graph = graph
        self.edge_attr = edge_attr
        self.node_attr = node_attr
        self.cut_edges = cut_edges
        self.min_samples = min_samples

    def initial_edge_attributes(self, x1, x2, edge_index, distance='euclidean'):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if self.edge_attr == 'diffpos':
            a = [x2[edge_index[0]]]
            b = [x2[edge_index[1]]]
        elif self.edge_attr == 'difftraj':
            a = x1[edge_index[0]]
            b = x1[edge_index[1]]
            a_shape = a.shape
            a = [a.view(a_shape[0], -1, 3)]
            b = [b.view(a_shape[0], -1, 3)]
        elif self.edge_attr == 'difftraj_diffpos':
            a = torch.stack([x2[edge_index[0]], x1[edge_index[0]]])
            b = torch.stack([x2[edge_index[1]], x1[edge_index[1]]])
            a_shape = a.shape
            a = [a.view(a_shape[0], -1, 3)]
        elif self.edge_attr == 'diffpostraj':
            a = x2[edge_index[0]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[0]]
            b = x2[edge_index[1]].repeat((1, int(x1.shape[1]/x2.shape[1]))) + x1[edge_index[1]]
            a_shape = a.shape
            a = [a.view(a_shape[0], -1, 3)]
            b = [b.view(a_shape[0], -1, 3)]

        # get distance
        edge_attr = None
        if distance == 'cosine':
            d = torch.nn.functional.cosine_similarity
        elif distance == 'euclidean':
            d = torch.nn.PairwiseDistance(p=2)
        elif distance == 'l1':
            d = torch.nn.PairwiseDistance(p=1)
        for aa, bb in zip(a, b):
            if edge_attr is None:
                dist = d(aa, bb)
                if self.edge_attr == 'diffpostraj' or self.edge_attr == 'difftraj':
                    dist = dist.view(a_shape[0], int(a_shape[1]/3))
                    dist = dist.mean(dim=1)
                edge_attr = dist
            else:
                edge_attr += d(aa, bb)
        edge_attr /= len(a)

        # computation is cosine similarity --> 1-sim=dist
        if distance == 'cosine':
            edge_attr = 1 - edge_attr
        
        return edge_attr
    
    def initial_node_attributes(self, x1, x2):
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        if self.node_attr == 'pos':
            node_attr = x2
        elif self.node_attr == 'traj':
            node_attr = x1
        elif self.node_attr == 'traj_pos':
            node_attr = torch.hstack([x1, x2])

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
            edge_index = radius_graph(traj, self.r, batch)

        edge_attr = self.initial_edge_attributes(traj, pc, edge_index)

        # get which edges belong to which batch
        batch_edge = torch.zeros(edge_index.shape[1]).cuda()
        for i in range(1, batch_idx.shape[0]-1):
            mask = (edge_index.T < batch_idx[i+1]) & (edge_index.T >= batch_idx[i])
            mask = mask[:, 0] * i
            batch_edge += mask      

        src, dst = edge_index
        # computes per edge index by computing dot product between node features
        if not use_edge_att:
            score = (node_attr[src] * node_attr[dst]).sum(dim=-1)
        # directly uses edge attirbutes
        else:
            score = edge_attr
        
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
    
    def visualize(self, nodes, edge_indices, pos, clusters, timestamp):
        import networkx as nx
        import matplotlib 
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

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

        print(col_dict)
        # save graph
        plt.figure(figsize=(50, 50))
        nx.draw_networkx_edges(G, pos, width=3)
        nx.draw_networkx_nodes(G, pos, node_size=2, node_color=colors)
        plt.axis("off")
        plt.savefig(f'../../../vis_graph_{timestamp}.png', bbox_inches='tight')

class SimpleGraphLoss(nn.Module):
    def __init__(self, bce_loss=True) -> None:
        super().__init__()
        
        self.bce_loss = nn.BCELoss().cuda() if bce_loss else bce_loss
        self.max_iter = 2000

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
