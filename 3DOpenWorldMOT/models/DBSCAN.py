import sklearn.cluster
import torch.nn as nn
import numpy as np
from sklearn.metrics.cluster import contingency_matrix
from scipy.optimize import linear_sum_assignment
from scipy.stats import mode


class DBSCAN():
    def __init__(
            self,
            rank=0, 
            min_samples=2, 
            thresh=6,
            input='traj_pos',
            dataset='waymo',
            flow_thresh=0.2) -> None:

        self.model = sklearn.cluster.DBSCAN(min_samples=min_samples, eps=thresh)
        self.input = input
        self.dataset = dataset
        self.flow_thresh = flow_thresh
    
    def forward(self, clustering):
        traj = clustering.traj.numpy()
        pc = clustering['pc_list'].numpy()
        if len(pc.shape) != 2:
            pc = pc[0]
        timestamps = clustering['timestamps'].squeeze().numpy()
        diff_traj = traj[:, :-1] - traj[:, 1:]
        
        # if no moving point
        if traj.shape[0] == 0:
            return None, [], None, None
        
        # get mask to remove static points
        mask = np.linalg.norm(diff_traj[:, :, :-1], ord=2, axis=2)
        time = timestamps[1:traj.shape[1]]-timestamps[:traj.shape[1]-1]
        if 'waymo' in self.dataset:
            time = time / np.power(10, 6.0) 
        else:
            time = time / np.power(10, 9.0)
        
        mask = mask / time
        mask = mask.mean(axis=1)
        
        time = np.expand_dims(time, axis=1)
        time = np.tile(time, (1, 3))
        
        # get input
        if self.input == 'traj':
            traj = diff_traj # / time
            inp = traj.reshape(traj.shape[0], -1)
        elif self.input == 'flow':
            inp = diff_traj[:, 0, :].reshape(pc.shape[0], -1) # / time
        elif self.input == 'traj_pos':
            pc = np.expand_dims(pc, axis=1)
            pc = np.repeat(pc, traj.shape[1], axis=1)
            traj = traj + pc
            inp = traj.reshape(traj.shape[0], -1)
        elif self.input == 'pos':
            inp = pc.reshape(pc.shape[0], -1)
        labels = np.ones(mask.shape[0]) * -1
        
        # filter pos und traj
        idxs = np.where(mask > self.flow_thresh)[0]
        mask = mask > self.flow_thresh
        inp = inp[mask]
        
        # if no moving point
        if inp.shape[0] == 0:
            return None, [labels], None, None

        # get clustering
        clustering = self.model.fit(inp) # only flow 0.0015
        _labels = clustering.labels_
        labels[mask] = _labels
        labels = labels.astype(int)
        return None, [labels], None, None
    
    def __call__(self, clustering, eval=False, name='General', corr_clustering=False):
        return self.forward(clustering)
