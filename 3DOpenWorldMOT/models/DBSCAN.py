from turtle import forward
import sklearn.cluster
import torch.nn as nn
import numpy as np


class DBSCAN():
    def __init__(self, min_samples=2, thresh=6, input='traj_pos') -> None:
        self.model = sklearn.cluster.DBSCAN(min_samples=min_samples, eps=thresh)
        self.input = input
    
    def forward(self, clustering):
        traj = clustering.traj.numpy()
        pc = clustering['pc_list'].numpy()
        if self.input == 'traj':
            inp = traj.reshape(traj.shape[0], -1)
        elif self.input == 'traj_pos':
            pc = np.expand_dims(pc, axis=1)
            pc = np.repeat(pc, traj.shape[1], axis=1)
            traj = traj + pc
            inp = traj.reshape(traj.shape[0], -1)
        elif self.input == 'pos':
            inp = pc.reshape(pc.shape[0], -1)
        clustering = self.model.fit(inp) # only flow 0.0015
        labels = clustering.labels_
        return None, labels, None, None
    
    def __call__(self, clustering, eval=False):
        return self.forward(clustering)
