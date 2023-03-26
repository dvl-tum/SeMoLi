import sklearn.cluster
import torch.nn as nn
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy import sparse
from scipy import linalg
import pandas as pd


class SpectralClustering():
    def __init__(self, min_samples=2, thresh=6, input='traj_pos', graph='knn', num_ev=50, k=32) -> None:
        self.model = sklearn.cluster.DBSCAN(min_samples=min_samples, eps=thresh)
        self.input = input
        self.graph = graph
        self.num_ev = num_ev
        self.k = k

    def generate_graph_laplacian(self, feats):
        """Generate graph Laplacian from data."""
        if self.graph == 'knn':
            # Adjacency Matrix.
            connectivity = kneighbors_graph(X=feats, n_neighbors=self.k, mode='connectivity')
            adjacency_matrix_s = (1/2)*(connectivity + connectivity.T)

        # Graph Laplacian.
        graph_laplacian_s = sparse.csgraph.laplacian(csgraph=adjacency_matrix_s, normed=False)
        graph_laplacian = graph_laplacian_s.toarray()
        return graph_laplacian 
    
    def project_and_transpose(self, eigenvals, eigenvcts):
        """Select the eigenvectors corresponding to the first 
        (sorted) num_ev eigenvalues as columns in a data frame.
        """
        eigenvals_sorted_indices = np.argsort(eigenvals)
        indices = eigenvals_sorted_indices[: self.num_ev]

        proj_df = pd.DataFrame(eigenvcts[:, indices.squeeze()])
        proj_df.columns = ['v_' + str(c) for c in proj_df.columns]
        return proj_df
    
    def compute_spectrum_graph_laplacian(self, graph_laplacian):
        """Compute eigenvalues and eigenvectors and project 
        them onto the real numbers.
        """
        eigenvals, eigenvcts = linalg.eig(graph_laplacian)
        eigenvals = np.real(eigenvals)
        eigenvcts = np.real(eigenvcts)
        return eigenvals, eigenvcts
    
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

        graph_laplacian = self.generate_graph_laplacian(inp)
        eigenvals, eigenvcts = self.compute_spectrum_graph_laplacian(graph_laplacian)

        proj_vcts = self.project_and_transpose(eigenvals, eigenvcts)

        clustering = self.model.fit(proj_vcts) # only flow 0.0015
        labels = clustering.labels_
        #print(np.unique(labels))
        #quit()
        return None, labels, None, None
    
    def __call__(self, clustering, eval=False):
        return self.forward(clustering)
