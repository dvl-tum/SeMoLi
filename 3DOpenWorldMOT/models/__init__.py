from .DBSCAN import DBSCAN
from .SpectralClustering import SpectralClustering
from .tracker import Tracker3D
from .GNN import ClusterGNN, GNNLoss
from .SimpleGraph import SimpleGraph, SimpleGraphLoss

_model_factory = {
    'DBSCAN': DBSCAN,
    'GNN': ClusterGNN,
    'SpectralClustering': SpectralClustering,
    'SimpleGraph': SimpleGraph}
_loss_factory = {
    'DBSCAN': None,
    'GNN': GNNLoss,
    'SpectralClustering': None,
    'SimpleGraph': SimpleGraphLoss}