from .DBSCAN import DBSCAN
from .DBSCAN_Intersection import DBSCAN_Intersection
from .SpectralClustering import SpectralClustering
from .tracker import Tracker3D
from .GNN import ClusterGNN, GNNLoss
from .SimpleGraph import SimpleGraph, SimpleGraphLoss

_model_factory = {
    'DBSCAN': DBSCAN,
    'DBSCAN_Intersection': DBSCAN_Intersection,
    'GNN': ClusterGNN,
    'SpectralClustering': SpectralClustering,
    'SimpleGraph': SimpleGraph}
_loss_factory = {
    'DBSCAN': None,
    'DBSCAN_Intersection': None,
    'GNN': GNNLoss,
    'SpectralClustering': None,
    'SimpleGraph': SimpleGraphLoss}