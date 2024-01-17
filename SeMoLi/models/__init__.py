from .DBSCAN import DBSCAN
from .DBSCAN_Intersection import DBSCAN_Intersection
from .detector import Detector3D
from .GNN import ClusterGNN, GNNLoss

_model_factory = {
    'DBSCAN': DBSCAN,
    'DBSCAN_Intersection': DBSCAN_Intersection,
    'GNN': ClusterGNN}
_loss_factory = {
    'DBSCAN': None,
    'DBSCAN_Intersection': None,
    'GNN': GNNLoss}


