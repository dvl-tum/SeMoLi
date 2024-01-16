from .DBSCAN import DBSCAN
from .DBSCAN_Intersection import DBSCAN_Intersection
from .detector import Detector3D
from .GNN import ClusterGNN, GNNLoss
from .SimpleTracker import SimpleTracker
from .icp_registration import ICPRegistration
from .constrained_icp import ConstrainedICPRegistration
from .flow_registration import FlowRegistration, FlowRegistration_From_Max

_model_factory = {
    'DBSCAN': DBSCAN,
    'DBSCAN_Intersection': DBSCAN_Intersection,
    'GNN': ClusterGNN}
_loss_factory = {
    'DBSCAN': None,
    'DBSCAN_Intersection': None,
    'GNN': GNNLoss}
_tracker_factory = {
    'SimpleTracker': SimpleTracker}
_registration_factory = {
    'ICP': ICPRegistration,
    'Flow': FlowRegistration,
    'ConstICP': ConstrainedICPRegistration,
    'FlowMax': FlowRegistration_From_Max
}
