

import sklearn.cluster
import sklearn.metrics.cluster
import numpy as np


def calc_normalized_mutual_information(ys, x):
    return sklearn.metrics.cluster.normalized_mutual_info_score(x, ys)
