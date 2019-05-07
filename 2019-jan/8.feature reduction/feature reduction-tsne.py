import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

import numpy as np
from common_utils import *
from clustering_utils import *
from classification_utils import *
from tsne_utils import *

X, y = generate_synthetic_data_2d_clusters(n_samples=100, n_centers=4, cluster_std=0.10)
plot_data_2d_classification(X, y)
plot_tsne_result(X, y, 2)

X, y = generate_linear_synthetic_data_classification(100, 3, 2, [.5,.5])
plot_data_3d_classification(X, y)
plot_tsne_result(X, y, 3)
plot_tsne_result(X, y, 2)
