import sys
path = 'J://utils'
sys.path.append(path)

import common_utils as utils
import clustering_utils as cl_utils
import classification_utils as cutils

X, y = cl_utils.generate_synthetic_data_2d_clusters(n_samples=300, n_centers=4, cluster_std=0.60)
utils.plot_data_2d(X)

X, _ = cl_utils.generate_synthetic_data_3d_clusters(n_samples=500, n_centers=3, cluster_std=1.4)
utils.plot_data_3d(X)

X, _ = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=300)
utils.plot_data_2d(X)

X, _ = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=300)
utils.plot_data_2d(X)

