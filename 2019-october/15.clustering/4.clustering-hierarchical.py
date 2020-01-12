import sys
path = 'J://utils'
sys.path.append(path)

from sklearn import cluster
import common_utils as utils
import clustering_utils as cl_utils
import classification_utils as cutils

X, _= cl_utils.generate_synthetic_data_2d_clusters(n_samples=300, n_centers=4, cluster_std=0.60)
utils.plot_data_2d(X)

X, _ = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=300)
utils.plot_data_2d(X)

X, _ = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=300)
utils.plot_data_2d(X)

scoring = 's_score'
agg_estimator = cluster.AgglomerativeClustering()
agg_grid = {'linkage':['ward', 'complete', 'average'], 'n_clusters':list(range(2,7))}
agg_final_model = cl_utils.grid_search_best_model_clustering(agg_estimator, agg_grid, X, scoring=scoring)
cl_utils.plot_model_2d_clustering(agg_final_model, X)
