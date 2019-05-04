import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

import numpy as np
from sklearn import cluster,metrics
from common_utils import *
from clustering_utils import *
from classification_utils import *

scoring = 's_score'

X, y = generate_synthetic_data_2d_clusters(n_samples=300, n_centers=4, cluster_std=0.60)
plot_data_2d(X)

agg_estimator = cluster.AgglomerativeClustering()
agg_grid = {'n_clusters':list(range(2,5)), 'linkage':['ward', 'complete', 'average']}
grid_search_plot_models_2d_clustering(agg_estimator, agg_grid, X)
grid_search_plot_one_parameter_curves_clustering(agg_estimator, agg_grid, X, scoring = scoring)
agg_final_model = grid_search_best_model_clustering(agg_estimator, agg_grid, X, scoring=scoring)
plot_model_2d_clustering(agg_final_model, X)

X, y = generate_synthetic_data_3d_clusters(n_samples=300, n_centers=5, cluster_std=0.60)
plot_data_3d(X)

agg_estimator = cluster.AgglomerativeClustering()
agg_grid = {'n_clusters':list(range(2,5)), 'linkage':['ward', 'complete', 'average']}
grid_search_plot_models_3d_clustering(agg_estimator, agg_grid, X)
grid_search_plot_one_parameter_curves_clustering(agg_estimator, agg_grid, X, scoring = scoring)
agg_final_model = grid_search_best_model_clustering(agg_estimator, agg_grid, X, scoring=scoring)
plot_model_3d_clustering(agg_final_model, X)

# data sets where k-means fails to cluster
X, y = generate_nonlinear_synthetic_data_classification3(n_samples=300)
plot_data_2d(X)

agg_estimator = cluster.AgglomerativeClustering()
agg_grid = {'n_clusters':list(range(2,5)), 'linkage':['ward', 'complete', 'average']}
grid_search_plot_models_2d_clustering(agg_estimator, agg_grid, X)
grid_search_plot_one_parameter_curves_clustering(agg_estimator, agg_grid, X, scoring = scoring)
agg_final_model = grid_search_best_model_clustering(agg_estimator, agg_grid, X, scoring=scoring)
plot_model_2d_clustering(agg_final_model, X)

X, y = generate_nonlinear_synthetic_data_classification2(n_samples=300)
plot_data_2d(X)

agg_estimator = cluster.AgglomerativeClustering()
agg_grid = {'n_clusters':list(range(2,5)), 'linkage':['ward', 'complete', 'average']}
grid_search_plot_models_2d_clustering(agg_estimator, agg_grid, X)
grid_search_plot_one_parameter_curves_clustering(agg_estimator, agg_grid, X, scoring = scoring)
agg_final_model = grid_search_best_model_clustering(agg_estimator, agg_grid, X, scoring=scoring)
plot_model_2d_clustering(agg_final_model, X)