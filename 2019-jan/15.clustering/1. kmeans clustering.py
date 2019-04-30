import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'E://'
sys.path.append(path)

import numpy as np
from sklearn import cluster,metrics
from common_utils import *
from clustering_utils import *
from classification_utils import *

scoring = 's_score'

X, y = generate_synthetic_data_2d_clusters(n_samples=300, n_centers=4, cluster_std=0.60)
plot_data_2d(X)

kmeans_estimator = cluster.KMeans()
kmeans_grid = {'n_clusters':list(range(3,7))}
grid_search_plot_models_kmeans(kmeans_estimator, kmeans_grid, X)
kmeans_final_model = grid_search_best_model_clustering(kmeans_estimator, kmeans_grid, X, scoring=scoring)
plot_model_2d_kmeans(kmeans_final_model, X)

X, y = generate_synthetic_data_3d_clusters(n_samples=300, n_centers=5, cluster_std=0.60)
plot_data_3d(X)

kmeans_estimator = cluster.KMeans()
kmeans_grid = {'n_clusters':list(range(3,7))}
kmeans_final_model = grid_search_best_model_clustering(kmeans_estimator, kmeans_grid, X, scoring=scoring)
plot_model_3d_kmeans(kmeans_final_model, X)

# data sets where k-means fails to cluster
X, y = generate_nonlinear_synthetic_data_classification3(n_samples=300)
plot_data_2d(X)

kmeans_estimator = cluster.KMeans()
kmeans_grid = {'n_clusters':list(range(2,6))}
grid_search_plot_models_kmeans(kmeans_estimator, kmeans_grid, X)
kmeans_final_model = grid_search_best_model_clustering(kmeans_estimator, kmeans_grid, X, scoring=scoring)
plot_model_2d_kmeans(kmeans_final_model, X)

X, y = generate_nonlinear_synthetic_data_classification2(n_samples=300)
plot_data_2d(X)

kmeans_estimator = cluster.KMeans()
kmeans_grid = {'n_clusters':list(range(2,6))}
grid_search_plot_models_kmeans(kmeans_estimator, kmeans_grid, X)
kmeans_final_model = grid_search_best_model_clustering(kmeans_estimator, kmeans_grid, X, scoring=scoring)
plot_model_2d_kmeans(kmeans_final_model, X)

