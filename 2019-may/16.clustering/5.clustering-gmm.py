import sys
path = 'E://utils'
sys.path.append(path)

from sklearn import cluster, mixture
import common_utils as utils
import clustering_utils as cl_utils

X, _= cl_utils.generate_synthetic_data_2d_clusters(n_samples=300, n_centers=4, cluster_std=0.60)
utils.plot_data_2d(X)

scoring = 's_score'
gmm_estimator = mixture.GaussianMixture(n_components=3)
gmm_grid = {'n_components':list(range(10,40))}
gmm_estimator.fit(X)
gmm_estimator.predict(X)
gmm_final_model = cl_utils.grid_search_best_model_clustering(gmm_estimator, gmm_grid, X, scoring=scoring)
cl_utils.plot_model_2d_clustering(gmm_estimator, X)
