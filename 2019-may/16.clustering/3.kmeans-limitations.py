import sys
path = 'E://utils'
sys.path.append(path)

from sklearn import cluster
import common_utils as utils
import clustering_utils as cl_utils
import classification_utils as cutils

X, _ = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=300)
utils.plot_data_2d(X)

X, _ = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=300)
utils.plot_data_2d(X)

scoring = 's_score'
kmeans_estimator = cluster.KMeans()
kmeans_grid = {'n_clusters':list(range(2,7))}
kmeans_final_model = cl_utils.grid_search_best_model_clustering(kmeans_estimator, kmeans_grid, X, scoring=scoring)
print(kmeans_final_model.labels_)
print(kmeans_final_model.cluster_centers_)
cl_utils.plot_model_2d_clustering(kmeans_final_model, X)
