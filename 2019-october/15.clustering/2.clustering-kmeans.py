import sys
path = 'J://utils'
sys.path.append(path)

from sklearn import cluster
import common_utils as utils
import clustering_utils as cl_utils

X, _= cl_utils.generate_synthetic_data_2d_clusters(n_samples=300, n_centers=4, cluster_std=0.60)
utils.plot_data_2d(X)

kmeans = cluster.KMeans(5)
kmeans.fit(X)
print(kmeans.cluster_centers_)
print(kmeans.labels_)
cl_utils.plot_model_2d_clustering(kmeans, X)

scoring = 's_score'
kmeans_estimator = cluster.KMeans()
kmeans_grid = {'n_clusters':list(range(3,9))}
kmeans_final_model = cl_utils.grid_search_best_model_clustering(kmeans_estimator, kmeans_grid, X, scoring=scoring)
print(kmeans_final_model.labels_)
print(kmeans_final_model.cluster_centers_)
cl_utils.plot_model_2d_clustering(kmeans_final_model, X)
