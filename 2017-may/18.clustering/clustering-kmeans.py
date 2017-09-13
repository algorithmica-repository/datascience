from sklearn import cluster
from sklearn import metrics
from sklearn.datasets import make_blobs

    
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1) 

kmeans_model = cluster.KMeans(n_clusters=6, random_state=1)
kmeans_model.fit(X)
kmeans_model.cluster_centers_
kmeans_model.labels_

silhouette_avg = metrics.silhouette_score(X,kmeans_model.labels_,metric='euclidean')
silhouette_samples = metrics.silhouette_samples(X,kmeans_model.labels_,metric='euclidean')
ch_score = metrics.calinski_harabaz_score(X,kmeans_model.labels_)

range_n_clusters = [2, 3, 4, 5, 6]

for n_clusters in range_n_clusters:
    cluster_model = cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = cluster_model.fit_predict(X)
    silhouette_avg = metrics.silhouette_score(X,cluster_labels,metric='euclidean')
    ch_score = metrics.calinski_harabaz_score(X,cluster_labels)
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is:", silhouette_avg)
    print("For n_clusters =", n_clusters, 
          "The ch_score is:", ch_score)