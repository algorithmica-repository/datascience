from sklearn import cluster
from sklearn import metrics
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import pandas as pd

def plot_data(X, y, figsize=None):
    if not figsize:
        figsize = (8, 6)
    plt.figure(figsize=figsize)
    plt.plot(X[y==0, 0], X[y==0, 1], 'or', alpha=0.5, label=0)
    plt.plot(X[y==1, 0], X[y==1, 1], 'ob', alpha=0.5, label=1)
    plt.plot(X[y==2, 0], X[y==2, 1], 'oy', alpha=0.5, label=2)
    plt.plot(X[y==3, 0], X[y==3, 1], 'om', alpha=0.5, label=3)
    plt.xlim((min(X[:, 0])-0.1, max(X[:, 0])+0.1))
    plt.ylim((min(X[:, 1])-0.1, max(X[:, 1])+0.1))
    plt.legend()
    
X, y = make_blobs(n_samples=500,
                  n_features=2,
                  centers=4,
                  cluster_std=1,
                  center_box=(-10.0, 10.0),
                  shuffle=True,
                  random_state=1) 
plot_data(X, y)

kmeans_model = cluster.KMeans(n_clusters=4, random_state=1)
kmeans_model.fit(X)
kmeans_model.cluster_centers_
kmeans_model.labels_

#metrics when target labels are not known
silhouette_avg = metrics.silhouette_score(X,kmeans_model.labels_,metric='euclidean')
print(silhouette_avg)
silhouette_samples = metrics.silhouette_samples(X,kmeans_model.labels_,metric='euclidean')
print(silhouette_samples)
ch_score = metrics.calinski_harabaz_score(X,kmeans_model.labels_)
print(ch_score)

#metrics when target labels are known
df = pd.DataFrame({'GT':y, 'Pred':kmeans_model.labels_})
print(metrics.adjusted_rand_score(y, kmeans_model.labels_))
print(metrics.adjusted_mutual_info_score(y, kmeans_model.labels_))
