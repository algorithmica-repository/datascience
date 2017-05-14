import pandas as pd
from sklearn import cluster
from sklearn import metrics

votes = pd.read_csv("E:/congress.csv")
votes.shape
kmeans_model = cluster.KMeans(n_clusters=4, random_state=1)
kmeans_model.fit(votes.iloc[:, 3:])

labels = kmeans_model.labels_

silhouette_avg = metrics.silhouette_score(votes.iloc[:, 3:],labels,metric='euclidean')
silhouette_samples = metrics.silhouette_samples(votes.iloc[:, 3:],labels,metric='euclidean')
ch_score = metrics.calinski_harabaz_score(votes.iloc[:, 3:],labels)

for n_clusters in range(2,10):
    cluster_model = cluster.KMeans(n_clusters=n_clusters, random_state=10)
    cluster_labels = cluster_model.fit_predict(votes.iloc[:, 3:])
    silhouette_avg = metrics.silhouette_score(votes.iloc[:, 3:],cluster_labels,metric='euclidean')
    ch_score = metrics.calinski_harabaz_score(votes.iloc[:, 3:],cluster_labels)
    print("For n_clusters =", n_clusters, 
          "The average silhouette_score is:", silhouette_avg)
    print("For n_clusters =", n_clusters, 
          "The ch_score is:", ch_score)

