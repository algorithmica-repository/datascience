import pandas as pd
from sklearn import cluster
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

votes = pd.read_csv("E:/congress.csv")
votes.shape
kmeans_model = cluster.KMeans(n_clusters=2, random_state=1)
kmeans_model.fit(votes.iloc[:, 3:])
kmeans_model.cluster_centers_
labels = kmeans_model.labels_

pd.crosstab(index=labels, columns="count")

print(pd.crosstab(labels, votes["party"]))

democratic_oddballs = votes[(labels == 1) & (votes["party"] == "D")]
print(democratic_oddballs["name"])

republican_oddballs = votes[(labels == 0) & (votes["party"] == "R")]
print(republican_oddballs["name"])

pca_2 = PCA(2)

plot_columns = pca_2.fit_transform(votes.iloc[:,3:18])

plt.scatter(x=plot_columns[:,0], y=plot_columns[:,1], c=labels)
plt.show()
