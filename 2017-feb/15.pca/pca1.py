import numpy as np
from sklearn import decomposition
X1 = np.array([[10, 20, 10], [2, 5, 2], [8, 17, 7], [9, 20, 10], [12, 22, 11]])
pca = decomposition.PCA(n_components=2)
pca.fit(X1)
pca.components_[0]
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
X2 = pca.transform(X1)
X_original = pca.inverse_transform(X2)

X3 = np.array([[10, 20], [12, 8], [20, 30], [40, 25]])
pca = decomposition.PCA(n_components=2)
pca.fit(X3)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())
X4 = pca.transform(X3)
