import numpy as np
from sklearn import decomposition
X1 = np.array([[10, 20, 10], [2, 5, 2], [8, 17, 7], [9, 20, 10], [12, 22, 11]])
pca = decomposition.PCA(n_components=3)
pca.fit(X1)

#variance of data along original axes
np.var([10,2,8,9,12]) + np.var([20,5,17,20,22]) + np.var([10,2,7,10,11])
#variance of data along principal component axes
np.sum(pca.explained_variance_)

#understand how much variance captured by each principal component
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.explained_variance_ratio_.cumsum())

#show the principal components
pca.components_[0]
pca.components_[1]
pca.components_[2]

#transform data from original axes to princicpal components axes
X2 = pca.transform(X1)
X_original = pca.inverse_transform(X2)

#specify number of required dimensions as n_components
pca = decomposition.PCA(n_components=2)
pca.fit(X1)
pca.explained_variance_
pca.components_[0]
pca.components_[1]
X3 = pca.transform(X1)