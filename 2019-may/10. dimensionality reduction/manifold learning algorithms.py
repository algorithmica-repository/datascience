from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
import numpy as np
import matplotlib.pyplot as plt

x1 = np.random.multivariate_normal([-3,3], np.eye(2), 100)
x2 = np.random.multivariate_normal([3,3], np.eye(2), 100)
x3 = np.random.multivariate_normal([0,-10], np.eye(2), 100)
xs = np.r_[x1, x2, x3]
xs = (xs - xs.mean(0))/xs.std()
zs = np.r_[np.zeros(100), np.ones(100), 2*np.ones(100)]

plt.scatter(xs[:, 0], xs[:, 1], c=zs)
plt.show()

pca = PCA(n_components=1)
ys = pca.fit_transform(xs)
plt.scatter(ys[:, 0], np.random.uniform(-1, 1, len(ys)), c=zs)
plt.axhline(0, c='red')
plt.show()

mds = MDS()
ms = mds.fit_transform(xs)
plt.scatter(ms[:, 0], np.random.uniform(-1, 1, len(ms)), c=zs)
plt.axhline(0, c='red')

tsne = TSNE(n_components=1)
ts = tsne.fit_transform(xs)
plt.scatter(ts[:, 0], np.random.uniform(-1, 1, len(ts)), c=zs)
plt.axhline(0, c='red')