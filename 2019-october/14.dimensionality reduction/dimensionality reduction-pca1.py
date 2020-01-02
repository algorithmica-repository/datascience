import sys
path = 'I:/utils'
sys.path.append(path)
import common_utils as utils
import pca_utils as putils
import tsne_utils as tutils
import classification_utils as cutils
import pandas as pd
import numpy as np
from sklearn import decomposition

#pca effect on linearly related data
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_redundant=0, n_classes=2, class_sep=0, weights=[.5,.5])
X = pd.DataFrame(X, columns=['X1', 'X2'])
utils.plot_data_2d(X)
print(X.corr())
lpca = decomposition.PCA(2)
lpca.fit(X)
print(lpca.components_)
print(lpca.explained_variance_)
print(lpca.explained_variance_ratio_)
np.cumsum(lpca.explained_variance_ratio_)
putils.plot_pca_result(lpca, X)

#pca effect on linearly related data(1 redundant feature)
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=3, n_redundant=1, n_classes=2, weights=[.5,.5])
X = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
utils.plot_data_3d(X)
print(X.corr())
lpca = decomposition.PCA(2)
lpca.fit(X)
print(lpca.explained_variance_)
print(lpca.explained_variance_ratio_)
np.cumsum(lpca.explained_variance_ratio_)
putils.plot_pca_result(lpca, X)

#pca effect on linearly related data(2 redundant featues)
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=3, n_redundant=2, n_classes=2, weights=[.5,.5])
X = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
utils.plot_data_3d(X)
print(X.corr())
lpca = decomposition.PCA(1)
lpca.fit(X)
print(lpca.explained_variance_)
print(lpca.explained_variance_ratio_)
np.cumsum(lpca.explained_variance_ratio_)
putils.plot_pca_result(lpca, X)

#pca effect on non-linearly related data
X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000)
X = pd.DataFrame(X, columns=['X1', 'X2'])
utils.plot_data_2d(X)
print(X.corr())
lpca = decomposition.PCA(2)
lpca.fit(X)
print(lpca.explained_variance_)
print(lpca.explained_variance_ratio_)
np.cumsum(lpca.explained_variance_ratio_)
putils.plot_pca_result(lpca, X)

tutils.plot_tsne_result(X, y, 2)
