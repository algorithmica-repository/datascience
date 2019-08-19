import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'E://'
sys.path.append(path)

import common_utils as utils
import pca_utils as putils
import tsne_utils as tutils
import classification_utils as cutils
from sklearn import metrics, decomposition, manifold
from sklearn import tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_redundant=0, n_classes=2, class_sep=0, weights=[.5,.5])
utils.plot_data_2d(X)
X = pd.DataFrame(X, columns=['X1', 'X2'])

lpca = decomposition.PCA(2)
lpca.fit(X)
putils.plot_pca_result(lpca, X)

X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=3, n_redundant=1, n_classes=2, weights=[.5,.5])
utils.plot_data_3d(X)
X = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
print(X.corr())

lpca = decomposition.PCA(3)
lpca.fit(X)
putils.plot_pca_result(lpca, X)

X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=3, n_redundant=2, n_classes=2, weights=[.5,.5])
utils.plot_data_3d(X)
X = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])
print(X.corr())

lpca = decomposition.PCA(3)
lpca.fit(X)
print(lpca.explained_variance_)
print(lpca.explained_variance_ratio_)
print(lpca.components_)
putils.plot_pca_result(lpca, X)

X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000)
utils.plot_data_2d(X)
lpca = decomposition.PCA(2)
lpca.fit(X)
print(lpca.explained_variance_)
print(lpca.explained_variance_ratio_)
print(lpca.components_)
X = pd.DataFrame(X, columns=['X1', 'X2'])
putils.plot_pca_result(lpca, X)

tutils.plot_tsne_result(X, y, 2)

