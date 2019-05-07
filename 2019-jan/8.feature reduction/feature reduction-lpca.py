import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

from common_utils import *
from pca_utils import *
from classification_utils import *
from sklearn import metrics, decomposition, manifold
from sklearn import tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X, y = generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_redundant=0, n_classes=2, weights=[.5,.5])
plot_data_2d(X)
X = pd.DataFrame(X, columns=['X1', 'X2'])

lpca = decomposition.PCA(2)
lpca.fit(X)
plot_pca_result(lpca, X)

X, y = generate_linear_synthetic_data_classification(n_samples=1000, n_features=3, n_redundant=1, n_classes=2, weights=[.5,.5])
plot_data_3d(X)
X = pd.DataFrame(X, columns=['X1', 'X2', 'X3'])

lpca = decomposition.PCA(2)
lpca.fit(X)
plot_pca_result(lpca, X)
