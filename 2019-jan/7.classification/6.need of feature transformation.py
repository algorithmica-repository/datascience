import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

from common_utils  import *
from classification_utils import *
from kernel_utils import *
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing, tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
from sklearn.preprocessing import PolynomialFeatures
import numpy as np

#impact of basis change for classification pattern      
X, y = generate_nonlinear_synthetic_data_classification2(1000, 0.1)
plot_data_2d_classification(X, y, title="Non-linear separable data in 2D")

tmp = np.exp(-(X ** 2).sum(1))
X_3d = np.c_[X, tmp]
plot_data_3d_classification(X_3d, y, new_window=True, title="Linearly separable data in 3D with basis change")

#impact of basis change for regression pattern      
X, y = generate_nonlinear_synthetic_sine_data_regression(100)
plot_data_2d_regression(X, y, title="Non-linear data in 2D")

tmp = np.exp(-(X ** 2).sum(1))
X_2d = np.c_[X, tmp]
plot_data_3d_regression(X_2d, y, new_window=True, title="Linear data in 3D with basis change")
