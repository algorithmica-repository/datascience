import sys
sys.path.append("E:/New Folder/utils")

import classification_utils as cutils
from sklearn import preprocessing
import numpy as np

X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
X, y = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=1000, noise=0.1)

cutils.plot_data_2d_classification(X, y)

#guassian basis transformation
tmp = np.exp(-(X ** 2).sum(1))
X_3d = np.c_[X, tmp]
cutils.plot_data_3d_classification(X_3d, y, new_window=True, title="Linearly separable data in 3D with basis change")

#polynomial basis transformation
poly_features = preprocessing.PolynomialFeatures()
X_poly1 = poly_features.fit_transform(X)

poly_features = preprocessing.PolynomialFeatures(degree=3)
X_poly2 = poly_features.fit_transform(X)
