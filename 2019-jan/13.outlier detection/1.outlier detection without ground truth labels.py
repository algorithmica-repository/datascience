import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

from common_utils  import *
from outlier_utils import *
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing, tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection, kernel_ridge
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
 
n_samples = 400
outliers_fraction = 0.25
cluster_separation = 2
X, y = generate_synthetic_data_outliers(n_samples, outliers_fraction, cluster_separation)
plot_data_2d_outliers(X, xlim=[-7,7], ylim=[-7,7])

iso_forest_estimator = ensemble.IsolationForest()
iso_forest_grid = {'contamination':[0.1, 0.2, 0.25, 0.3]}
grid_search_plot_models_outliers(iso_forest_estimator, iso_forest_grid, X, xlim=[-7,7], ylim=[-7,7])

cov_estimator = covariance.EllipticEnvelope()
cov_grid = {'contamination':[0.1, 0.2, 0.25, 0.3]}
grid_search_plot_models_outliers(cov_estimator, cov_grid, X, xlim=[-7,7], ylim=[-7,7])

svm_estimator = svm.OneClassSVM(kernel="rbf", gamma=0.1)
tmp = 0.95 * outliers_fraction
svm_grid = {'nu':[tmp+0.03, tmp+0.05, tmp+0.06, tmp+0.07]}
grid_search_plot_models_outliers(svm_estimator, svm_grid, X, xlim=[-7,7], ylim=[-7,7])