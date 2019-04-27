import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

from common_utils  import *
from outlier_utils import *
from feature_reduction_utils import *
from sklearn.model_selection import train_test_split
from sklearn import metrics, preprocessing, tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection, kernel_ridge
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import numpy as np
 
card_data = pd.read_csv(os.path.join(path,'creditcard.csv'))
card_data.info()
X = drop_features(card_data, ['Time', 'Amount', 'Class'])
y = card_data['Class']

tnse_data = feature_reduction_tsne(X, 3)
plot_data_3d_outliers(tnse_data, y, title="Credit card data")

iso_forest_estimator = ensemble.IsolationForest()
iso_forest_grid = {'contamination':[0.1, 0.2, 0.25, 0.3]}
grid_search_plot_models_outliers(iso_forest_estimator, iso_forest_grid, X, y, xlim=[-7,7], ylim=[-7,7])
iso_best_model = grid_search_best_model_outliers(iso_forest_estimator, iso_forest_grid, X, y, scoring='roc_auc')
plot_model_2d_outliers(iso_best_model, X, y)