import sys
path_to_scripts = 'G://'
sys.path.append(path_to_scripts)

from common_utils  import get_continuous_features, get_categorical_features, \
    cast_cont_to_cat, get_categorical_imputers, get_continuous_imputers, \
    get_features_to_drop_on_missingdata, ohe, drop_features, get_scaler, \
    get_zero_variance_filter, feature_reduction_pca, feature_reduction_tsne, \
    plot_feature_importances, select_features, grid_search_plot_one_parameter_curves, \
    grid_search_plot_two_parameter_curves, get_best_model
from classification_utils import generate_synthetic_data_outliers, \
    plot_model_2d_classification, plot_data_2d_classification, \
    grid_search_plot_models_classification
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
import pandas as pd
import numpy as np

outliers_fraction = 0.25
cluster_separation = 2
n_samples = 200
X, y = generate_synthetic_data_outliers(n_samples, outliers_fraction, cluster_separation)
plot_data_2d_classification(X, y, xlim=[-7,7], ylim=[-7,7])

iso_forest_estimator = ensemble.IsolationForest()
iso_forest_grid = {'contamination':[0.1, 0.2, 0.25, 0.3]}
grid_search_plot_models_classification(iso_forest_estimator, iso_forest_grid, X, y, xlim=[-7,7], ylim=[-7,7], outlier_estimator=True)

cov_estimator = covariance.EllipticEnvelope()
cov_grid = {'contamination':[0.1, 0.2, 0.25, 0.3]}
grid_search_plot_models_classification(cov_estimator, cov_grid, X, y, xlim=[-7,7], ylim=[-7,7], outlier_estimator=True)

svm_estimator = svm.OneClassSVM(kernel="rbf", gamma=0.1)
tmp = 0.95 * outliers_fraction
svm_grid = {'nu':[tmp+0.03, tmp+0.05, tmp+0.06, tmp+0.07]}
grid_search_plot_models_classification(svm_estimator, svm_grid, X, y, xlim=[-7,7], ylim=[-7,7], outlier_estimator=True)