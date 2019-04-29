import sys
path = 'E://'
sys.path.append(path)

from common_utils  import get_continuous_features, get_categorical_features, \
    cast_cont_to_cat, get_categorical_imputers, get_continuous_imputers, \
    get_features_to_drop_on_missingdata, ohe, drop_features, get_scaler, \
    get_zero_variance_filter, plot_feature_importances, select_features, grid_search_plot_one_parameter_curves, \
    grid_search_plot_two_parameter_curves, get_best_model, plot_data_1d, plot_data_2d, plot_data_3d
from classification_utils import  \
    generate_nonlinear_synthetic_data_classification1, generate_nonlinear_synthetic_data_classification2, \
    generate_linear_synthetic_data_classification, \
    plot_model_2d_classification, plot_data_2d_classification, plot_data_3d_classification, \
    grid_search_plot_models_classification
from regression_utils import generate_nonlinear_synthetic_data_regression, generate_nonlinear_synthetic_sine_data_regression, generate_linear_synthetic_data_regression, \
    plot_model_2d_regression, plot_model_3d_regression, plot_data_2d_regression, plot_data_3d_regression, \
    grid_search_plot_models_regression, plot_coefficients_regression, \
    plot_target_and_transformed_target_regression, rmse, regression_performance
from feature_reduction_utils import feature_reduction_linear_pca, feature_reduction_kernel_pca, \
    feature_reduction_tsne, feature_reduction_isomap
from kernel_utils import GaussianFeatures, KernelTransformer
from sklearn.model_selection import train_test_split
from sklearn import metrics, decomposition, manifold
from sklearn import tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

X, y = generate_linear_synthetic_data_classification(n_samples=1000, n_features=3, n_redundant=0, n_classes=3, weights=[.3,.3,.4])
plot_data_3d(X)
X_lpca = feature_reduction_linear_pca(X, 2)
plot_data_2d(X_lpca, new_window=True)
X_kpca = feature_reduction_kernel_pca(X, 2)
plot_data_2d(X_kpca, new_window=True)
X_tsne = feature_reduction_tsne(X, 2)
plot_data_2d(X_tsne, new_window=True)
X_isomap = feature_reduction_isomap(X, 2)
plot_data_2d(X_isomap, new_window=True)

X, y = generate_nonlinear_synthetic_data_classification2(n_samples=1000)
plot_data_2d(X)
X_lpca = feature_reduction_linear_pca(X, 2)
plot_data_2d(X_lpca, new_window=True)
X_kpca = feature_reduction_kernel_pca(X, 2, 'rbf', 15)
plot_data_2d(X_kpca, new_window=True)
X_tsne = feature_reduction_tsne(X, 2)
plot_data_2d(X_tsne, new_window=True)
X_isomap = feature_reduction_isomap(X, 2, 100)
plot_data_2d(X_isomap, new_window=True)

X, y = generate_linear_synthetic_data_regression(n_samples=100, n_features=2, n_informative=2, noise=0)
plot_data_2d(X)
X_lpca = feature_reduction_linear_pca(X, 2)
plot_data_2d(X_lpca, new_window=True)
X_kpca = feature_reduction_kernel_pca(X, 2, 'rbf', 15)
plot_data_2d(X_kpca, new_window=True)
X_tsne = feature_reduction_tsne(X, 2)
plot_data_2d(X_tsne, new_window=True)
X_isomap = feature_reduction_isomap(X, 2)
plot_data_2d(X_isomap, new_window=True)
