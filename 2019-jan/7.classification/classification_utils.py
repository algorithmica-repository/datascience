import os
import pandas as pd
import seaborn as sns
import numpy as np
import math
from itertools import product
from sklearn import covariance, preprocessing, tree, svm, neighbors, metrics, linear_model, manifold, linear_model
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn import model_selection, ensemble, preprocessing, decomposition, feature_selection
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D

def generate_synthetic_data_outliers(n_samples, outliers_fraction, cluster_separation):
    n_inliers = int((1. - outliers_fraction) * n_samples)
    n_outliers = int(outliers_fraction * n_samples)
    y = np.ones(n_samples, dtype=int)
    y[-n_outliers:] = 0
    
    np.random.seed(42)
    X1 = 0.3 * np.random.randn(n_inliers // 2, 2) - cluster_separation
    X2 = 0.3 * np.random.randn(n_inliers // 2, 2) + cluster_separation
    X = np.r_[X1, X2]
    X = np.r_[X, np.random.uniform(low=-6, high=6, size=(n_outliers, 2))]
    return X, y

def plot_model_2d_classification(estimator, X, y, ax = None, xlim=None, ylim=None, title=None, new_window=False, levels=[-1, 0, 1]):
    ax = plot_data_2d_classification(X, y, ax, xlim, ylim, title, new_window)
    if xlim and ylim:
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    else:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contour(xx, yy, Z, levels=levels, linewidths=2, colors='red', alpha=0.5)
    plt.tight_layout()

def plot_data_2d_classification(X, y, ax = None, xlim=None, ylim=None, title=None, new_window=False):
    if isinstance(X, np.ndarray) :
        labels =['X'+str(i) for i in range(X.shape[1])]
    else:
        X = X.values()
        labels = X.columns
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()   
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
        
    colors = "bry"
    n_classes = set(y)
    class_labels = [str(i) for i in n_classes]
    for i, color in zip(n_classes, colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=color, label = class_labels[i],
                    cmap=plt.cm.Paired, edgecolor='black', s=30) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    return ax

def grid_search_plot_models_classification(estimator, grid, X, y, xlim=None, ylim=None, outlier_estimator=False, levels=[-1,0,1]):
    items = sorted(grid.items())
    keys, values = zip(*items)
    params =[]
    for v in product(*values):
        params.append(dict(zip(keys, v)))
    n = len(params)
    fig, axes = plt.subplots(int(math.sqrt(n)), math.ceil(math.sqrt(n)), figsize=(20, 20), dpi=80)
    print(axes)
    axes = np.array(axes)
    for ax, param in zip(axes.reshape(-1), params):
        estimator.set_params(**param)
        if outlier_estimator:
            estimator.fit(X)
        else:
            estimator.fit(X, y)        
        plot_model_2d_classification(estimator, X, y, ax, xlim, ylim, str(param), False, levels)
    plt.tight_layout()