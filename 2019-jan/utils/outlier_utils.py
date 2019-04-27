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
from sklearn.datasets import make_circles, make_moons, make_classification

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

def plot_data_2d_outliers(X, y=None, ax = None, xlim=None, ylim=None, title=None, new_window=True):
    plt.style.use('seaborn')
    if isinstance(X, np.ndarray) :
        labels =['X'+str(i) for i in range(X.shape[1])]
    else:
        labels = X.columns
        X = X.values
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()   
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
      
    if y is not None:
        colors = "ryb"
        n_classes = set(y)
        class_labels = [str(i) for i in n_classes]
        for i, color in zip(n_classes, colors):
            idx = np.where(y == i)
            ax.scatter(X[idx, 0], X[idx, 1], c=color, label = class_labels[i],
                       cmap=plt.cm.RdYlBu, edgecolor='black', s=30) 
    else:
        ax.scatter(X[:, 0], X[:, 1], c='red', cmap=plt.cm.coolwarm, edgecolor='black', s=30) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()

def plot_data_3d_outliers(X, y=None, ax = None, xlim=None, ylim=None, zlim=None, title=None, new_window=True, rotation=False):
    plt.style.use('seaborn')
    if isinstance(X, np.ndarray) :
        labels =['X'+str(i) for i in range(X.shape[1])]
    else:
        labels = X.columns
        X = X.values
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes(projection='3d')   
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
    if zlim:
        ax.set_zlim(zlim[0], zlim[1])     
      
    if y is not None:
        colors = "ryb"
        n_classes = set(y)
        class_labels = [str(i) for i in n_classes]
        for i, color in zip(n_classes, colors):
            idx = np.where(y == i)
            ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], c=color, label = class_labels[i],
                       cmap=plt.cm.RdYlBu, edgecolor='black', s=30) 
    else:
        ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='red', cmap=plt.cm.coolwarm, edgecolor='black', s=30) 
        
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    ax.set_title(title)
    ax.legend()
    if rotation:
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.1)

def plot_model_2d_outliers(estimator, X, y=None, ax = None, xlim=None, ylim=None, title=None, new_window=True, levels=None):
    plt.style.use('seaborn')
    if isinstance(X, np.ndarray) :
        labels =['X'+str(i) for i in range(X.shape[1])]
    else:
        labels = X.columns
        X = X.values
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()   
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
        
    if xlim and ylim:
        xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 500), np.linspace(ylim[0], ylim[1], 500))
    else:
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if levels:
        ax.contour(xx, yy, Z, levels=levels, linewidths=2, colors='red', alpha=1)
    else:
        ax.contourf(xx, yy, Z, cmap=plt.cm.Pastel1, alpha=1)
    
    if y is not None:
        y_pred = estimator.predict(X)
        #print(y_pred)
        y_pred[y_pred == -1] = 0
        f_score = metrics.f1_score(y, y_pred)
        
        colors = "ryb"
        n_classes = set(y)
        class_labels = [str(i) for i in n_classes]
        for i, color in zip(n_classes, colors):
            idx = np.where(y == i)
            ax.scatter(X[idx, 0], X[idx, 1], c=color, label = class_labels[i],
                       cmap=plt.cm.coolwarm, edgecolor='black', s=30) 
        ax.text(4,4, 'f_score:' + str(f_score), ha='center', va='center', fontsize=15)
    else:
        ax.scatter(X[:, 0], X[:, 1], c='red', cmap=plt.cm.coolwarm, edgecolor='black', s=30) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()

def grid_search_plot_models_outliers(estimator, grid, X, y=None, xlim=None, ylim=None, levels=None):
    plt.style.use('seaborn')
    items = sorted(grid.items())
    keys, values = zip(*items)
    params =[]
    for v in product(*values):
        params.append(dict(zip(keys, v)))
    n = len(params)
    fig, axes = plt.subplots(int(math.sqrt(n)), math.ceil(math.sqrt(n)), figsize=(20, 20), dpi=80)
    axes = np.array(axes)
    for ax, param in zip(axes.reshape(-1), params):
        estimator.set_params(**param)
        estimator.fit(X)  
        plot_model_2d_outliers(estimator, X, y, ax, xlim, ylim, str(param), False, levels)
    plt.tight_layout()

def grid_search_best_model_outliers(estimator, grid, X, y, scoring):
    items = sorted(grid.items())
    keys, values = zip(*items)
    params =[]
    for v in product(*values):
        params.append(dict(zip(keys, v)))
    n = len(params)
    best_param = None
    best_score = 0.0
    for param in params:
        estimator.set_params(**param)
        estimator.fit(X)  
        y_pred = estimator.predict(X)
        y_pred[y_pred == -1] = 0
        if scoring == 'roc_auc':
            score = metrics.roc_auc_score(y, y_pred)
        elif scoring == 'f1':
            score =  metrics.f1_score(y, y_pred)
        else:
            print(scoring+' metric not supported')
            break
        if score > best_score :
            best_score = score
            best_param = param
    if best_param is not None:
        estimator.set_params(**best_param)
        estimator.fit(X)
        return estimator
    else:
        return None