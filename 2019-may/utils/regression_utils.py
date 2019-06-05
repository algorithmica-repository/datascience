import os
import pydot
import io
import pandas as pd
import seaborn as sns
import numpy as np
import math
from itertools import product
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LightSource
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, tree, svm, neighbors, metrics, linear_model, manifold, linear_model, ensemble
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn import model_selection, metrics, ensemble, preprocessing, decomposition, feature_selection
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles, make_blobs, make_moons, make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
import sklearn

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred) )

def regression_performance(estimator, X, y):
    y_pred = estimator.predict(X)
    print("rmse:" + str(rmse(y, y_pred)))
    print("r2:" + str(metrics.r2_score(y, y_pred)))
   
def generate_nonlinear_synthetic_sine_data_regression(n_samples):
    np.random.seed(0)
    X = np.random.normal(size=n_samples)
    y =  np.sin(X)
    X = X.reshape(-1,1)
    return X, y

def generate_nonlinear_synthetic_data_regression(n_samples, n_features):
    np.random.seed(0)
    if n_features == 1:
        X = np.linspace(0, 1, n_samples)
        y =  np.sin(2 * np.pi * X)
        X = X.reshape(-1,1)
    elif n_features == 2:
        X1 = np.linspace(0, 1, n_samples)
        X2 = np.linspace(0, 1, n_samples)
        y = 0.5 + 1.5 * X1**2 + 1.6 * X2**3
        X = np.array([X1,X2]).T
    else:
        print("error")
    y += np.random.normal(scale=0.3, size=n_samples)
    return X, y

def generate_linear_synthetic_data_regression(n_samples, n_features, n_informative, noise):
    X, y = make_regression(n_samples=n_samples, n_features=n_features, 
                           n_informative=n_informative, 
                           random_state=0, noise=noise)
    return X, y


def plot_data_2d_regression(X, y, ax = None, x_limit=None, y_limit=None, title=None, new_window=False, color='blue'):
    if isinstance(X, np.ndarray) :
        labels =['X'+str(i) for i in range(X.shape[1])]
    else:
        labels = X.columns
        X = X.values
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()   
    if y_limit:
        ax.set_ylim(y_limit[0], y_limit[1])
    if x_limit:
        ax.set_xlim(x_limit[0], x_limit[1])
    ax.scatter(X, y, c=color)
    ax.set_xlabel(labels[0])
    ax.set_ylabel("target")
    ax.set_title(title)
    return ax
    
def plot_model_2d_regression(estimator, X, y, ax=None, x_limit=None, y_limit=None, title=None, new_window=False, color_model='red', color_data='blue'):
    ax = plot_data_2d_regression(X, y, ax, x_limit, y_limit, title, new_window, color_data)
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    xx = np.arange(x_min, x_max, 0.1)
    y_pred = estimator.predict(xx.reshape(-1, 1))
    ax.plot(xx, y_pred, color=color_model)
    plt.tight_layout()    


def plot_data_3d_regression(X, y, ax=None, x_limit=None, y_limit=None, z_limit=None, title=None, new_window=False, rotation=False, color='blue'):
    aux_plot_data_3d_regression(X, y, ax, x_limit, y_limit, z_limit, title, new_window, rotation, color, False)
    
def aux_plot_data_3d_regression(X, y, ax=None, x_limit=None, y_limit=None, z_limit=None, title=None, new_window=False, rotation=False, color='blue', inner_call=False):
    if isinstance(X, np.ndarray) :
        labels =['X'+str(i) for i in range(X.shape[1])]
    else:
        labels = X.columns
        X.values
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes(projection='3d')  
    if x_limit:
        ax.set_xlim(x_limit[0], x_limit[1])
    if y_limit:
        ax.set_ylim(y_lim[0], y_limit[1])
    if z_limit:
        ax.set_zlim(z_limit[0], z_limit[1])
    ax.scatter(X[:, 0], X[:, 1], y, s=30, c = color)  
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel("target")
    ax.set_title(title)

    if rotation and not inner_call:
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.1)
    return ax

def plot_model_3d_regression(estimator, X, y, ax=None, x_limit=None, y_limit=None, z_limit=None, title=None, new_window=False, rotation=False):
    ax = aux_plot_data_3d_regression(X, y, ax, x_limit, y_limit, z_limit, title, new_window, rotation, inner_call=True)
     
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    ax.plot_surface(xx, yy, Z)
    plt.tight_layout()    
    
    if rotation:
        for angle in range(0, 360):
            ax.view_init(20, angle)
            plt.draw()
            plt.pause(.1) 

def grid_search_plot_models_regression(estimator, grid, X, y, xlim=None, ylim=None):
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
        estimator.fit(X, y)        
        plot_model_2d_regression(estimator, X, y, ax, xlim, ylim, str(param))
    plt.tight_layout()
    
def plot_coefficients_regression(estimator, X, y, n_alphas=200):
    alphas = np.logspace(-6, 6, n_alphas)

    coefs = []
    for a in alphas:
        if isinstance(estimator, sklearn.pipeline.Pipeline) :
            estimator.set_params(estimator__alpha=a)
            estimator.fit(X, y)
            tmp = estimator.named_steps['estimator'].coef_
        else :
            estimator.set_params(alpha=a)
            estimator.fit(X, y)
            tmp = estimator.coef_
        coefs.append(tmp)

    ax = plt.gca()
    ax.plot(alphas, coefs)
    ax.set_xscale('log')
    plt.xlabel('alpha')
    plt.ylabel('weights')
    plt.title('Coefficients as a function of the regularization')
    plt.axis('tight')

def plot_target_and_transformed_target_regression(y, y_trans):
    fig, (ax0, ax1) = plt.subplots(1, 2)

    ax0.hist(y, bins=100)
    ax0.set_xlim([0, 2000])
    ax0.set_ylabel('Probability')
    ax0.set_xlabel('Target')
    ax0.set_title('Target distribution')

    ax1.hist(y_trans, bins=100)
    ax1.set_ylabel('Probability')
    ax1.set_xlabel('Target')
    ax1.set_title('Transformed target distribution')
    fig.tight_layout()
    