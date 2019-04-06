import os
import pydot
import io
import pandas as pd
import seaborn as sns
import numpy as np
import math
from matplotlib import cm
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing, tree, svm, neighbors, metrics, linear_model, manifold, linear_model, ensemble
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn import model_selection, metrics, ensemble, preprocessing, decomposition, feature_selection
from sklearn.model_selection import validation_curve, train_test_split
from sklearn.datasets import make_classification
from sklearn.datasets import make_circles, make_blobs, make_moons, make_regression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def rmse(y_orig, y_pred):
    return math.sqrt(metrics.mean_squared_error(y_orig,y_pred) )

def generate_data1(n_samples):
    np.random.seed(0)
    X = np.random.rand(n_samples)
    y = np.cos(2 * np.pi * X) + np.random.randn(n_samples) * 0.1
    return X.reshape(-1,1), y

def generate_data2():
    X = np.array([0.1,0.13,0.2, 0.22,0.25, 0.3, 0.5, 0.6, 0.65, 0.7, 0.5 ])
    y = np.array([0.9,0.88, 1, 0.5, 0.6, 0.5, -0.5, -0.4, -0.3, -0.42, -0.75 ])
    return X.reshape(-1, 1), y

def generate_data(n_samples, n_features):
    np.random.seed(0)
    X = 6 * np.random.rand(n_samples, n_features)
    if n_features == 1:
        y = np.sin(X).ravel()
        y[::5] += 2 * (0.5 - np.random.rand((int)(n_samples/5)) )
    elif n_features == 2:
        y = 0.5 + 1.5 * X[:,0]**2 + 1.6 * X[:,1]**3
    return X, y

def plot_data_2d(X, y, limit_y=None):
    labels =['X'+str(i) for i in range(X.shape[1])]
    if limit_y:
        plt.ylim(limit_y[0], limit_y[1])
    plt.scatter(X, y, c='blue')
    plt.xlabel(labels[0])
    plt.ylabel("target")

def plot_model_2d(estimator, X, y, limit_y=None):
    labels =['X'+str(i) for i in range(X.shape[1])]

    plt.scatter(X, y, c='blue')
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    xx = np.arange(x_min, x_max, 0.1)
    y_pred = estimator.predict(xx.reshape(-1, 1))
    if limit_y:
        plt.ylim(limit_y[0], limit_y[1])
    plt.plot(xx, y_pred, color='red')

def plot_data_3d(X, y):
    labels =['X'+str(i) for i in range(X.shape[1])]
    fig = plt.figure()
    ax = plt.axes(projection='3d')    
    ax.scatter(X[:, 0], X[:, 1], y, s=30, c = 'grey')  
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel("target")
    for angle in range(0, 360):
        ax.view_init(30, angle)
        plt.draw()
        plt.pause(.1)

def plot_model_3d(estimator, X, y):
    labels =['X'+str(i) for i in range(X.shape[1])]

    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))

    Z = estimator.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d') 
    ax.scatter(X[:,0], X[:,1], y, c = 'grey', s=30)
    ax.plot_surface(xx, yy, Z)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel("target")
    plt.tight_layout()    
    
    for angle in range(0, 360):
        ax.view_init(20, angle)
        plt.draw()
        plt.pause(.1)        

def plot_residuals(estimator, X, y):
    y_pred = estimator.predict(X)
    error = y_pred - y
    plt.scatter(y_pred, error, c='blue', marker='o', label='Training data') 
    xmin, xmax = y_pred.min(), y_pred.max()
    plt.hlines(y=0, xmin=xmin, xmax=xmax, lw=2, color='red')

def get_model_objective(estimator, grid, X, y):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse, greater_is_better=False), cv=10)
    grid_estimator.fit(X, y)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    #print(final_model.coef_)
    #print(final_model.intercept_)
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X, y))
    return final_model

def get_model_neighbors(estimator, grid, X, y):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse, greater_is_better=False), cv=10)
    grid_estimator.fit(X_train, y_train)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X_train, y_train))
    return final_model

def write_to_pdf(estimator, X, path):
    dot_data = io.StringIO() 
    tree.export_graphviz(estimator, out_file = dot_data, feature_names = X.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf(path)

def get_model_tree(estimator, grid, X, y, path="C://tree.pdf"):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse, greater_is_better=False), cv=10)
    labels =['X'+str(i) for i in range(X.shape[1])]
    tmp_df = pd.DataFrame(X, columns=labels)
    grid_estimator.fit(tmp_df, y)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    #write_to_pdf(final_model, tmp_df, path)
    print(grid_estimator.best_score_)
    print(grid_estimator.score(tmp_df, y))
    return final_model

def get_model_ensemble(estimator, grid, X, y):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=metrics.make_scorer(rmse, greater_is_better=False), cv=10)
    grid_estimator.fit(X, y)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X, y))
    return final_model

def grid_search_one_parameter(estimator, grid, X, y, scoring="accuracy"):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=10, return_train_score=True, scoring = scoring)
    grid_estimator.fit(X, y)

    train_mean = grid_estimator.cv_results_.get('mean_train_score')
    train_std = grid_estimator.cv_results_.get('std_train_score')
    test_mean = grid_estimator.cv_results_.get('mean_test_score')
    test_std = grid_estimator.cv_results_.get('std_test_score')

    plt.style.use('seaborn')

    param_name = list(grid.keys())[0]
    param_range = grid.get(param_name)
    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="red")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Parameter Values vs Performance:" +  str(estimator).split('(')[0] + ' model')
    plt.xlabel(param_name)
    plt.ylabel("Performance")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()
    
def grid_search_two_parameters(estimator, grid, X, y, scoring="accuracy"):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=10, scoring = scoring, return_train_score=True)
    grid_estimator.fit(X, y)

    param1_name = list(grid.keys())[0]
    param1_range = grid.get(param1_name)    
    param2_name = list(grid.keys())[1]
    param2_range = grid.get(param2_name)
        
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(param1_range, param2_range)
    
    Z = grid_estimator.cv_results_.get('mean_test_score').reshape(X.shape)
    ax.plot_surface(X, Y, Z, color='red', label="Cross-validation score")
    
    Z = grid_estimator.cv_results_.get('mean_train_score').reshape(X.shape)
    ax.plot_surface(X, Y, Z, color='black', label="Training score")
    
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel('performance')
    #plt.legend(loc="best")
    plt.tight_layout()

    for angle in range(0, 360):
        ax.view_init(20, angle)
        plt.draw()
        plt.pause(.1)
        
#non-linear pattern in 2d
X_train, y_train = generate_data2()
plot_data_2d(X_train, y_train, limit_y=[-1,1])

poly_estimator = Pipeline([('features', PolynomialFeatures()) ,
                          ('estimator', linear_model.LinearRegression())]
                        )
poly_grid = {'features__degree':list(range(1,10))}
final_estimator = get_model_objective(poly_estimator, poly_grid, X_train, y_train)
grid_search_one_parameter(poly_estimator, poly_grid, X_train, y_train, scoring = metrics.make_scorer(rmse))
plot_model_2d(final_estimator, X_train, y_train, limit_y=[-1,1])

poly_ridge_estimator = Pipeline([('features', PolynomialFeatures()) ,
                                 ('estimator', linear_model.Ridge())]
                                )
poly_ridge_grid = {'features__degree':list(range(2,20)), 
                   'estimator__alpha':[0.001, 0.002, 0.004, 0.01, 0.05, 0.1, 0.2, 0.5]
                   }
final_estimator = get_model_objective(poly_ridge_estimator, poly_ridge_grid, X_train, y_train)
plot_model_2d(final_estimator, X_train, y_train)

poly_lasso_estimator = Pipeline([('features', PolynomialFeatures()) ,
                                 ('estimator', linear_model.Lasso())]
                                )
poly_lasso_grid = {'features__degree':list(range(2,20)), 
                   'estimator__alpha':[0.001, 0.002, 0.004, 0.01, 0.05, 0.1, 0.2, 0.5]
                   }
final_estimator = get_model_objective(poly_lasso_estimator, poly_lasso_grid, X_train, y_train)
plot_model_2d(final_estimator, X_train, y_train)
