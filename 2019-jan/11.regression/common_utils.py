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

def grid_search_plot_one_parameter_curves(estimator, grid, X, y, scoring="accuracy",cv=10):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=cv, return_train_score=True, scoring = scoring)
    grid_estimator.fit(X, y)

    train_mean = grid_estimator.cv_results_.get('mean_train_score')
    train_std = grid_estimator.cv_results_.get('std_train_score')
    test_mean = grid_estimator.cv_results_.get('mean_test_score')
    test_std = grid_estimator.cv_results_.get('std_test_score')

    plt.figure()
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


def grid_search_plot_two_parameter_curves(estimator, grid, X, y, scoring="accuracy", cv=10):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=cv, scoring = scoring, return_train_score=True)
    grid_estimator.fit(X, y)

    param1_name = list(grid.keys())[0]
    param1_range = grid.get(param1_name)    
    param2_name = list(grid.keys())[1]
    param2_range = grid.get(param2_name)
    
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    X, Y = np.meshgrid(param1_range, param2_range)
    
    Z = grid_estimator.cv_results_.get('mean_test_score').reshape(X.shape)
    ls = LightSource(azdeg=0, altdeg=65)
    rgb = ls.shade(Z, plt.cm.RdYlBu)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=None,
                       antialiased=False, facecolors=rgb,
                       label="Cross-validation score")
    
    Z = grid_estimator.cv_results_.get('mean_train_score').reshape(X.shape)
    rgb = ls.shade(Z, plt.cm.RdYlBu)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, alpha=None,
                       antialiased=False, facecolors=rgb,
                       label="Training score")
    
    ax.set_xlabel(param1_name)
    ax.set_ylabel(param2_name)
    ax.set_zlabel('performance')
    #plt.legend(loc="best")
    plt.tight_layout()

    for angle in range(0, 360):
        ax.view_init(20, angle)
        plt.draw()
        plt.pause(.1)

def get_best_model(estimator, grid, X, y, scoring='accuracy', cv=10, path='C://'):
    if isinstance(X, np.ndarray) :
        labels =['X'+str(i) for i in range(X.shape[1])]
        X = pd.DataFrame(X, columns=labels)
        
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=scoring, cv=cv)
    grid_estimator.fit(X, y)
    best_est = grid_estimator.best_estimator_

    if isinstance(estimator, sklearn.pipeline.Pipeline) :
        final_model = best_est.named_steps['estimator']
    else :
        final_model = best_est
       
    name = str(final_model)
    print(name)
    if name.startswith('DecisionTree'):
        write_to_pdf(final_model, X, path)
    elif (name.startswith('SV') or name.startswith('Logistic') or name.startswith('Linear') or
          name.startswith('Ridge') or name.startswith('Lasso') or name.startswith('Elastic')
         ) :
        print("Coefficients:" + str(final_model.coef_) )
        print("Intercept:" + str(final_model.intercept_) )
    else :
        print("No model to display")
    print("Best parameters:" + str(grid_estimator.best_params_) )
    print("Validation score:" + str(grid_estimator.best_score_) )
    print("Train score:" + str(grid_estimator.score(X, y)) )
    return best_est

def write_to_pdf(estimator, X, path):
    dot_data = io.StringIO() 
    tree.export_graphviz(estimator, out_file = dot_data, feature_names = X.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf(path)


       