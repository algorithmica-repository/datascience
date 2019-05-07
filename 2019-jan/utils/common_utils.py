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

def get_continuous_features(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_features(df):
    return df.select_dtypes(exclude=['number']).columns

def cast_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def get_categorical_imputers(df, features):    
    feature_defs = []
    for col_name in features:
        feature_defs.append((col_name, CategoricalImputer()))
    multi_imputer = DataFrameMapper(feature_defs, input_df=True, df_out=True)
    multi_imputer.fit(df[features])
    return multi_imputer

def get_continuous_imputers(df, features):
    cont_imputer = preprocessing.Imputer()
    cont_imputer.fit(df[features])
    print(cont_imputer.statistics_)
    return cont_imputer

def get_features_to_drop_on_missingdata(df, threshold) :
    tmp = df.isnull().sum()
    return list(tmp[tmp/float(df.shape[0]) > threshold].index)

def drop_features(df, features):
    return df.drop(features, axis=1, inplace=False)

def ohe(df, features):
    return pd.get_dummies(df, columns = features)

def get_scaler(df):
    scaler = preprocessing.StandardScaler()
    scaler.fit(df)
    return scaler

def get_zero_variance_filter(X_train):
    tmp = feature_selection.VarianceThreshold()
    return tmp.fit()

def corr_heatmap(X):
    corr = np.corrcoef(X, rowvar=False)
    sns.heatmap(corr, annot=False)
  
def get_important_features(estimator, X, threshold='median'):
    if isinstance(estimator, sklearn.linear_model.Lasso) :
        selected_features = X.columns[estimator.coef_!=0]
    else:
        tmp_model = feature_selection.SelectFromModel(estimator, prefit=True, threshold=threshold)
        selected_features = X.columns[tmp_model.get_support()]
    return selected_features 

def select_features(estimator, X, threshold='median'):
    tmp_model = feature_selection.SelectFromModel(estimator, prefit=True, threshold=threshold)
    selected_features = X.columns[tmp_model.get_support()]
    return pd.DataFrame(tmp_model.transform(X), columns = selected_features)  


def plot_feature_importances(estimator, X, cutoff=40):
    if isinstance(estimator, sklearn.linear_model.Lasso) :
        importances = estimator.coef_
    else:
        importances = estimator.feature_importances_
    indices = np.argsort(importances)[::-1][:cutoff]
    plt.figure()
    g = sns.barplot(y=X.columns[indices][:cutoff],x = importances[indices][:cutoff] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("Feature importances based on: " + str(estimator).split('(')[0] + ' model' ) 
 
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


def grid_search_plot_two_parameter_curves(estimator, grid, X, y, scoring="accuracy", cv=10, rotation=False):
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

    if rotation:
        for angle in range(0, 360):
            ax.view_init(20, angle)
            plt.draw()
            plt.pause(.1)

def plot_data_1d(X, ax = None, xlim=None, title=None, new_window=True, s=30):
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
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
        
    zero_array = np.zeros((X.shape[0],1))
    ax.scatter(X[:, 0], zero_array[:, 0], color='red',
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=s) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel('')
    ax.set_title(title)
    ax.legend()
    
def plot_data_2d(X, ax = None, xlim=None, ylim=None, title=None, new_window=True, s=30):
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
        
    ax.scatter(X[:, 0], X[:, 1], color='red',
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=s) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()

def plot_data_3d(X, ax = None, xlim=None, ylim=None, zlim=None, title=None, new_window=True, rotation=False, s=30):
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
      
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], color='red', 
                       cmap=plt.cm.RdYlBu, edgecolor='black', s=s) 
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

def write_ensemble_model_to_dir(estimator, X, path, prefix="dt"):
    n_tree = 0
    for est in final_model.estimators_:
        write_dt_model_to_pdf(est, X, path, prefix + str(n_tree))
        n_tree = n_tree + 1

def write_dt_model_to_dir(estimator, X, path, prefix="dt"):
    dot_data = io.StringIO() 
    tree.export_graphviz(estimator, out_file = dot_data, feature_names = X.columns)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf(os.path.join(path, prefix+".pdf"))

def grid_search_best_model(estimator, grid, X, y, cv=10, scoring='accuracy'):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, scoring=scoring, cv=cv)
    grid_estimator.fit(X, y)
    best_est = grid_estimator.best_estimator_
    
    print("Best parameters:" + str(grid_estimator.best_params_) )
    print("Validation score:" + str(grid_estimator.best_score_) )
    print("Train score:" + str(grid_estimator.score(X, y)) )

    return best_est
       