import numpy as np
import pandas as pd
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, gridspec
from matplotlib.colors import ListedColormap
from sklearn import decomposition, tree
from sklearn.datasets import make_classification
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import validation_curve, learning_curve, train_test_split


def plot_data_2d(X, y, labels=['X1', 'X2']):
    colors = ['red','green','purple','blue']
    plt.scatter(X[:,0], X[:,1], c=y, cmap=ListedColormap(colors), s=30)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    
def plot_data_3d(X, y, labels=['X1', 'X2','X3']):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = ['red','green','purple','blue']
    ax.scatter(X[:,0], X[:,1], X[:,2], c = y, cmap=ListedColormap(colors), s=30)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])
    
def plot_parameter_impact_on_performance(estimator, data_features, data_target, param_name, param_range, scoring="accuracy"):
    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator, 
                                             data_features, 
                                             data_target, 
                                             param_name=param_name, 
                                             param_range=param_range,
                                             cv=10, 
                                             scoring=scoring)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.style.use('seaborn')

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
    
def plot_datasize_impact_on_performance(estimator, data_features, data_target, scoring="accuracy"):
    # Create CV training and test scores for various training set sizes
    train_sizes, train_scores, test_scores = learning_curve(estimator, 
                                             data_features, 
                                             data_target, 
                                             cv=10, 
                                             scoring=scoring, 
                                             train_sizes=np.linspace(0.01, 1.0, 50))


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.style.use('seaborn')
    
    # Plot mean accuracy scores for training and test sets
    plt.plot(train_sizes, train_mean, label="Training score", color="black")
    plt.plot(train_sizes, test_mean, label="Cross-validation score", color="red")

    # Plot accurancy bands for training and test sets
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("TrainingSet Size vs Performance:"+  str(estimator).split('(')[0] + ' Model')
    plt.xlabel("TrainingSet Size")
    plt.ylabel("Performance")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

def plot_parameter_impact_on_boundaries(estimator, data_features, data_target, param_name, param_range):
    grid = gridspec.GridSpec(3, 3)
    fig = plt.figure(figsize=(10,8))

    for depth, grd in zip(range(1,10), itertools.product([0,1,2], repeat=2)):
        dt = tree.DecisionTreeClassifier(max_depth=depth)
        dt.fit(X_train, y_train)
        ax = plt.subplot(grid[grd[0], grd[1]])
        plot_decision_regions(X=X_train, y=y_train, clf=dt, ax = ax, legend = 2)
        plt.title('depth='+str(depth))
    plt.tight_layout()
    plt.show()


X, y = make_classification(n_samples = 100,
                                       n_features = 2,
                                       n_informative = 2,
                                       n_redundant = 0,
                                       n_classes = 2,
                                       weights = [.4, .6])

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.1, 
                                                    random_state=1)
plot_data_2d(X_train, y_train) 

plot_parameter_impact_on_boundaries(tree.DecisionTreeClassifier(), X_train, y_train, 'max_depth', list(range(1,10)))
plot_parameter_impact_on_performance(tree.DecisionTreeClassifier(), X_train, y_train, "max_depth", list(range(1,10)))
plot_datasize_impact_on_performance(tree.DecisionTreeClassifier(), X_train, y_train)
