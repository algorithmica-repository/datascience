import os
import pandas as pd
import seaborn as sns
import numpy as np
import math
from itertools import product, cycle
from sklearn import covariance, preprocessing, tree, svm, neighbors, metrics, linear_model, manifold, linear_model
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn import model_selection, ensemble, preprocessing, decomposition, feature_selection
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles, make_moons, make_classification
import matplotlib.cm as cm
from sklearn import cluster,metrics
from common_utils import *
from classification_utils import *

def generate_linear_synthetic_data_classification(n_samples, n_features, n_classes, weights, n_redundant=0):
    return make_classification(n_samples = n_samples,
                                       n_features = n_features,
                                       n_informative = n_features - n_redundant,
                                       n_clusters_per_class=1,
                                       n_redundant = n_redundant,
                                       n_classes = n_classes,
                                       weights = weights, random_state=100)

def generate_nonlinear_synthetic_data_classification1(n_samples, n_features, n_classes):
    N = n_samples//n_classes  # number of points per class
    D = n_features   # dimensionality
    K = n_classes   # number of classes

    X = np.zeros((N*K,D)) # data matrix (each row = single example)
    y = np.zeros(N*K, dtype=int) # class labels

    for j in range(K):
        ix = range(N*j,N*(j+1))
        r = np.linspace(0.0,1,N) # radius
        t = np.linspace(j*4,(j+1)*4,N) + np.random.randn(N)*0.4 # theta
        X[ix] = np.c_[r*np.sin(t), r*np.cos(t)]
        y[ix] = j
    return X, y

def generate_nonlinear_synthetic_data_classification2(n_samples, noise=0.1):
    return make_circles(n_samples=n_samples, random_state=123, noise=noise, factor=0.2)

def generate_nonlinear_synthetic_data_classification3(n_samples, noise=0.1):
    return make_moons(n_samples=n_samples, noise = noise, random_state=100)


def plot_data_1d_classification(X, y, ax = None, xlim=None, ylim=[-15,15], title=None, new_window=True, s=30):
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
        
    n_classes = set(y)
    colors = cm.rainbow(np.linspace(0, 1, len(n_classes)) )
    class_labels = [str(i) for i in n_classes]
    zero_array = np.zeros((X.shape[0],1))
    for i, color in zip(n_classes, colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], zero_array[idx, 0]+i*0.2, c=color, label = class_labels[i],
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=s) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel('')
    ax.set_title(title)
    ax.legend()
    return ax

def plot_data_2d_classification(X, y, ax = None, xlim=None, ylim=None, title="Data", labels=None, alpha=1, s=30, legend=True, marker='o'):
    if isinstance(X, np.ndarray) and labels is None:
        labels =['X'+str(i) for i in range(X.shape[1])]      
    if isinstance(X, pd.core.frame.DataFrame):
        labels = X.columns
        X = X.values
    if ax is None:
        plt.figure(figsize=(20, 20))
        ax = plt.axes()   
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])
        
    n_classes = set(y)
    colors = cm.rainbow(np.linspace(0, 1, len(n_classes)) )
    class_labels = [str(i) for i in n_classes]
    for i, color in zip(n_classes, colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=color, label = class_labels[i], marker=marker,
                    cmap=plt.cm.RdYlBu, edgecolor='black', s=s, alpha=alpha) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    if legend:
        ax.legend()
    return ax
	
def plot_data_3d_classification(X, y=None, ax = None, xlim=None, ylim=None, zlim=None, title=None, new_window=True, rotation=False, s=30):
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
      
    n_classes = set(y)
    colors = cm.rainbow(np.linspace(0, 1, len(n_classes)) )
    class_labels = [str(i) for i in n_classes]
    for i, color in zip(n_classes, colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], X[idx, 2], c=color, label = class_labels[i],
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
    return ax


def plot_model_2d_classification(estimator, X, y, ax = None, xlim=None, ylim=None, title=None, new_window=True, levels=None, s=30):
    plt.style.use('seaborn')
    if isinstance(X, np.ndarray) :
        labels =['X'+str(i) for i in range(X.shape[1])]
    else:
        labels = X.columns
        X = X.values()
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
    
    n_classes = set(y)
    colors = cm.rainbow(np.linspace(0, 1, len(n_classes)) )
    class_labels = [str(i) for i in n_classes]
    for i, color in zip(n_classes, colors):
        idx = np.where(y == i)
        ax.scatter(X[idx, 0], X[idx, 1], c=color, label = class_labels[i],
                    cmap=plt.cm.coolwarm, edgecolor='black', s=s) 
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    return ax


def grid_search_plot_models_classification(estimator, grid, X, y, xlim=None, ylim=None, outlier_estimator=False, levels=None):
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
        if outlier_estimator:
            estimator.fit(X)
        else:
            estimator.fit(X, y)        
        plot_model_2d_classification(estimator, X, y, ax, xlim, ylim, str(param), False, levels)
    plt.tight_layout()

def performance_metrics_hard_binary_classification(estimator, X, y):
    print('[Confusion Matrix]')
    y_pred = estimator.predict(X)
    print(metrics.confusion_matrix(y, y_pred))
    print('\n[Precision]')
    p = metrics.precision_score(y, y_pred)
    print(p)
    print('\n[Recall]')
    r = metrics.recall_score(y, y_pred)
    print(r)
    print('\n[F1-score]')
    f = metrics.f1_score(y, y_pred)
    print(f)

def performance_metrics_hard_multiclass_classification(estimator, X, y):
    onevsrest = False
    name = str(estimator)
    if "Logistic" in name or "SV" in name:
        onevsrest = True
    print('[Confusion Matrix]')
    y_pred = estimator.predict(X)
    print(metrics.confusion_matrix(y, y_pred))
    print('\n[Precision]')
    if onevsrest:
        p = metrics.precision_score(y, y_pred, average=None)
        print('Individual: %.2f, %.2f, %.2f' % (p[0], p[1], p[2]))
    p = metrics.precision_score(y, y_pred, average='micro')
    print('Micro: %.2f' % p)
    p = metrics.precision_score(y, y_pred, average='macro')
    print('Macro: %.2f' % p)

    print('\n[Recall]')
    if onevsrest:
        r = metrics.recall_score(y, y_pred,average=None)
        print('Individual: %.2f, %.2f, %.2f' % (r[0], r[1], r[2]))
    r = metrics.recall_score(y, y_pred, average='micro')
    print('Micro: %.2f' % r)
    r = metrics.recall_score(y, y_pred, average='macro')
    print('Macro: %.2f' % r)

    print('\n[F1-score]')
    if onevsrest:
        f = metrics.f1_score(y, y_pred, average=None)
        print('Individual: %.2f, %.2f, %.2f' % (f[0], f[1], f[2]))
    f = metrics.f1_score(y, y_pred, average='micro')
    print('Micro: %.2f' % f)
    f = metrics.f1_score(y, y_pred, average='macro')
    print('Macro: %.2f' % f)
    
def performance_metrics_soft_binary_classification(estimator, X, y):
    y_prob = estimator.predict_proba(X)[:,1]
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_binary_roc_curve(y, y_prob, ax = ax1)
    plot_binary_pr_curve(y, y_prob, ax = ax2)

def performance_metrics_soft_multiclass_classification(estimator, X, y):
    y_prob = estimator.predict_proba(X)
    fig, (ax1, ax2) = plt.subplots(1, 2)
    plot_multiclass_roc_curve(y, y_prob, ax = ax1)
    plot_multiclass_pr_curve(y, y_prob, ax = ax2)

def plot_confusion_matrix(y_true, y_pred):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    print(conf_mat)
    plt.figure()
    ax = plt.axes()
    ax.matshow(conf_mat, cmap=plt.cm.coolwarm, alpha=0.1)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            ax.text(x=j, y=i, s=conf_mat[i, j], va='center', ha='center')
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.tight_layout()
    return conf_mat
    
def plot_binary_roc_curve(y_truth, y_prob, label="", ax=None, new_window=False):
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()
    fpr, tpr, _ = metrics.roc_curve(y_truth, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    ax.step(fpr, tpr, color='darkorange', alpha=0.2, label= label + str(roc_auc), lw= 2,
         where='post')
    ax.fill_between(fpr, tpr, alpha=0.2, color='b')

    ax.plot([0, 1], [0, 1], linestyle='--')
    #ax.plot(fpr, tpr, color='darkorange', lw=2, label= label + str(roc_auc))
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.set_title('ROC curve')
    plt.tight_layout()

def plot_multiclass_roc_curve(y_truth, y_prob, label="", ax=None, new_window=False):
    classes = list(set(y_truth))
    y_truth = preprocessing.label_binarize(y_truth, classes=classes)
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_truth.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_truth[:, i], y_prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_truth.ravel(), y_prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    ax.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i])) 
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.set_title('ROC curve')
    plt.tight_layout()

def plot_binary_pr_curve(y_truth, y_prob, label="", ax=None, new_window=False):
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()
    pr, rec, _ = metrics.precision_recall_curve(y_truth, y_prob)
    pr_auc = metrics.auc(rec, pr)
    ax.step(rec, pr, color='darkorange', alpha=0.2, label= label + str(pr_auc), lw= 2,
         where='post')
    ax.fill_between(rec, pr, alpha=0.2, color='b')

    #ax.plot(rec, pr, color='darkorange', lw=2, label= label + str(pr_auc))
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc="lower right")
    ax.set_title('PR curve')
    plt.tight_layout()

def plot_multiclass_pr_curve(y_truth, y_prob, label="", ax=None, new_window=False):
    classes = list(set(y_truth))
    y_truth = preprocessing.label_binarize(y_truth, classes=classes)
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = y_truth.shape[1]
    for i in range(n_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(y_truth[:, i], y_prob[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = metrics.roc_curve(y_truth.ravel(), y_prob.ravel())
    roc_auc["micro"] = metrics.auc(fpr["micro"], tpr["micro"])
    
    ax.plot(fpr["micro"], tpr["micro"],
                 label='micro-average ROC curve (area = {0:0.2f})'
                 ''.format(roc_auc["micro"]),
                 color='deeppink', linestyle=':', linewidth=4)

    colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
    for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                     label='ROC curve of class {0} (area = {1:0.2f})'
                     ''.format(i, roc_auc[i])) 
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc="lower right")
    ax.title('ROC curve')
    plt.tight_layout()
