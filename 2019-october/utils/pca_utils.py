from sklearn.model_selection import train_test_split
from sklearn import metrics, decomposition, manifold
from sklearn import tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plot_data_2d_with_pcs(pca_transformer, X, ax=None, xlim=None, ylim=None, title=None, s=30, new_window = True):
    plt.style.use('seaborn')

    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()   
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])

    features =  [ 'PC1', 'PC2']
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], c='blue', s=s)
    
    for i in range(pca_transformer.components_.shape[0]):
        arrow_start = pca_transformer.mean_
        arrow_end = pca_transformer.mean_ + (pca_transformer.components_[i] * pca_transformer.explained_variance_[i])
        ax.plot([ arrow_start[0], arrow_end[0] ],
                [ arrow_start[1], arrow_end[1] ],
                color='r')
        ax.text(arrow_end[0] * 1.05,
                arrow_end[1] * 1.05,
                features[i], color='r')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_title(title)
    
def plot_data_3d_with_pcs(pca_transformer, X, ax=None, xlim=None, ylim=None, zlim=None, title=None, s=30, new_window = True):
    plt.style.use('seaborn')

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

    features =  [ 'PC1', 'PC2', 'PC3' ]
    ax.scatter(X.iloc[:, 0], X.iloc[:, 1], X.iloc[:,2], c='blue', s=s)
    
    for i in range(pca_transformer.components_.shape[0]):
        arrow_start = pca_transformer.mean_
        arrow_end = pca_transformer.mean_ + (pca_transformer.components_[i] * pca_transformer.explained_variance_[i])
        ax.plot([ arrow_start[0], arrow_end[0] ],
                [ arrow_start[1], arrow_end[1] ],
                [ arrow_start[2], arrow_end[2] ],
                color='r')
        ax.text(arrow_end[0] * 1.05,
                arrow_end[1] * 1.05,
                arrow_end[2] * 1.05,
                features[i], color='r')

    ax.set_xlabel('X1')
    ax.set_ylabel('X2')
    ax.set_zlabel('X3')
    ax.set_title(title)
    

def plot_pca_variance_ratios(pca_transformer, ax=None, title=None, new_window=True):
    plt.style.use('seaborn')
    
    var = np.cumsum(np.round(pca_transformer.explained_variance_ratio_, decimals=3)*100)
    
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()
        
    ax.set_ylabel('% Variance Explained')
    ax.set_xlabel('# of Principal Components')
    ax.set_title(title)
    ax.plot(var)

def plot_pca_components_1d_biplot(pca_transformer, X, ax=None, xlim=None, ylim=None, title=None, s=30, new_window = True):
    plt.style.use('seaborn')

    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()   
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])

    features = X.columns
    X_pca = pca_transformer.transform(X)
    zero_array = np.zeros((X.shape[0],1))
    ax.scatter(X_pca[:, 0], zero_array, c='blue', s=s)

    for i in range(pca_transformer.components_.shape[1]):
        ax.plot([0, pca_transformer.components_[0, i] ],
                [0, pca_transformer.components_[1, i] ],
                color='r')
        ax.text(pca_transformer.components_[0, i] *  1.05,
                pca_transformer.components_[1, i] *  1.05,
                features[i], color='r')
    
    ax.set_xlabel('PC1')
    ax.set_ylable('')
    ax.set_title(title)

def plot_pca_components_barplot(pca_transformer, X, ax=None, new_window = True, title=None):
    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes() 
    
    n_dims = X.shape[1]
    n_pcs = len(pca_transformer.components_)
    dimensions = ['PC' + str(i) for i in range(1, n_pcs+1)]
    labels = X.keys()
    base = np.arange(n_pcs)

    for i in range(n_dims):
        ax.bar([x + i*0.25 for x in base] , pca_transformer.components_[:,i], width=0.25, label=labels[i])
    
    ax.set_ylabel("Feature Weights") 
    ax.set_xticks([x + 0.25 for x in range(n_pcs) ])
    ax.set_xticklabels(dimensions)
    ax.set_title(title)
    ax.legend()

def plot_pca_components_2d_biplot(pca_transformer, X, ax=None, xlim=None, ylim=None, title=None, s=30, new_window = True):
    plt.style.use('seaborn')

    if new_window:
        plt.figure()
    if ax is None:
        ax = plt.axes()   
    if ylim:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim:
        ax.set_xlim(xlim[0], xlim[1])

    features = X.columns
    X_pca = pca_transformer.transform(X)
    ax.scatter(X_pca[:, 0], X_pca[:, 1], c='blue', s=s)

    for i in range(pca_transformer.components_.shape[1]):
        ax.plot([0, pca_transformer.components_[0, i] ],
                [0, pca_transformer.components_[1, i] ],
                color='r')
        ax.text(pca_transformer.components_[0, i] *  1.05,
                pca_transformer.components_[1, i] *  1.05,
                features[i], color='r')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_title(title)

def plot_pca_components_3d_biplot(pca_transformer, X, ax = None, xlim=None, ylim=None, zlim=None, title=None, rotation=False, s=30, new_window=True):
    plt.style.use('seaborn')

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

    features = X.columns
    X_pca = pca_transformer.transform(X)
    
    ax.scatter(X_pca[:, 0], X_pca[:, 1], X_pca[:,2], c='blue', s = s)
    
    for i in range(pca_transformer.components_.shape[1]):
        ax.plot([0, pca_transformer.components_[0, i] ],
                [0, pca_transformer.components_[1, i] ],
                [0, pca_transformer.components_[2, i] ],
                color='r')
        ax.text(pca_transformer.components_[0, i] *  1.05,
                pca_transformer.components_[1, i] *  1.05,
                pca_transformer.components_[2, i] *  1.05,
                features[i], color='r')
    
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.set_zlabel('PC3')
    ax.set_title(title)
    
    if rotation:
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.1)

def plot_pca_result(lpca, X):    
    fig = plt.figure(figsize=(20,20), dpi=80)
    
    if X.shape[1] == 3:
        ax1 = fig.add_subplot(221, projection='3d')
        plot_data_3d_with_pcs(lpca, X, ax=ax1, new_window=False, title="GivenData with PCs")
    elif X.shape[1] == 2:
        ax1 = fig.add_subplot(221)
        plot_data_2d_with_pcs(lpca, X, ax=ax1, new_window=False, title="GivenData with PCs")

    ax2 = fig.add_subplot(222)
    plot_pca_variance_ratios(lpca, ax=ax2, new_window=False, title="Variance Ratio of PCS")
    
    ax3 = fig.add_subplot(223)
    plot_pca_components_barplot(lpca, X, ax=ax3, new_window=False, title="PC Component Barplot")
    
    if lpca.n_components_ == 3:
        ax4 = fig.add_subplot(224, projection='3d')
        plot_pca_components_3d_biplot(lpca, X, ax=ax4, new_window=False, title="PC Component BiPlot")
    elif lpca.n_components_ == 2:
        ax4 = fig.add_subplot(224)
        plot_pca_components_2d_biplot(lpca, X, ax=ax4, new_window=False, title="PC Component BiPlot")
    else:
        ax4 = fig.add_subplot(224)
        plot_pca_components_1d_biplot(lpca, X, ax=ax4, new_window=False, title="PC Component BiPlot")
   
    plt.subplots_adjust(wspace=0.1, hspace=0.5)