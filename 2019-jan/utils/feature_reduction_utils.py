from sklearn.model_selection import train_test_split
from sklearn import metrics, decomposition, manifold
from sklearn import tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def feature_reduction_linear_pca(X, n_components):    
    lpca = decomposition.PCA(n_components)
    lpca_data = lpca.fit_transform(X)
    var = np.cumsum(np.round(lpca.explained_variance_ratio_, decimals=3)*100)
    print(var)
    plt.style.use('seaborn')
    plt.figure()
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Principal Components')
    plt.title('PCA Analysis')
    plt.plot(var)
    plt.show()
    return lpca_data

def feature_reduction_kernel_pca(X, n_components, kernel='linear', gamma=None, degree=None, intercept=1):    
    if kernel == 'rbf':
        kpca = decomposition.KernelPCA(kernel='rbf', gamma=gamma)
    elif kernel =='poly':
        kpca = decomposition.KernelPCA(kernel='poly', degree=degree, coef0=intercept)
    else:
        kpca = decomposition.KernelPCA()        
    kpca_data = kpca.fit_transform(X)
    return kpca_data

def feature_reduction_tsne(X, n_components=2):
    tsne = manifold.TSNE(n_components)
    tsne_data = tsne.fit_transform(X)
    return tsne_data

def feature_reduction_isomap(X, n_components=2, n_neighbors=5):
    isomap = manifold.Isomap(n_components=n_components, n_neighbors=n_neighbors)
    isomap_data = isomap.fit_transform(X)
    return isomap_data