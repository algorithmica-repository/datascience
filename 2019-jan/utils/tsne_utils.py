import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'G://'
sys.path.append(path)

import numpy as np
from sklearn import cluster,metrics, mixture
from sklearn.model_selection import train_test_split
from sklearn import metrics, decomposition, manifold
from sklearn import tree, covariance, linear_model, ensemble, neighbors, svm, model_selection, feature_selection
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import (_joint_probabilities, 
                                    _kl_divergence)
from scipy.spatial.distance import squareform, pdist
from time import time
from scipy import linalg
import seaborn as sns
from classification_utils import *

def plot_distance_matrix(D, ax, cmap, title):
    ax.imshow(D, interpolation='none', cmap=cmap)
    ax.set_title(title)
    
def plot_tsne_result(X, y, n_components):
    positions = []
    errors = []
    def _gradient_descent(objective, p0, it, n_iter,
                      n_iter_check=1, n_iter_without_progress=300,
                      momentum=0.8, learning_rate=200.0, min_gain=0.01,
                      min_grad_norm=1e-7, verbose=0, args=None, kwargs=None):
        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        p = p0.copy().ravel()
        update = np.zeros_like(p)
        gains = np.ones_like(p)
        error = np.finfo(np.float).max
        best_error = np.finfo(np.float).max
        best_iter = i = it

        tic = time()
        for i in range(it, n_iter):
            positions.append(p.copy())
        
            error, grad = objective(p, *args, **kwargs)
            errors.append(error)
            grad_norm = linalg.norm(grad)

            inc = update * grad < 0.0
            dec = np.invert(inc)
            gains[inc] += 0.2
            gains[dec] *= 0.8
            np.clip(gains, min_gain, np.inf, out=gains)
            grad *= gains
            update = momentum * update - learning_rate * grad
            p += update

            if (i + 1) % n_iter_check == 0:
                toc = time()
                duration = toc - tic
                tic = toc

            if verbose >= 2:
                print("[t-SNE] Iteration %d: error = %.7f,"
                      " gradient norm = %.7f"
                      " (%s iterations in %0.3fs)"
                      % (i + 1, error, grad_norm, n_iter_check, duration))

            if error < best_error:
                best_error = error
                best_iter = i
            elif i - best_iter > n_iter_without_progress:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: did not make any progress "
                          "during the last %d episodes. Finished."
                          % (i + 1, n_iter_without_progress))
                break
            if grad_norm <= min_grad_norm:
                if verbose >= 2:
                    print("[t-SNE] Iteration %d: gradient norm %f. Finished."
                          % (i + 1, grad_norm))
                break

        return p, error, i

    D = pairwise_distances(X, squared=True)
    P_binary = _joint_probabilities(D, 30., False)
    P_binary_s = squareform(P_binary)
    
    positions.clear()
    errors.clear()
    manifold.t_sne._gradient_descent = _gradient_descent
    manifold.TSNE(n_components=n_components, random_state=100).fit_transform(X)
    if n_components == 3:
        X_iter = np.dstack(position.reshape(-1, 3) for position in positions)
    elif n_components == 2:
        X_iter = np.dstack(position.reshape(-1, 2) for position in positions)

    cmap = sns.light_palette("blue", as_cmap=True)

    fig = plt.figure(figsize=(12, 12))
    if X.shape[1] == 3:
        ax1 = fig.add_subplot(3, 4, 1, projection='3d')
        plot_data_3d_classification(X, y, ax = ax1, new_window = False, title = "Original Data")
    elif X.shape[1] == 2:
        ax1 = fig.add_subplot(3, 4, 1)
        plot_data_2d_classification(X, y, ax = ax1, new_window = False,  title = "Original Data")  
    
    ax2 = fig.add_subplot(3, 4, 2)        
    plot_distance_matrix(P_binary_s, ax2, cmap, 'Pairwise Similarities')
 
    iter_size = int(len(positions)/5)
    k = 2
    for i in range(5):
        iter_index =  i * iter_size
        tmp = X_iter[..., iter_index]
        err = round(errors[iter_index],2)
        title = "Iter: " + str(iter_index) + " Loss:" + str(err)
        
        k = k + 1
        if X_iter.shape[1] == 3:
            ax3 = fig.add_subplot(3, 4, k, projection='3d')
            plot_data_3d_classification(tmp, y, ax = ax3, new_window = False, title = title)
        elif X_iter.shape[1] == 2:
            ax3 = fig.add_subplot(3, 4, k)
            plot_data_2d_classification(tmp, y, ax = ax3, new_window = False, title = title)  

        k = k + 1       
        ax4 = fig.add_subplot(3, 4, k)        
        n = 1. / (pdist(tmp, "sqeuclidean") + 1)
        Q = n / (2.0 * np.sum(n))
        Q = squareform(Q)
        plot_distance_matrix(Q, ax4, cmap, title = title)
    plt.subplots_adjust(wspace=0.1, hspace=0.5)