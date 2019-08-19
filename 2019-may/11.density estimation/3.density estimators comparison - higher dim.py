import sys
import os
path = 'E://'
sys.path.append(path)

from sklearn import datasets, manifold, neighbors, decomposition, model_selection, mixture
import matplotlib.pyplot as plt
import numpy as np
import classification_utils as cutils
import common_utils as utils

plt.style.use('seaborn')

def plot_digits(data, title):
    fig, ax = plt.subplots(10, 10, subplot_kw=dict(xticks=[], yticks=[]))
    ax = ax.ravel()
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    for i, axi in enumerate(ax):
        im = axi.imshow(data[i].reshape(8, 8), cmap='binary')
        im.set_clim(0, 16)
    fig.suptitle(title)
    
# load the data
digits = datasets.load_digits()
print(digits.data.shape)

plot_digits(digits.data, "Original Digits")

np.corrcoef(digits.data)
# project the 64-dimensional data to a lower dimension
pca = decomposition.PCA(n_components=30, whiten=False)
pca_digits = pca.fit_transform(digits.data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca_digits.shape)
#incorrect visualization with only 2 pcs
utils.plot_data_2d(pca_digits[:, 0:2])

#tsne on pca  data
tsne = manifold.TSNE()
tsne_digits = tsne.fit_transform(pca_digits)
print(tsne.kl_divergence_)
print(tsne_digits.shape)
cutils.plot_data_2d_classification(tsne_digits, digits.target)

#using GMM
gmm_estimator = mixture.GaussianMixture()
gmm_params = {'n_components': np.arange(50, 200, 10) }
gmm_grid_estimator = model_selection.GridSearchCV(gmm_estimator, gmm_params)
gmm_grid_estimator.fit(pca_digits)
gmm_best_estimator = gmm_grid_estimator.best_estimator_

pca_new_data = gmm_best_estimator.sample(1000)
print(pca_new_data[0].shape)
new_tsne_digits = tsne.fit_transform(pca_new_data[0])
utils.plot_data_2d(new_tsne_digits)

digits_new = pca.inverse_transform(pca_new_data[0])
print(digits_new.shape)
plot_digits(digits_new, 'Generated Digits')
