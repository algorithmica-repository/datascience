import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors, mixture, model_selection
import sklearn
from scipy import stats
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('classic')

def plot_2d_density(estimator, X_sample, orig_dens, model_dens, xmin, xmax):
    fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
    fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)
    ax = ax.ravel()

    ax[0].imshow(orig_dens, origin='lower', extent=[xmin, xmax, xmin, xmax],
                  cmap=plt.cm.binary)
    ax[0].set_title("Source Distribution Model")
    
    ax[1].scatter(X_sample[:, 0], X_sample[:, 1], color='b', edgecolor='black', s=20, alpha=0.5)
    ax[1].set_title("Random sample from Source Distribution")
    ax[1].set_ylabel('$y$')

    ax[2].imshow(model_dens, origin='lower', extent=[xmin, xmax, xmin, xmax],
                  cmap=plt.cm.binary)
    ax[2].set_title("Density Model")

    if isinstance(estimator, sklearn.mixture.GaussianMixture):
        X_new = estimator.sample(1000)[0]
    else:
        X_new = estimator.sample(1000)
    ax[3].scatter(X_new[:, 0], X_new[:, 1], color='r', edgecolor='black', alpha=0.5, s=20)
    ax[3].set_title("Random Generated Data from Density-Model")

    for ax in fig.axes:
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(xmin, xmax)
        ax.set_xlabel('$x$')
    fig.suptitle("Understanding Density Estimation Algorithm", size=20)

#create input grid of data
xmin = -3
xmax = 9
n = 100
xx, yy = np.meshgrid(np.linspace(xmin, xmax, n),
                        np.linspace(xmin, xmax, n))
Xgrid = np.column_stack( (xx.ravel(), yy.ravel()) )
   
#create a combined multivariate gaussian distribution
mu_vec = [ [0,0], [5,3] ]
cov_mat = [ [[1,0.5],[0.5,1]], [[1,0.8],[0.8,1]]]
f = [0.5, 0.5]
orig_dens =  sum([f[i] * stats.multivariate_normal(mean=mu_vec[i], cov=cov_mat[i]).pdf(Xgrid)
                            for i in range(len(mu_vec)) ])
orig_dens = orig_dens.reshape(xx.shape) 

#randomly sample from combined multivariate gaussian distribution
N = 300
X_sample = np.concatenate([ stats.multivariate_normal(mean=mu_vec[i], cov=cov_mat[i]).rvs(int(f[i]*N)) 
                            for i in range(len(mu_vec)) ])

#using GMM
gmm_estimator = mixture.GaussianMixture()
gmm_params = {'n_components': np.arange(1, 10, 1) }
gmm_grid_estimator = model_selection.GridSearchCV(gmm_estimator, gmm_params)
gmm_grid_estimator.fit(X_sample)
gmm_best_estimator = gmm_grid_estimator.best_estimator_
model_dens = np.exp(gmm_best_estimator.score_samples(Xgrid))
model_dens = model_dens.reshape(xx.shape)
plot_2d_density(gmm_best_estimator, X_sample, orig_dens, model_dens, xmin, xmax)
   
#using KDE
kde_estimator = neighbors.KernelDensity(kernel='gaussian')
kde_params = {'bandwidth': np.logspace(-1, 1, 20)}
kde_grid_estimator = model_selection.GridSearchCV(kde_estimator, kde_params)
kde_grid_estimator.fit(X_sample)
kde_best_estimator = kde_grid_estimator.best_estimator_
model_dens = np.exp(kde_best_estimator.score_samples(Xgrid))
model_dens = model_dens.reshape(xx.shape)
plot_2d_density(kde_best_estimator, X_sample, orig_dens, model_dens, xmin, xmax)
