import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors, model_selection, mixture
from astroML import density_estimation

mu_sigma_f = [(0, 1, 0.3), (5, 1, 0.7)]
X_plot = np.linspace(-5, 10, 1000)[:, np.newaxis]
true_dens = sum([f * stats.norm(mu, sigma).pdf(X_plot[:, 0]) 
                            for (mu, sigma, f) in mu_sigma_f])

fig, ax = plt.subplots(2, 2, sharex=True, sharey=True)
fig.subplots_adjust(left=0.05, right=0.95, hspace=0.05, wspace=0.05)
ax = ax.ravel()

for i, N in enumerate([50, 100, 200, 500]):
    ax[i].fill(X_plot[:, 0], true_dens, fc='black', alpha=0.2, label='input distribution')
    X = np.concatenate([stats.norm(mu, sigma).rvs(int(f*N)) 
                            for (mu, sigma, f) in mu_sigma_f])
    X = X[:, np.newaxis]
    
    kde = neighbors.KernelDensity(kernel='gaussian', bandwidth=0.5).fit(X)
    log_dens = kde.score_samples(X_plot)
    ax[i].plot(X_plot[:, 0], np.exp(log_dens), '-', label="gaussian kernel(bw=0.5)")
    
    nbrs = density_estimation.KNeighborsDensity('bayesian', n_neighbors=40).fit(X)
    dens_nbrs = nbrs.eval(X_plot) / N
    ax[i].plot(X_plot[:, 0], dens_nbrs, '-', label="knn(k=40)")
        
    gmm = mixture.GaussianMixture(3, 'full').fit(X)
    log_dens = gmm.score_samples(X_plot)
    ax[i].plot(X_plot[:, 0], np.exp(log_dens), '-', label="gaussian mixture(comp=3)")
   
    ax[i].legend(loc='upper left')
    ax[i].text(6, 0.38, "N={0} points".format(N))
    ax[i].plot(X[:, 0], -0.001 * np.random.random(X.shape[0])-0.005, '|k')
    ax[i].set_xlim(-4, 9)
    ax[i].set_ylim(-0.02, 0.4)
        
fig.suptitle('Comparison of Nonparametric DE Algorithms')