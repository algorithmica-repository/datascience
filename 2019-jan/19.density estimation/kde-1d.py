import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV
from sklearn.cross_validation import LeaveOneOut

def make_data(N, f=0.3, rseed=1):
    rand = np.random.RandomState(rseed)
    x = rand.randn(N)
    x[int(f * N):] += 5
    return x

def plot_density_curve(estimator, x):
    x_d = np.linspace(-4, 8, 1000)
    logprob = estimator.score_samples(x_d[:, None])
    plt.fill_between(x_d, np.exp(logprob), alpha=0.5)
    plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
    plt.ylim(-0.02, 0.25)

def plot_data_1d(x):
    plt.figure()
    plt.plot(x, np.full_like(x, -0.01), '|k', markeredgewidth=1)
    plt.ylim(-0.02, 0.25)


x = make_data(1000)
plot_data_1d(x)

# instantiate and fit the KDE model
kde_estimator = KernelDensity(bandwidth=0.5, kernel='gaussian')
kde_estimator.fit(x[:, None])
plot_density_curve(kde_estimator, x)

#practical usage of kde in 1-d
kde_estimator = KernelDensity(kernel='gaussian')
kde_grid = { 'bandwidth': list(10 ** np.linspace(-1, 1, 100)) }
kde_grid_estimator = GridSearchCV( kde_estimator, kde_grid, cv=10)
kde_grid_estimator.fit(x[:, None])
print(kde_grid_estimator.best_params_)
best_estimator = kde_grid_estimator.best_estimator_
plot_density_curve(best_estimator, x)

new_data = best_estimator.sample(500, random_state=0)
plot_data_1d(new_data)