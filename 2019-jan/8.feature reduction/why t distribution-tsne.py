from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm

#why t-distribution in lower dimensional space instead of guassian?
def plot_data_in_higer_dim():
    npoints = 1000
    plt.figure(figsize=(15, 4))
    for i, D in enumerate((2, 5, 10)):
        # Normally distributed points.
        u = np.random.randn(npoints, D)
        # Now on the sphere.
        u /= norm(u, axis=1)[:, None]
        # Uniform radius.
        r = np.random.rand(npoints, 1)
        # Uniformly within the ball.
        points = u * r**(1./D)
        # Plot.
        ax = plt.subplot(1, 3, i+1)
        ax.set_xlabel('Ball radius')
        if i == 0:
            ax.set_ylabel('Distance from origin')
        ax.hist(norm(points, axis=1),
            bins=np.linspace(0., 1., 50))
        ax.set_title('D=%d' % D, loc='left')
        
def plot_guassian_and_t_dist():
    z = np.linspace(0., 5., 1000)
    gauss = np.exp(-z**2)
    cauchy = 1/(1+z**2)
    plt.plot(z, gauss, label='Gaussian distribution')
    plt.plot(z, cauchy, label='Cauchy distribution')
    plt.legend()

plot_data_in_higer_dim()
plot_guassian_and_t_dist()