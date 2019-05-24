#practical usage of kde in 2-d
import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'E://'
sys.path.append(path)

import numpy as np
import matplotlib.pyplot as plt
import math
from common_utils import *

def plot_density_cuve_2d(xy, ax):
    d = xy.shape[0]
    n = xy.shape[1]
    bw = (n * (d + 2) / 4.)**(-1. / (d + 4)) # silverman
    #bw = n**(-1./(d+4)) # scott
    print('bw: {}'.format(bw))
    
    kde = KernelDensity(bandwidth=bw, metric='euclidean',
                        kernel='gaussian', algorithm='ball_tree')
    kde.fit(xy.T)
    
    xmin = x.min()
    xmax = x.max()
    ymin = y.min()
    ymax = y.max()
    
    X, Y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([X.ravel(), Y.ravel()])

    Z = np.reshape(np.exp(kde.score_samples(positions.T)), X.shape)

    ax.imshow(np.rot90(Z), cmap=plt.cm.viridis,
              extent=[xmin, xmax, ymin, ymax])
    
    ax.scatter(x, y, c='k', s=5, edgecolor='')
    ax.set_xlim((-2,2))
    ax.set_ylim((-2,2))
    return kde

N1 = np.random.normal(size=500)
N2 = np.random.normal(scale=0.3, size=500)
x = N1+N2
y = N1-N2
xy = np.vstack([x,y])

plot_data_2d(xy.T, xlim=(-2,2), ylim=(-2,2))

fig, axarr = plt.subplots(1, 2)
fig.subplots_adjust(left=0.11, right=0.95, wspace=0.0, bottom=0.18)
kde = plot_density_cuve_2d(xy, axarr[0])

new_data = kde.sample(100, random_state=0)
plot_data_2d(new_data, ax=axarr[1], xlim=(-2,2), ylim=(-2,2))
