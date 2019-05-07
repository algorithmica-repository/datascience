from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm


def plot_kl_divergence():
    x = np.linspace(-10.0, 10.0, 1000)
    plt.figure(figsize=(12,8))

    # gradually shift the distribution
    for i in np.arange(3):
        for j in np.arange(3):
            # index to shift
            index = i*3 + j
            # probabilistic distribution function
            p = stats.norm.pdf(x, loc=0, scale=1)
            q = stats.norm.pdf(x, loc=index*0.5, scale=1)
            kl = stats.entropy(p, q)
            print("KL divergence: " + str(kl) )
        
            plt.subplot(3,3,i*3+j+1)
            plt.fill_between(x, p, facecolor="g", alpha=0.5)
            plt.fill_between(x, q, facecolor="r", alpha=0.5)
            plt.xlim(-5, 7)
            plt.ylim(0,0.45)
            plt.legend()
            plt.title("KLD:" + str(np.round(kl,2)) ) 

    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.show()
