import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

def plot_normalized_data_2d(X, X_norm):
    plt.style.use('seaborn')

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.set_title('Before Normalization')
    ax1.scatter(X[:,0], X[:,1], c='blue')

    ax2.set_title('After Normalization')
    ax2.scatter(X_norm[:,0], X_norm[:,1], c='blue')


def plot_normalized_data_3d(X, X_norm):
    plt.style.use('seaborn')

    fig = plt.figure()
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    ax1.set_title('Before Normalization')
    ax1.scatter(X[:,0], X[:,1], X[:,2], c='blue')

    ax2.set_title('After Normalization')
    ax2.scatter(X_norm[:,0], X_norm[:,1], X_norm[:,2], c='blue')


np.random.seed(1)

x1 = np.random.randint(-100, 100, 1000).astype(float)
x2 = np.random.randint(-80, 80, 1000).astype(float)
X = np.column_stack((x1, x2))

normalizer = preprocessing.Normalizer()
X_norm = normalizer.fit_transform(X)
plot_normalized_data_2d(X, X_norm)

x1 = np.random.randint(-100, 100, 1000).astype(float)
x2 = np.random.randint(-80, 80, 1000).astype(float)
x3 = np.random.randint(-150, 150, 1000).astype(float)
X = np.column_stack((x1, x2, x3))

normalizer = preprocessing.Normalizer()
X_norm = normalizer.fit_transform(X)
plot_normalized_data_3d(X, X_norm)
