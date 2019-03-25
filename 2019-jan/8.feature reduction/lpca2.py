import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm, gridspec
from matplotlib.colors import ListedColormap
from sklearn import decomposition, tree
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def plot_data_2d(X, y, labels=['X1', 'X2']):
    colors = ['red','green','purple','blue']
    plt.scatter(X[:,0], X[:,1], c=y, cmap=ListedColormap(colors), s=30)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    
def plot_data_3d(X, y, labels=['X1', 'X2','X3']):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = ['red','green','purple','blue']
    ax.scatter(X[:,0], X[:,1], X[:,2], c = y, cmap=ListedColormap(colors), s=30)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

X, y = make_classification(n_samples = 100,
                                       n_features = 4,
                                       n_informative = 2,
                                       n_redundant = 2,
                                       n_classes = 2,
                                       weights = [.4, .6])

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.1, 
                                                    random_state=1)
corr = np.corrcoef(X_train, rowvar=False)
sns.heatmap(corr)
#plot_data_3d(X_train, y_train) 

lpca = decomposition.PCA(2)
pca_data = lpca.fit_transform(X_train)
print(lpca.explained_variance_ratio_)
plot_data_2d(pca_data, y_train, ['PC1', 'PC2'])
