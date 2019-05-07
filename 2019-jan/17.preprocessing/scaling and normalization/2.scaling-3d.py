import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D

def plot_distributions_3d(X, titles):
    plt.style.use('seaborn')
    
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    ax1.set_title(titles[0])
    ax1.scatter(X[:,0], X[:,1], X[:,2], c='blue')

    ax2.set_title(titles[1])
    sns.kdeplot(X[:,0], ax=ax2)
    sns.kdeplot(X[:,1], ax=ax2)
    sns.kdeplot(X[:,2], ax=ax2)

np.random.seed(1)

##generate normal distributed data on different scales
x1 = np.random.normal(0, 2, 10000)
x2 = np.random.normal(5, 3, 10000)
x3 = np.random.normal(-5, 5, 10000)
X = np.column_stack((x1, x2, x3))

##generate heterogeneous distributed data on different scales
x1 = np.random.chisquare(8, 1000) # positive skew
x2 = np.random.beta(8, 2, 1000) * 40 #negative skew
x3 = np.random.normal(50, 3, 1000) # no skew
X = np.column_stack((x1, x2, x3))

##generate data with outliers
x1 = np.random.normal(0, 2, 100)
x2 = np.random.normal(5, 3, 100)
x3 = np.random.normal(-5, 5, 100)
X = np.column_stack((x1, x2, x3))
#add outlier
X[:10:10] = (100, 0, 0)

titles = ['Original Data - Scatterplot View', 'Original Data - Distribution View']
plot_distributions_3d(X, titles)

s_scaler = preprocessing.StandardScaler()
X_ss = s_scaler.fit_transform(X)
titles = ['Standard Scaler Data - Scatterplot View', 'Standard Scaler Data - Distribution View']
plot_distributions_3d(X_ss, titles)

m_scaler = preprocessing.MinMaxScaler()
X_mm = m_scaler.fit_transform(X)
titles = ['MinMax Scaler Data - Scatterplot View', 'MinMax Scaler Data - Distribution View']
plot_distributions_3d(X_mm, titles)

r_scaler = preprocessing.RobustScaler()
X_rs = r_scaler.fit_transform(X)
titles = ['Robust Scaler Data - Scatterplot View', 'Robust Scaler Data - Distribution View']
plot_distributions_3d(X_rs, titles)