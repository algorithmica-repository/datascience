import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin 
from sklearn import metrics

class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, n_centres=10, width_factor=2.0):
        self.n_centres = n_centres
        self.width_factor = width_factor
   
    def fit(self, X, y=None):
        # create N centers spread along the data range
        self.centers_ = np.linspace(X.min(), X.max(), self.n_centres)
        self.width_ = self.width_factor * (self.centers_[1] - self.centers_[0])
        return self
        
    def transform(self, X):
        if isinstance(X, np.ndarray):
            X = X
        else:
            X = X.values
        arg = (X[:,:,np.newaxis] - self.centers_) / self.width_
        return np.exp(-0.5 * np.sum(arg ** 2, 1))

class KernelTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, ktype, gamma=None, degree=None, intercept=1):
        self.ktype = ktype
        self.gamma = gamma
        self.degree = degree
        self.intercept = intercept
    
    def fit(self, X, y=None):
        self.X = X
        return self
        
    def transform(self, X):
        if self.ktype == 'poly':
            return metrics.pairwise.polynomial_kernel(X, self.X, degree=self.degree, gamma = self.gamma, coef0 = self.intercept)
        elif self.ktype == 'rbf':
            return metrics.pairwise.rbf_kernel(X, self.X, gamma = self.gamma)
        else:
            return None

def polynomial_kernel(X, Z, d):
    m1 = X.shape[0]
    m2 = Z.shape[0]
    K = np.zeros((m1, m2))
    for i in range(m1):
        for j in range(m2):
            K[i,j] = (np.dot(X[i,:], Z[j,:]) + 1)**d            
    return K

def gaussian_kernel(X, Z, gamma):
    m1 = X.shape[0]
    m2 = Z.shape[0]    
    K = np.zeros((m1, m2))    
    for i in range(m1):
        for j in range(m2):
            K[i,j] = np.exp( -gamma * np.linalg.norm(X[i,:] - Z[j,:])**2 )            
    return K