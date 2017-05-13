import numpy as np
from sklearn import ensemble

size = 10000
np.random.seed(seed=10)
X_seed = np.random.normal(0, 1, size)
X0 = X_seed + np.random.normal(0, .1, size)
X1 = X_seed + np.random.normal(0, .1, size)
X2 = X_seed + np.random.normal(0, .1, size)
X = np.array([X0, X1, X2]).T
Y = X0 + X1 + X2
  
rf = ensemble.RandomForestRegressor(n_estimators=20, max_features=2, random_state=2017)
rf.fit(X, Y);
rf.feature_importances_