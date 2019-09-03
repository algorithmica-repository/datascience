import sys
path = 'E://utils'
sys.path.append(path)

import regression_utils as rutils
from sklearn import model_selection


#linear pattern in 2d
X, y = rutils.generate_linear_synthetic_data_regression(n_samples=200, n_features=1, 
                                                 n_informative=1,
                                                 noise = 100)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_2d_regression(X_train, y_train)

#non-linear pattern in 2d
X, y = rutils.generate_nonlinear_synthetic_data_regression(n_samples=200, n_features=1)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_2d_regression(X_train, y_train)

#linear pattern in 3d
X, y = rutils.generate_linear_synthetic_data_regression(n_samples=200, n_features=2, 
                                                 n_informative=2,
                                                 noise = 10)
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.1, random_state=1)
rutils.plot_data_3d_regression(X_train, y_train)


