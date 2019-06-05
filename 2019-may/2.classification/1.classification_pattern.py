import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection
import numpy as np
import pandas as pd
import os

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=4, weights=[0.3,0.3,0.2,0.2])
cutils.plot_data_2d_classification(X, y)

X, y = cutils.generate_nonlinear_synthetic_data_classification1(n_samples=1000, n_features=2, n_classes=2)
cutils.plot_data_2d_classification(X, y)

X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.05)
cutils.plot_data_2d_classification(X, y)

X, y = cutils.generate_nonlinear_synthetic_data_classification3(n_samples=1000, noise=0.05)
cutils.plot_data_2d_classification(X, y)

#3-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=3, n_classes=5, weights=list(np.repeat(0.1,5)))
cutils.plot_data_3d_classification(X, y)

X, y = cutils.generate_nonlinear_synthetic_data_classification1(n_samples=1000, n_features=3, n_classes=2)
cutils.plot_data_3d_classification(X, y)

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
features = ['Pclass', 'Fare', 'SibSp']
X_train = titanic_train[features]
y_train = titanic_train['Survived']
cutils.plot_data_3d_classification(X_train, y_train)
