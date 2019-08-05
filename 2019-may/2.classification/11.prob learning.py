import sys
sys.path.append("E:/")

import classification_utils as cutils
from sklearn import model_selection, naive_bayes

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.5, 0.5], class_sep=3)
X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

nb_estimator = naive_bayes.GaussianNB()
nb_estimator.fit(X_train, y_train)
print(nb_estimator.class_prior_)
print(nb_estimator.sigma_)
print(nb_estimator.theta_)
cutils.plot_model_2d_classification(nb_estimator, X_train, y_train)

#predict distances and classes for test data
print(nb_estimator.predict(X_test))

