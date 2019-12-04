import sys
sys.path.append("E:/New Folder/utils")

import classification_utils as cutils
from sklearn import model_selection, naive_bayes, preprocessing
import seaborn as sns

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=2, n_classes=2, weights=[0.5, 0.5], class_sep=2)
X, y = cutils.generate_nonlinear_synthetic_data_classification2(n_samples=1000, noise=0.1)
cutils.plot_data_2d_classification(X, y)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2, random_state=1)
cutils.plot_data_2d_classification(X_train, y_train)

sns.distplot(X_train[:,0], hist=False)
sns.distplot(X_train[:,1], hist=False)

#grid search for parameter values
gnb_estimator = naive_bayes.GaussianNB()
gnb_grid  = {'priors':[None] }
final_estimator = cutils.grid_search_best_model(gnb_estimator, gnb_grid, X_train, y_train)
cutils.plot_model_2d_classification(final_estimator, X_train, y_train)

final_estimator.predict_proba(X_test)
