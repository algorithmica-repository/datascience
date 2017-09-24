from sklearn import tree
import os
import utilities as util
from sklearn.datasets import make_classification

os.chdir('E:/decision-trees')
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                           n_informative=2, random_state=7, n_clusters_per_class=1)
util.plot_data(X, y)

tree_estimator = tree.DecisionTreeClassifier(random_state=2017, max_depth=3)
tree_estimator.fit(X, y)
util.plot_decision_boundary(lambda x: tree_estimator.predict(x), X, y)


