from sklearn import tree
import os
import utilities as util
from sklearn.datasets import  make_circles

os.chdir('E:/decision-trees')
X, y = make_circles(n_samples=1000, noise=0.05, factor=0.3, random_state=0)
util.plot_data(X, y)

tree_estimator = tree.DecisionTreeClassifier(random_state=2017, max_depth=5)
tree_estimator.fit(X, y)
util.plot_decision_boundary(lambda x: tree_estimator.predict(x), X, y)


