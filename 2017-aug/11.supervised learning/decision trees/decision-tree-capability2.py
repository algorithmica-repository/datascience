from sklearn import tree
import os
import utilities as util
from sklearn.datasets import make_moons

os.chdir('E:/decision-trees')
X, y = make_moons(n_samples=1000, noise=0.05, random_state=0)
util.plot_data(X, y)

tree_estimator = tree.DecisionTreeClassifier(random_state=2017, max_depth=7)
tree_estimator.fit(X, y)
util.plot_decision_boundary(lambda x: tree_estimator.predict(x), X, y)


