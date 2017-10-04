from sklearn import tree
import os
import utilities as util
import pandas as pd
import numpy as np

os.chdir('E:/decision-trees')

tamu = pd.read_csv("tamu.txt", sep=' ', header=None)

#explore the dataframe
tamu.shape
tamu.info()

X = np.array(tamu[[1,0]])
y = np.array(tamu[2])

util.plot_data(X, y)

tree_estimator = tree.DecisionTreeClassifier(random_state=2017, max_depth=1)
tree_estimator.fit(X, y)
util.plot_decision_boundary(lambda x: tree_estimator.predict(x), X, y)


