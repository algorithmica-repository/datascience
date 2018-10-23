#import entire sklearn module
import sklearn
clf1 = sklearn.tree.DecisionTreeClassifier()

#import tree submodule of sklearn
from sklearn import tree
clf2 = tree.DecisionTreeClassifier()

#import only decistionTreeClassifer class from sklearn.tree submodule
from sklearn.tree import DecisionTreeClassifier
clf3 = DecisionTreeClassifier()

from pandas import read_csv, DataFrame
df1 = read_csv("")
df2 = DataFrame({})
