import pandas as pd
import os
from sklearn import tree, model_selection

path = 'C:\\Users\\Algorithmica\\Downloads'
titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

features = ['Parch','SibSp']
X_train = titanic_train[features]
y_train = titanic_train[['Survived']]
classifer = tree.DecisionTreeClassifier()
classifer.fit(X_train,y_train)
results = model_selection.cross_validate(classifer, X_train, y_train, cv = 10)
print(results.get('test_score').mean())
print(results.get('train_score').mean())