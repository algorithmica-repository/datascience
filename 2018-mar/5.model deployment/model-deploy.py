import pandas as pd
import os
from sklearn import tree, model_selection
from sklearn.externals import joblib

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

#copy the model to pkl file and keep the model file at required server location
joblib.dump(classifer,os.path.join(path, 'dt-v1.pkl') )
#cross check the dumped model with load
classifier_loaded = joblib.load(os.path.join(path, 'dt-v1.pkl') )
