import os
import pandas as pd
from sklearn import tree
import pydot
from sklearn import preprocessing
from sklearn import model_selection
from sklearn.externals import joblib
import io

#returns current working directory
os.getcwd()
#changes working directory
os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("train.csv")

#EDA
titanic_train.shape
titanic_train.info()

#type conversions
titanic_train['Survived'] = titanic_train['Survived'].astype('category')
titanic_train['Pclass'] = titanic_train['Pclass'].astype('category')
titanic_train['Sex'] = titanic_train['Sex'].astype('category')
titanic_train['Embarked'] = titanic_train['Embarked'].astype('category')

sum(titanic_train['Pclass'].isnull())
titanic_train.apply(lambda x : sum(x.isnull()))
titanic_train.Embarked[titanic_train['Embarked'].isnull()] = 'S'


titanic_train1 = titanic_train.copy()
le = preprocessing.LabelEncoder()
titanic_train1.Sex = le.fit_transform(titanic_train1.Sex)
titanic_train1.Embarked = le.fit_transform(titanic_train.Embarked)
titanic_train1.Pclass = le.fit_transform(titanic_train1.Pclass)

#ohe = preprocessing.OneHotEncoder()
#titanic_train1.Sex = ohe.fit_transform(titanic_train1.Sex)
#titanic_train1.Embarked = le.fit_transform(titanic_train.Embarked)
#titanic_train1.Pclass = le.fit_transform(titanic_train1.Pclass)

X_train = titanic_train1[['Sex','Embarked','Pclass','Fare']]
y_train = titanic_train1['Survived']

dt = tree.DecisionTreeClassifier()
cv_scores = model_selection.cross_val_score(dt, X_train, y_train, cv=10)
#compute bias of model
print( cv_scores.mean() )
#compute variance of model
print( cv_scores.std())

#build model using entire train data
dt.fit(X_train,y_train)
joblib.dump(dt, "dt1.pkl")

#visualize the model built by ML algorithms
dot_data = io.StringIO() 
tree.export_graphviz(dt, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("dt2.pdf")


