import pandas as pd
from sklearn import tree, model_selection
import sklearn
from sklearn.externals import joblib


print(sklearn.__version__)
titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

cat_columns = ['Sex', 'Embarked', 'Pclass']
titanic_train1 = pd.get_dummies(titanic_train, columns = cat_columns)
titanic_train1.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train['Survived']
classifier = tree.DecisionTreeClassifier()
res = model_selection.cross_validate(classifier, X_train, y_train, cv=10)
print(res.get('test_score').mean())
classifier.fit(X_train, y_train)
joblib.dump(classifier, "C:/Users/Algorithmica/Downloads/all/titanic_dt_v1.pkl")
joblib.dump(cat_columns, "C:/Users/Algorithmica/Downloads/all/features_v1.pkl")
