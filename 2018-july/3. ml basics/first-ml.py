import pandas as pd
from sklearn import tree

titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

features = ['SibSp', 'Parch']
X_train = titanic_train[features]
y_train = titanic_train['Survived']
classifer = tree.DecisionTreeClassifier()
classifer.fit(X_train, y_train)

titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
print(titanic_test.shape)
print(titanic_test.info())
X_test = titanic_test[features]
titanic_test['Survived'] = classifer.predict(X_test)
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/all/submission.csv", columns=['PassengerId', 'Survived'], index=False)
