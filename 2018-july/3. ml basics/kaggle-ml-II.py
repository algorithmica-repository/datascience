#fix the issues 
#transform categoricalfeatures to continuous features
#handle missing data: imputation
import pandas as pd
from sklearn import tree
import pydot
import io

print(pd.__version__)
titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

titanic_train.loc[titanic_train['Age'].isnull() == True, 'Age'] = titanic_train['Age'].mean()

cat_columns = ['Sex', 'Embarked', 'Pclass']
titanic_train1 = pd.get_dummies(titanic_train, columns = cat_columns)
titanic_train1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train['Survived']
classifer = tree.DecisionTreeClassifier()
classifer.fit(X_train, y_train)

#visualize the deciion tree
dot_data = io.StringIO() 
tree.export_graphviz(classifer, out_file = dot_data, feature_names = X_train.columns)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
graph.write_pdf("C:/Users/Algorithmica/Downloads/all/tree.pdf")

titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
print(titanic_test.shape)
print(titanic_test.info())

titanic_test.loc[titanic_test['Age'].isnull() == True, 'Age'] = titanic_test['Age'].mean()
titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()
titanic_test1 = pd.get_dummies(titanic_test, columns = cat_columns)
titanic_test1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

X_test = titanic_test1
titanic_test['Survived'] = classifer.predict(X_test)
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/all/submission.csv", columns=['PassengerId', 'Survived'], index=False)
