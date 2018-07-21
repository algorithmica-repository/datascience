from sklearn.externals import joblib
import pandas as pd

def predict():
     cat_columns = joblib.load("C:/Users/Algorithmica/Downloads/all/features_v1.pkl");
     classifier = joblib.load("C:/Users/Algorithmica/Downloads/all/titanic_dt_v1.pkl");
     titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
     print(titanic_test.shape)
     print(titanic_test.info())
     titanic_test.loc[titanic_test['Fare'].isnull() == True, 'Fare'] = titanic_test['Fare'].mean()

     titanic_test1 = pd.get_dummies(titanic_test, columns = cat_columns)
     titanic_test1.drop(['PassengerId', 'Name', 'Age', 'Ticket', 'Cabin'], axis=1, inplace=True)
     predictions = classifier.predict(titanic_test1)
     print(predictions)

predict()
