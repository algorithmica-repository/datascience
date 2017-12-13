import pandas as pd

titanic_train = pd.read_csv('C:/Users/Algorithmica/Downloads/titanic_train.csv')
print(type(titanic_train))
titanic_train.shape
titanic_train.info()
import pandas as pd

titanic_train = pd.read_csv('C:/Users/Algorithmica/Downloads/titanic_train.csv')
print(type(titanic_train))
titanic_train.shape
titanic_train.info()

titanic_train.groupby(['Survived']).size()

titanic_test = pd.read_csv('C:/Users/Algorithmica/Downloads/titanic_test.csv')
titanic_test.shape
titanic_test.info()

titanic_test['Survived'] = 0
titanic_test.to_csv('D:/submission.csv', columns=['PassengerId','Survived'],index=False)


