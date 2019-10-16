import pandas as pd
import os
from sklearn import dummy

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.info())
print(titanic_train.columns)

X_train = titanic_train[ ['SibSp', 'Parch'] ]
y_train = titanic_train['Survived']
dummy_estimator = dummy.DummyClassifier(strategy="uniform", random_state=10)
dummy_estimator.fit(X_train, y_train)

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(titanic_test.info())
X_test = titanic_test[ ['SibSp', 'Parch'] ]
titanic_test['Survived'] = dummy_estimator.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)
