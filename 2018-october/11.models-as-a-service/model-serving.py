from sklearn.externals import joblib
import pandas as pd
import numpy as np
import os

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())

path = 'C:\\Users\\Algorithmica\\Downloads'
objects_map = joblib.load( os.path.join(path, 'deployment.pkl') )

imputable_cont_features = objects_map.get('imputable-features')
cont_imputer = objects_map.get('cont-imputer')
titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])

cat_imputer = objects_map.get('cat-imputer')
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])

titanic_test['FamilySize'] = titanic_test['SibSp'] +  titanic_test['Parch'] + 1
titanic_test['FamilyGroup'] = titanic_test['FamilySize'].map(objects_map.get('family-size-func'))

titanic_test['Sex'] = objects_map.get('le-sex').transform(titanic_test['Sex'])
titanic_test['Embarked'] = objects_map.get('le-emb').transform(titanic_test['Embarked'])
titanic_test['Pclass'] = objects_map.get('le-pclass').transform(titanic_test['Pclass'])
titanic_test['FamilyGroup'] = objects_map.get('le-fgroup').transform(titanic_test['FamilyGroup'])

cat_features = objects_map.get('cat-features')
tmp1 = objects_map.get('ohe').transform(titanic_test[cat_features]).toarray()

cont_features = objects_map.get('cont-features')
tmp2 = titanic_test[cont_features].values
X_test = np.concatenate((tmp1,tmp2), axis=1)

estimator = objects_map.get('estimator')
titanic_test['Survived'] = estimator.predict(X_test)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)
