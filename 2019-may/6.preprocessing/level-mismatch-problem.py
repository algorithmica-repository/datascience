import sys
sys.path.append("E:/")

import pandas as pd
import os
import common_utils as utils
from sklearn import preprocessing, neighbors, svm, linear_model, ensemble, pipeline, model_selection, feature_selection
import classification_utils as cutils
import seaborn as sns

dir = 'E:/'
titanic_train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(titanic_train.shape)
print(titanic_train.info())

def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)
sns.factorplot(x="Title", hue="Survived", data=titanic_train, kind="count", size=6)

#create family size feature from sibsp, parch
titanic_train['FamilySize'] = titanic_train['SibSp'] +  titanic_train['Parch'] + 1
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()

#create family group feature from family-size
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=5): 
        return 'Medium'
    else: 
        return 'Large'
titanic_train['FamilyGroup'] = titanic_train['FamilySize'].map(convert_familysize)
sns.factorplot(x="FamilyGroup", hue="Survived", data=titanic_train, kind="count", size=6)

sns.countplot(x='Cabin',data=titanic_train)
titanic_train['Cabin'] = titanic_train['Cabin'].fillna('U')

titanic_train1 = utils.drop_features(titanic_train, ['PassengerId', 'Name', 'Survived', 'Ticket'])

#type casting
utils.cast_to_cat(titanic_train1, ['Sex', 'Pclass', 'Embarked', 'Title', 'FamilyGroup', 'Cabin'])

cat_features = utils.get_categorical_features(titanic_train1)
print(cat_features)
cont_features = utils.get_continuous_features(titanic_train1)
print(cont_features)

#handle missing data(imputation)
cat_imputers = utils.get_categorical_imputers(titanic_train1, cat_features)
titanic_train1[cat_features] = cat_imputers.transform(titanic_train1[cat_features])
cont_imputers = utils.get_continuous_imputers(titanic_train1, cont_features)
titanic_train1[cont_features] = cont_imputers.transform(titanic_train1[cont_features])

#one hot encoding
titanic_train1 = utils.ohe(titanic_train1, cat_features)

#scaling
#scaler = preprocessing.StandardScaler()
#titanic_train1 = scaler.fit_transform(titanic_train1)
y_train = titanic_train['Survived']

#feature selection
rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'max_depth':list(range(1,9)), 'n_estimators':list(range(1,300,100)) }
rf_final_estimator = cutils.grid_search_best_model(rf_estimator, rf_grid, titanic_train1, y_train)
X_train = utils.select_features(rf_final_estimator, titanic_train1, threshold='mean')

kernel_svm_estimator = svm.SVC(kernel='rbf')
kernel_svm_grid = {'gamma':[0.001, 0.01], 'C':[100, 1e4, 1e5] }
svm_final_estimator = cutils.grid_search_best_model(kernel_svm_estimator, kernel_svm_grid, X_train, y_train)

titanic_test = pd.read_csv(os.path.join(dir, 'test.csv'))

print(titanic_test.shape)
print(titanic_test.info())

titanic_test['Title'] = titanic_test['Name'].map(extract_title)
titanic_test['FamilySize'] = titanic_test['SibSp'] +  titanic_test['Parch'] + 1
titanic_test['FamilyGroup'] = titanic_test['FamilySize'].map(convert_familysize)
titanic_test['Cabin'] = titanic_test['Cabin'].fillna('U')

titanic_test1 = utils.drop_features(titanic_test, ['PassengerId', 'Name', 'Ticket'])

utils.cast_to_cat(titanic_test1, ['Sex', 'Pclass', 'Embarked', 'Title', 'FamilyGroup', 'Cabin'])

cat_features = utils.get_categorical_features(titanic_test1)
print(cat_features)
cont_features = utils.get_continuous_features(titanic_test1)
print(cont_features)

titanic_test1[cat_features] = cat_imputers.transform(titanic_test1[cat_features])
titanic_test1[cont_features] = cont_imputers.transform(titanic_test1[cont_features])

print(titanic_test1.info())

#level mismatch issue
titanic_test1 = utils.ohe(titanic_test1, cat_features)
#titanic_test1 = scaler.transform(titanic_test1)
X_test = utils.select_features(rf_final_estimator, titanic_test1, threshold='mean')

titanic_test['Survived'] = svm_final_estimator.predict(X_test)
titanic_test.to_csv(os.path.join(dir, 'submission.csv'), columns=['PassengerId', 'Survived'], index=False)
