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
X_train = utils.ohe(titanic_train1, cat_features)
y_train = titanic_train['Survived']

#embedded feature selectors
rf_estimator = ensemble.RandomForestClassifier()
rf_grid  = {'max_depth':list(range(1,9)), 'n_estimators':list(range(1,300,100)) }
rf_final_estimator = cutils.grid_search_best_model(rf_estimator, rf_grid, X_train, y_train)
embedded_selector = feature_selection.SelectFromModel(rf_final_estimator, prefit=True, threshold='mean')
X_train1 = embedded_selector.transform(X_train)
utils.plot_feature_importances(rf_final_estimator, X_train)

gb_estimator = ensemble.GradientBoostingClassifier()
gb_grid  = {'max_depth':[1,2,3], 'n_estimators':list(range(50,300, 100)), 'learning_rate':[0.001, 0.1, 1.0] }
gb_final_estimator = cutils.grid_search_best_model(gb_estimator, gb_grid, X_train, y_train)
embedded_selector = feature_selection.SelectFromModel(gb_final_estimator, prefit=True, threshold='mean')
X_train1 = embedded_selector.transform(X_train)
utils.plot_feature_importances(gb_final_estimator, X_train)

svm_estimator = svm.LinearSVC()
svm_grid = {'C':[0.01, 0.1, 1] }
svm_final_estimator = cutils.grid_search_best_model(svm_estimator, svm_grid, X_train, y_train)
embedded_selector = feature_selection.SelectFromModel(svm_final_estimator, prefit=True, threshold='mean')
X_train1 = embedded_selector.transform(X_train)

