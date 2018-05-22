import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper,CategoricalImputer
import numpy as np
from sklearn import tree, model_selection, ensemble


print(pd.__version__)
class MyLabelBinarizer(preprocessing.LabelBinarizer):
    def transform(self, y):
        Y = super().transform(y)
        if self.y_type_ == 'binary':
            return np.hstack((1-Y, Y))
        else:
            return Y

    def inverse_transform(self, Y, threshold=None):
        if self.y_type_ == 'binary':
            return super().inverse_transform(Y[:, 1], threshold)
        else:
            return super().inverse_transform(Y, threshold)
        
#changes working directory
path = "C:/Users/Thimma Reddy/Documents/GitHub/datascience/2014/kaggle/titanic/data"

titanic_train = pd.read_csv(os.path.join(path,"train.csv"))
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv(os.path.join(path,"test.csv"))
titanic_test.shape
titanic_test.info()

# normalize the titles
normalized_titles = {
    "Capt":       "Officer",
    "Col":        "Officer",
    "Major":      "Officer",
    "Jonkheer":   "Royalty",
    "Don":        "Royalty",
    "Sir" :       "Royalty",
    "Dr":         "Officer",
    "Rev":        "Officer",
    "the Countess":"Royalty",
    "Dona":       "Royalty",
    "Mme":        "Mrs",
    "Mlle":       "Miss",
    "Ms":         "Mrs",
    "Mr" :        "Mr",
    "Mrs" :       "Mrs",
    "Miss" :      "Miss",
    "Master" :    "Master",
    "Lady" :      "Royalty"
}
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title).map(normalized_titles)
titanic_test['Title'] = titanic_test['Name'].map(extract_title).map(normalized_titles)
print(titanic_train['Title'].value_counts())

values = np.union1d(titanic_train['Title'], titanic_test['Title'])
titanic_train['Title'] = titanic_train['Title'].astype(pd.api.types.CategoricalDtype(categories=values))
titanic_test['Title'] = titanic_test['Title'].astype(pd.api.types.CategoricalDtype(categories=values))

# view the median Age by the grouped features 
grouped = titanic_train.groupby(['Sex','Pclass', 'Title'])  
grouped.Age.median()
titanic_train['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))

#impute missing values for continuous features
imputable_cont_features = ['Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])

# fill Cabin NaN with U for unknown
titanic_train.Cabin = titanic_train.Cabin.fillna('U')
titanic_train.Cabin = titanic_train.Cabin.map(lambda x: x[0])

# size of families (including the passenger)
titanic_train['FamilySize'] = titanic_train.Parch + titanic_train.SibSp + 1

cat_features = ['Sex', 'Embarked', 'Pclass', 'Cabin', 'Title']
cont_features = ['Age', 'Fare', 'SibSp', 'Parch', 'FamilySize']

feature_defs = []
for col_name in cat_features:
    feature_defs.append((col_name, MyLabelBinarizer()))
    
for col_name in cont_features:
    feature_defs.append((col_name, None))

mapper = DataFrameMapper(feature_defs, input_df=True, df_out=True)
mapper.fit(titanic_train)
X_train = mapper.transform(titanic_train)
y_train = titanic_train['Survived']

kfold = model_selection.StratifiedKFold(n_splits=10)
random_state = 100

rf_classifier = ensemble.RandomForestClassifier(random_state=random_state)
rf_grid = {'max_depth':list(range(7,14)), 'n_estimators':list(range(10,100,10)),  
           'min_samples_split':list(range(4,11)), 'min_samples_leaf':list(range(2,5))}
grid_rf_classifier = model_selection.GridSearchCV(rf_classifier, rf_grid, cv=kfold, n_jobs=1, verbose=1)
grid_rf_classifier.fit(X_train, y_train)
print(grid_rf_classifier.best_score_)
rf_best_model = grid_rf_classifier.best_estimator_

ext_classifier = ensemble.ExtraTreesClassifier(random_state=random_state)
ext_grid = {"max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [False],
              "n_estimators" :[100,300],
              "criterion": ["gini"]}
grid_ext_classifier = model_selection.GridSearchCV(ext_classifier, ext_grid, cv=kfold, n_jobs=1, verbose=1)
grid_ext_classifier.fit(X_train, y_train)
print(grid_ext_classifier.best_score_)
ext_best_model = grid_ext_classifier.best_estimator_

dt = tree.DecisionTreeClassifier()
ada_classifier = ensemble.AdaBoostClassifier(dt, random_state=random_state)
ada_grid = {"base_estimator__criterion" : ["gini", "entropy"],
              "base_estimator__splitter" :   ["best", "random"],
              "algorithm" : ["SAMME","SAMME.R"],
              "n_estimators" :list(range(10,100,10)),
              "learning_rate":  [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5]}
grid_ada_classifier = model_selection.GridSearchCV(ada_classifier,ada_grid, cv=kfold, n_jobs= 1, verbose = 1)
grid_ada_classifier.fit(X_train, y_train)
print(grid_ada_classifier.best_score_)
ada_best_model = grid_ada_classifier.best_estimator_

gbm_classifier = ensemble.GradientBoostingClassifier(random_state=random_state)
gbm_grid = {'loss' : ["deviance"],
              'n_estimators' : [100,200,300],
              'learning_rate': [0.1, 0.05, 0.01],
              'max_depth': [4, 8],
              'min_samples_leaf': [100,150],
              'max_features': [0.3, 0.1] 
             }
grid_gbm_classifier = model_selection.GridSearchCV(gbm_classifier,gbm_grid, cv=kfold, n_jobs= 1, verbose = 1)
grid_gbm_classifier.fit(X_train, y_train)
print(grid_gbm_classifier.best_score_)
gbm_best_model = grid_gbm_classifier.best_estimator_

voting_classifier = ensemble.VotingClassifier([
        ('rfc', rf_best_model), ('adac',ada_best_model),
        ('gbc', gbm_best_model),('éxtc', ext_best_model)], 
        voting='soft', n_jobs=4)

voting_model = voting_classifier.fit(X_train, y_train)
results = model_selection.cross_validate(voting_classifier, X_train, y_train, cv = kfold, return_train_score=True)
print(results.get('test_score').mean())
print(results.get('train_score').mean())


titanic_test['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))
titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test.Cabin = titanic_test.Cabin.fillna('U')
titanic_test.Cabin = titanic_test.Cabin.map(lambda x: x[0])
titanic_test['FamilySize'] = titanic_test.Parch + titanic_test.SibSp + 1
X_test = mapper.transform(titanic_test)

titanic_test['Survived'] = voting_model.predict(X_test)
titanic_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)
