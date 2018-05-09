import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper,CategoricalImputer
import numpy as np
from sklearn import model_selection, ensemble
import seaborn as sns

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
path = 'C:\\Users\\Algorithmica\\Downloads'

titanic_train = pd.read_csv(os.path.join(path,"titanic_train.csv"))
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv(os.path.join(path,"titanic_test.csv"))
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
sns.countplot('Title', data = titanic_train)
sns.factorplot(x='Title', hue='Survived', data=titanic_train,  kind="count", size=6)

titanic_test['Title'] = titanic_test['Name'].map(extract_title).map(normalized_titles)
print(titanic_train['Title'].value_counts())

values = np.union1d(titanic_train['Title'], titanic_test['Title'])
titanic_train['Title'] = titanic_train['Title'].astype('category',categories=values)
titanic_test['Title'] = titanic_test['Title'].astype('category',categories=values)

# view the median Age by the grouped features 
grouped = titanic_train.groupby(['Sex','Pclass', 'Title'])  
grouped.Age.median()
titanic_train['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Age").add_legend()

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
sns.countplot('Cabin', data = titanic_train)
sns.factorplot(x='Cabin', hue='Survived', data=titanic_train,  kind="count", size=6)

# size of families (including the passenger)
titanic_train['FamilySize'] = titanic_train.Parch + titanic_train.SibSp + 1
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()

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

rf_classifier = ensemble.RandomForestClassifier(random_state=100, n_jobs=1, verbose=1, oob_score=True)
rf_grid = {'max_depth':list(range(7,14)), 'n_estimators':list(range(10,100,10)),  
           'min_samples_split':list(range(4,11)), 'min_samples_leaf':list(range(2,5))}
rf_grid_classifier = model_selection.GridSearchCV(rf_classifier, rf_grid, cv=10, refit=True, return_train_score=True)
rf_grid_classifier.fit(X_train, y_train)
results = rf_grid_classifier.cv_results_
print(results.get('params'))
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(rf_grid_classifier.best_params_)
print(rf_grid_classifier.best_score_)
final_model = rf_grid_classifier.best_estimator_

titanic_test['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))
titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test.Cabin = titanic_test.Cabin.fillna('U')
titanic_test.Cabin = titanic_test.Cabin.map(lambda x: x[0])
titanic_test['FamilySize'] = titanic_test.Parch + titanic_test.SibSp + 1
X_test = mapper.transform(titanic_test)

titanic_test['Survived'] = final_model.predict(X_test)
titanic_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)
