import pandas as pd
import os
from sklearn import ensemble,tree
from sklearn import preprocessing
from sklearn import model_selection
from sklearn import feature_selection
from sklearn_pandas import CategoricalImputer
import seaborn as sns
import io
import pydot

os.chdir("C:\\Users\\Algorithmica\\Downloads")

titanic_train = pd.read_csv("titanic_train.csv")
titanic_train.shape
titanic_train.info()

titanic_test = pd.read_csv("titanic_test.csv")
titanic_test.shape

titanic_all = pd.concat([titanic_train, titanic_test])
titanic_all.shape
titanic_all.info()

#impute missing values for continuous features
imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_all[imputable_cont_features])
titanic_all[imputable_cont_features] = cont_imputer.transform(titanic_all[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_all['Embarked'])
titanic_all['Embarked'] = cat_imputer.transform(titanic_all['Embarked'])

titanic_all['FamilySize'] = titanic_all['SibSp'] +  titanic_all['Parch'] + 1

def convert_family_size(size):
    if(size == 1): 
        return 'Single'
    elif(size <=3): 
        return 'Small'
    elif(size <= 6): 
        return 'Medium'
    else: 
        return 'Large'
titanic_all['FamilyCategory'] = titanic_all['FamilySize'].map(convert_family_size)

def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_all['Title'] = titanic_all['Name'].map(extract_title)

tmp_df = titanic_all[0:titanic_train.shape[0]]
sns.FacetGrid(tmp_df, row="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()
sns.factorplot(x="Title", hue="Survived", data=tmp_df, kind="count", size=6)
sns.factorplot(x="FamilyCategory", hue="Survived", data=tmp_df, kind="count", size=6)
sns.FacetGrid(tmp_df, row="Survived",size=8).map(sns.kdeplot, "Age").add_legend()


titanic_all.drop(['PassengerId', 'Name', 'Cabin','Ticket','Survived'], axis=1, inplace=True)

features = ['Sex', 'Embarked', 'Pclass', 'Title', 'FamilyCategory']
titanic_all = pd.get_dummies(titanic_all, columns=features)

X_train = titanic_all[0:titanic_train.shape[0]]
y_train = titanic_train['Survived']

#applying feature selection algorithm to get impactful features
ada = ensemble.AdaBoostClassifier()
ada.fit(X_train, y_train)

features = pd.DataFrame({'feature':X_train.columns, 'importance':ada.feature_importances_})
features.sort_values(by=['importance'], ascending=True, inplace=True)
features.set_index('feature', inplace=True)
features.plot(kind='barh', figsize=(20, 20))

fs_model = feature_selection.SelectFromModel(ada, prefit=True)
X_train1 = fs_model.transform(X_train)
X_train1.shape 
selected_features = X_train.columns[fs_model.get_support()]


#build model using selected features
ada_estimator = ensemble.AdaBoostClassifier(random_state=100)
ada_grid = {'n_estimators':[3,4,5], 'learning_rate':[0.05,1.0]}
grid_ada_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, cv=10)
grid_ada_estimator.fit(X_train1, y_train)

print(grid_ada_estimator.best_score_)
print(grid_ada_estimator.best_params_)

final_model = grid_ada_estimator.best_estimator_
n_tree = 0
for est in final_model.estimators_: 
    dot_data = io.StringIO()
    tree.export_graphviz(est, out_file = dot_data, feature_names = selected_features)
    graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
    graph.write_pdf("ada" + str(n_tree) + ".pdf")
    n_tree = n_tree + 1
