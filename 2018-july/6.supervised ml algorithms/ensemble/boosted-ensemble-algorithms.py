import pandas as pd
from sklearn import tree, preprocessing, ensemble, model_selection
from sklearn_pandas import CategoricalImputer
import pydot
import io

#extracting all the trees build by random forest algorithm
def export_all_trees(best_est, prefix):
    n_tree = 0
    for est in best_est.estimators_: 
        dot_data = io.StringIO()
        tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
        graph.write_pdf("C:/Users/Algorithmica/Downloads/all/" + prefix + str(n_tree) + ".pdf")
        n_tree = n_tree + 1
        
titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

imputable_cont_features = ['Age','Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic_train[imputable_cont_features])
print(cont_imputer.statistics_)
titanic_train[imputable_cont_features] = cont_imputer.transform(titanic_train[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic_train['Embarked'])
print(cat_imputer.fill_)
titanic_train['Embarked'] = cat_imputer.transform(titanic_train['Embarked'])

#creaate categorical age column from age
def convert_age(age):
    if(age >= 0 and age <= 18): 
        return 'Teen'
    elif(age <= 40): 
        return 'Young'
    elif(age <= 60): 
        return 'Middle'
    else: 
        return 'Old'
titanic_train['Age1'] = titanic_train['Age'].map(convert_age)

titanic_train['FamilySize'] = titanic_train['SibSp'] +  titanic_train['Parch'] + 1

def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=5): 
        return 'Medium'
    else: 
        return 'Large'
titanic_train['FamilySize1'] = titanic_train['FamilySize'].map(convert_familysize)
     
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)
    
cat_columns = ['Sex', 'Embarked', 'Pclass', 'Title', 'Age1', 'FamilySize1']
titanic_train1 = pd.get_dummies(titanic_train, columns = cat_columns)
titanic_train1.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)

X_train = titanic_train1
y_train = titanic_train['Survived']

dt = tree.DecisionTreeClassifier(random_state=100)
ada_estimator = ensemble.AdaBoostClassifier(base_estimator=dt, random_state=100)
ada_grid = {'n_estimators':[5], 'learning_rate':[0.5,1,1.5], 'base_estimator__max_depth':[2,3,4]}
grid_ada_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, cv=10)
grid_ada_estimator.fit(X_train, y_train)
best_est = grid_ada_estimator.best_estimator_
print(grid_ada_estimator.best_score_)
print(best_est.estimators_)
export_all_trees(best_est, "ada")



