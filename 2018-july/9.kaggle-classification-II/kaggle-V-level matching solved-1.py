import pandas as pd
from sklearn import tree, preprocessing, ensemble, model_selection, feature_selection
from sklearn_pandas import CategoricalImputer
import pydot
import io
import numpy as np
import seaborn as sns

#extracting all the trees build by random forest algorithm
def export_all_trees(best_est, prefix):
    n_tree = 0
    for est in best_est.estimators_: 
        dot_data = io.StringIO()
        tree.export_graphviz(est, out_file = dot_data, feature_names = X_train.columns)
        graph = pydot.graph_from_dot_data(dot_data.getvalue())[0] 
        graph.write_pdf("C:/Users/Algorithmica/Downloads/all/" + prefix + str(n_tree) + ".pdf")
        n_tree = n_tree + 1
        
def plot_feature_importances(classifier, X_train, y_train):
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("RF feature importances")       
    
titanic_train = pd.read_csv("C:/Users/Algorithmica/Downloads/all/train.csv")
print(titanic_train.shape)
print(titanic_train.info())

titanic_test = pd.read_csv("C:/Users/Algorithmica/Downloads/all/test.csv")
print(titanic_test.shape)
print(titanic_test.info())

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

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])

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
 
titanic_test['Age1'] = titanic_test['Age'].map(convert_age)
titanic_test['FamilySize'] = titanic_test['SibSp'] +  titanic_test['Parch'] + 1
titanic_test['FamilySize1'] = titanic_test['FamilySize'].map(convert_familysize)
titanic_test['Title'] = titanic_test['Name'].map(extract_title)

titanic = pd.concat([titanic_train, titanic_test])

cat_columns = ['Sex', 'Embarked', 'Pclass', 'Title', 'Age1', 'FamilySize1']
titanic = pd.get_dummies(titanic, columns = cat_columns)
titanic.drop(['PassengerId', 'Name', 'Ticket', 'Cabin', 'Survived'], axis=1, inplace=True)

X_train = titanic[0:titanic_train.shape[0]]
y_train = titanic_train['Survived']

rf_estimator = ensemble.RandomForestClassifier(random_state=100)
rf_estimator.fit(X_train, y_train)
print(rf_estimator.feature_importances_)
plot_feature_importances(rf_estimator, X_train, y_train)
selector = feature_selection.SelectFromModel(rf_estimator, prefit=True, threshold = '0.8*mean')
X_train_new = selector.transform(X_train)

dt = tree.DecisionTreeClassifier(random_state=100)
ada_estimator = ensemble.AdaBoostClassifier(base_estimator=dt, random_state=100)
ada_grid = {'n_estimators':[50,100,150,200], 'learning_rate':[0.5,1,1.5]}
grid_ada_estimator = model_selection.GridSearchCV(ada_estimator, ada_grid, cv=10)
grid_ada_estimator.fit(X_train_new, y_train)
best_est = grid_ada_estimator.best_estimator_
print(grid_ada_estimator.best_score_)
#export_all_trees(best_est, "ada")

X_test = titanic[titanic_train.shape[0]:]
X_test_new = selector.transform(X_test)
titanic_test['Survived'] = best_est.predict(X_test_new)
titanic_test.to_csv("C:/Users/Algorithmica/Downloads/all/submission.csv", columns=['PassengerId', 'Survived'], index=False)


