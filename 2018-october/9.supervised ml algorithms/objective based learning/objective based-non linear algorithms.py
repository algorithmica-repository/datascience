import pandas as pd
from sklearn import linear_model, svm, model_selection, preprocessing, ensemble, feature_selection, neighbors, naive_bayes
from sklearn_pandas import CategoricalImputer
import numpy as np

#creation of data frames from csv
titanic_train = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_train.csv")
print(titanic_train.info())

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())

titanic = pd.concat((titanic_train, titanic_test), axis = 0)
titanic.drop(['Survived'], axis=1, inplace=True)

#impute missing values for continuous features
imputable_cont_features = ['Age', 'Fare']
cont_imputer = preprocessing.Imputer()
cont_imputer.fit(titanic[imputable_cont_features])
print(cont_imputer.statistics_)
titanic[imputable_cont_features] = cont_imputer.transform(titanic[imputable_cont_features])

#impute missing values for categorical features
cat_imputer = CategoricalImputer()
cat_imputer.fit(titanic['Embarked'])
print(cat_imputer.fill_)
titanic['Embarked'] = cat_imputer.transform(titanic['Embarked'])

#create title feature from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
#extract_title('abc, mr.def')
titanic['Title'] = titanic['Name'].map(extract_title)

#create family size feature from sibsp, parch
titanic['FamilySize'] = titanic['SibSp'] +  titanic['Parch'] + 1

#create family group feature from family-size
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=5): 
        return 'Medium'
    else: 
        return 'Large'
titanic['FamilyGroup'] = titanic['FamilySize'].map(convert_familysize)

#label encoding of categorical (string) features
lab_encoder_sex = preprocessing.LabelEncoder()
lab_encoder_sex.fit(titanic['Sex'])
print(lab_encoder_sex.classes_)
titanic['Sex'] = lab_encoder_sex.transform(titanic['Sex'])

lab_encoder_emb = preprocessing.LabelEncoder()
lab_encoder_emb.fit(titanic['Embarked'])
print(lab_encoder_emb.classes_)
titanic['Embarked'] = lab_encoder_emb.transform(titanic['Embarked'])

lab_encoder_pclass = preprocessing.LabelEncoder()
lab_encoder_pclass.fit(titanic['Pclass'])
print(lab_encoder_pclass.classes_)
titanic['Pclass'] = lab_encoder_pclass.transform(titanic['Pclass'])

lab_encoder_title = preprocessing.LabelEncoder()
lab_encoder_title.fit(titanic['Title'])
print(lab_encoder_title.classes_)
titanic['Title'] = lab_encoder_title.transform(titanic['Title'])

lab_encoder_fgroup = preprocessing.LabelEncoder()
lab_encoder_fgroup.fit(titanic['FamilyGroup'])
print(lab_encoder_fgroup.classes_)
titanic['FamilyGroup'] = lab_encoder_fgroup.transform(titanic['FamilyGroup'])

#one hot encoding of categorical integer features
ohe_features = ['Sex','Embarked','Pclass', 'Title', 'FamilyGroup']
ohe = preprocessing.OneHotEncoder()
ohe.fit(titanic[ohe_features])
print(ohe.n_values_)
tmp1 = ohe.transform(titanic[ohe_features]).toarray()

features = ['Age', 'Fare', 'Parch' , 'SibSp', 'FamilySize']
tmp2 = titanic[features].values

tmp = np.concatenate((tmp1,tmp2), axis=1)
scaler = preprocessing.StandardScaler()
tmp = scaler.fit_transform(tmp)

X_train = tmp[:titanic_train.shape[0]]
y_train = titanic_train['Survived']

#kernel svm - poly kernel
ksvm_estimator = svm.SVC(random_state=100)
ksvm_grid = {'C':[0.1,0.2,0.5,1], 'kernel':['poly'], 'degree':[3,4,5] }
grid_ksvm_estimator = model_selection.GridSearchCV(ksvm_estimator, ksvm_grid, cv=10)
grid_ksvm_estimator.fit(X_train, y_train)
print(grid_ksvm_estimator.best_params_)
final_estimator = grid_ksvm_estimator.best_estimator_
print(final_estimator.dual_coef_)
print(final_estimator.intercept_)
print(final_estimator.n_support_)
print(final_estimator.support_)
print(grid_ksvm_estimator.best_score_)
print(final_estimator.score(X_train, y_train))

#kernel svm - rbf kernel
ksvm_estimator = svm.SVC(random_state=100)
ksvm_grid = {'C':[0.1,0.2,0.5,1], 'kernel':['rbf'] }
grid_ksvm_estimator = model_selection.GridSearchCV(ksvm_estimator, ksvm_grid, cv=10)
grid_ksvm_estimator.fit(X_train, y_train)
print(grid_ksvm_estimator.best_params_)
final_estimator = grid_ksvm_estimator.best_estimator_
print(final_estimator.dual_coef_)
print(final_estimator.intercept_)
print(final_estimator.n_support_)
print(final_estimator.support_)
print(grid_ksvm_estimator.best_score_)
print(final_estimator.score(X_train, y_train))

