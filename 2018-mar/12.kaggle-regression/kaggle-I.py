import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper,CategoricalImputer
import numpy as np
from sklearn import model_selection, ensemble
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.externals import joblib

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

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = model_selection.learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

def plot_feature_importances(classifier, X_train, name):
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title(name + " feature importance")
        
#changes working directory
path = 'D:/'

house_train = pd.read_csv(os.path.join(path,"house-train.csv"))
house_train.shape
house_train.info()

house_test = pd.read_csv(os.path.join(path,"house-test.csv"))
house_test.shape
house_test.info()

house_all = pd.concat([house_train, house_test])
house_all.shape

house_train = house_all[0:house_train.shape[0]].copy()

sns.kdeplot(house_train['SalePrice'])
house_train['log_sale_price'] = np.log(house_train['SalePrice'])
sns.kdeplot(house_train['log_sale_price'])

sns.countplot('MSZoning', data = house_train)
sns.FacetGrid(house_train, hue="MSZoning",size=8).map(sns.kdeplot, "SalePrice").add_legend()
sns.FacetGrid(house_train, row="MSZoning",size=8).map(sns.kdeplot, "SalePrice").add_legend()


#explore cabin column
sns.countplot('Cabin', data = titanic_train)
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "Fare").add_legend()

#explore ticket column
sns.countplot('Ticket', data = titanic_train)
sns.factorplot(x='Ticket', hue='Survived', data=titanic_train,  kind="count", size=6).set_xticklabels(rotation=90)

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

#size of families (including the passenger)
sns.FacetGrid(titanic_train, hue="Survived",size=8).map(sns.kdeplot, "FamilySize").add_legend()

cat_features = ['Sex', 'Embarked', 'Pclass', 'Cabin', 'Title', 'Ticket']
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
rf_classifier = ensemble.RandomForestClassifier(random_state=100, n_jobs=1, verbose=1, oob_score=True)
rf_grid = {'max_depth':list(range(7,14)), 'n_estimators':list(range(10,100,10)),  
           'min_samples_split':list(range(4,11)), 'min_samples_leaf':list(range(2,5))}
rf_grid_classifier = model_selection.GridSearchCV(rf_classifier, rf_grid, cv=kfold, refit=True, return_train_score=True)
rf_grid_classifier.fit(X_train, y_train)

#check overfitting due to parameters(i.e.,capacity of the model)
results = rf_grid_classifier.cv_results_
print(results.get('mean_test_score'))
print(results.get('mean_train_score'))
print(rf_grid_classifier.best_params_)
print(rf_grid_classifier.best_score_)
final_model = rf_grid_classifier.best_estimator_

#check overfitting due to size of the data
plot_learning_curve(final_model,"RF learning curves",X_train,y_train,cv=kfold)

#plot the feature importances
plot_feature_importances(final_model, X_train, "Random Forest")

#persist the model
joblib.dump(final_model, 'final_model.pkl')

titanic_test = titanic_all[titanic_train.shape[0]:]
titanic_test['Age'] = grouped.Age.apply(lambda x: x.fillna(x.median()))
titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
X_test = mapper.transform(titanic_test)

titanic_test['Survived'] = final_model.predict(X_test)
titanic_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)
