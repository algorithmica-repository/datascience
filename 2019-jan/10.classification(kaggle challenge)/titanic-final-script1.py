import os
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import preprocessing, tree, neighbors, metrics, linear_model, manifold, 
from sklearn_pandas import DataFrameMapper,CategoricalImputer
from sklearn import model_selection, ensemble, preprocessing, decomposition, feature_selection
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import validation_curve


def get_continuous_features(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_features(df):
    return df.select_dtypes(exclude=['number']).columns

def cast_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def get_categorical_imputers(df, features):    
    feature_defs = []
    for col_name in features:
        feature_defs.append((col_name, CategoricalImputer()))
    multi_imputer = DataFrameMapper(feature_defs, input_df=True, df_out=True)
    multi_imputer.fit(df[features])
    return multi_imputer

def apply_transformer(df, features, transformer):
    df[features] = transformer.transform(df[features])
    
def get_continuous_imputers(df, features):
    cont_imputer = preprocessing.Imputer()
    cont_imputer.fit(df[features])
    print(cont_imputer.statistics_)
    return cont_imputer

def get_features_to_drop_on_missingdata(df, threshold) :
    tmp = df.isnull().sum()
    return list(tmp[tmp/float(df.shape[0]) > threshold].index)

def drop_features(df, features):
    return df.drop(features, axis=1, inplace=False)

def ohe(df, features):
    return pd.get_dummies(df, columns = features)

def get_scaler(df):
    scaler = preprocessing.StandardScaler()
    scaler.fit(df)
    return scaler

def get_zero_variance_filter(X_train):
    tmp = feature_selection.VarianceThreshold()
    return tmp.fit()

def features_importances_from_model(selector, X, y):
    selector.fit(X, y)
    plot_feature_importances(selector, X, y)
    
def select_features(selector, X):
    tmp_model = feature_selection.SelectFromModel(selector, prefit=True, threshold="median")
    return tmp_model.transform(X)

def corr_heatmap(X):
    corr = np.corrcoef(X, rowvar=False)
    sns.heatmap(corr, annot=True, fmt='.2f')
  
def feature_reduction_pca(X, n_components):
    lpca = decomposition.PCA(n_components)
    pca_data = lpca.fit_transform(X)
    var = np.cumsum(np.round(lpca.explained_variance_ratio_, decimals=3)*100)
    plt.ylabel('% Variance Explained')
    plt.xlabel('# of Principal Components')
    plt.title('PCA Analysis')
    #plt.ylim(10,100.5)
    plt.style.context('seaborn-whitegrid')
    plt.plot(var)
    return pca_data

def feature_reduction_tsne(X, n_components=2):
    tsne = manifold.TSNE(n_components)
    tsne_data = tsne.fit_transform(X)
    return tsne_data

def plot_feature_importances(estimator, X_train, y_train):
    indices = np.argsort(estimator.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],x = estimator.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("Feature importances")   

def fit_model_objective(estimator, grid, X_train, y_train):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=10, return_train_score=True)
    grid_estimator.fit(X_train, y_train)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    print(final_model.coef_)
    print(final_model.intercept_)
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X_train, y_train))
    return final_model

def fit_model_neighbors(estimator, grid, X_train, y_train):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=10, return_train_score=True)
    grid_estimator.fit(X_train, y_train)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X_train, y_train))
    return final_model

def fit_model_tree(estimator, grid, X_train, y_train):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=10, return_train_score=True)
    grid_estimator.fit(X_train, y_train)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X_train, y_train))
    return final_model

def fit_model_ensemble(estimator, grid, X_train, y_train):
    grid_estimator = model_selection.GridSearchCV(estimator, grid, cv=10, return_train_score=True)
    grid_estimator.fit(X_train, y_train)
    print(grid_estimator.best_params_)
    final_model = grid_estimator.best_estimator_
    print(grid_estimator.best_score_)
    print(grid_estimator.score(X_train, y_train))
    return final_model

def plot_data_2d(X, y, labels=['X1', 'X2']):
    colors = ['red','green','purple','blue']
    plt.scatter(X[:,0], X[:,1], c=y, cmap=ListedColormap(colors), s=30)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    
def plot_data_3d(X, y, labels=['X1', 'X2','X3']):
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    colors = ['red','green','purple','blue']
    ax.scatter(X[:,0], X[:,1], X[:,2], c = y, cmap=ListedColormap(colors), s=30)
    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

def plot_parameter_impact_on_performance(estimator, data_features, data_target, param_name, param_range, scoring="accuracy"):
    # Calculate accuracy on training and test set using range of parameter values
    train_scores, test_scores = validation_curve(estimator, 
                                             data_features, 
                                             data_target, 
                                             param_name=param_name, 
                                             param_range=param_range,
                                             cv=10, 
                                             scoring=scoring)


    # Calculate mean and standard deviation for training set scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)

    # Calculate mean and standard deviation for test set scores
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)

    plt.style.use('seaborn')

    # Plot mean accuracy scores for training and test sets
    plt.plot(param_range, train_mean, label="Training score", color="black")
    plt.plot(param_range, test_mean, label="Cross-validation score", color="red")

    # Plot accurancy bands for training and test sets
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, color="gray")
    plt.fill_between(param_range, test_mean - test_std, test_mean + test_std, color="gainsboro")

    # Create plot
    plt.title("Parameter Values vs Performance:" +  str(estimator).split('(')[0] + ' model')
    plt.xlabel(param_name)
    plt.ylabel("Performance")
    plt.tight_layout()
    plt.legend(loc="best")
    plt.show()

#creation of dat frames from csv
path = 'C:\\Users\\Algorithmica\\Downloads'

titanic_train = pd.read_csv(os.path.join(path, 'titanic_train.csv'))
print(titanic_train.info())

#retrieve continuous & categorical features
get_continuous_features(titanic_train)
get_categorical_features(titanic_train)

#type cast features
features_to_cast = ['Pclass', 'Survived']
cast_cont_to_cat(titanic_train, features_to_cast)

#remove the features which have missing data more than 25%
missing_features = get_features_to_drop_on_missingdata(titanic_train, 0.25)
titanic_train = drop_features(titanic_train, missing_features)

#create title feature from name
def extract_title(name):
     return name.split(',')[1].split('.')[0].strip()
titanic_train['Title'] = titanic_train['Name'].map(extract_title)

#create family size feature from sibsp, parch
titanic_train['FamilySize'] = titanic_train['SibSp'] +  titanic_train['Parch'] + 1

#create family group feature from family-size
def convert_familysize(size):
    if(size == 1): 
        return 'Single'
    elif(size <=5): 
        return 'Medium'
    else: 
        return 'Large'
titanic_train['FamilyGroup'] = titanic_train['FamilySize'].map(convert_familysize)

continuous_features = ['Fare', 'Age']
cont_imputer = get_continuous_imputers(titanic_train, continuous_features)
titanic_train[continuous_features] = cont_imputer.transform(titanic_train[continuous_features])


cat_features = ['Embarked']
cat_imputer = get_categorical_imputers(titanic_train, cat_features)
titanic_train[cat_features] = cat_imputer.transform(titanic_train[cat_features])

titanic_train1 = drop_features(titanic_train, ['PassengerId', 'Name', 'Ticket', 'Survived'])

ohe_features = ['Sex', 'Pclass', 'Title', 'Embarked', 'FamilyGroup']
titanic_train1 = ohe(titanic_train1, ohe_features)

scaler = get_scaler(titanic_train1)
titanic_train2 = scaler.transform(titanic_train1)
titanic_train2 = pd.DataFrame(titanic_train2, columns =titanic_train1.columns)
y_train = titanic_train['Survived']

rf_selector = ensemble.RandomForestClassifier()
features_importances_from_model(rf_selector, titanic_train2, y_train)
titanic_train3 = select_features(rf_selector, titanic_train2)

corr_heatmap(titanic_train3)
feature_reduction_pca(titanic_train3, titanic_train3.shape[1])
tmp = feature_reduction_tsne(titanic_train3, 3)
plot_data_3d(tmp, y_train)

knn_estimator = neighbors.KNeighborsClassifier()
plot_parameter_impact_on_performance(knn_estimator, titanic_train3, y_train, 'n_neighbors', list(range(7,30)))

#read test data
titanic_test = pd.read_csv("C:\\Users\\Algorithmica\\Downloads\\titanic_test.csv")
print(titanic_test.info())

titanic_test[imputable_cont_features] = cont_imputer.transform(titanic_test[imputable_cont_features])
titanic_test['Embarked'] = cat_imputer.transform(titanic_test['Embarked'])
titanic_test['Embarked'] = le_embarked.transform(titanic_test['Embarked'])
titanic_test['Sex'] = le_sex.transform(titanic_test['Sex'])
titanic_test['Pclass'] = le_pclass.transform(titanic_test['Pclass'])
tmp1 = ohe.transform(titanic_test[categorical_features]).toarray()
tmp1 = pd.DataFrame(tmp1)
tmp2 = titanic_test[continuous_features]
tmp = pd.concat([tmp1, tmp2], axis=1)

X_test = scaler.transform(tmp)
X_test1 = lpca.transform(X_test)
titanic_test['Survived'] = final_estimator.predict(X_test1)
titanic_test.to_csv("C:\\Users\\Algorithmica\\Downloads\\submission.csv", columns=["PassengerId", "Survived"], index=False)