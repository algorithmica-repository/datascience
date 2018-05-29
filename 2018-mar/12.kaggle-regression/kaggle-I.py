import os
import pandas as pd
from sklearn import preprocessing
from sklearn_pandas import DataFrameMapper,CategoricalImputer
import numpy as np
from sklearn import model_selection, ensemble
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import feature_selection

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

def plot_feature_importances(X_train, y_train):
    classifier = ensemble.RandomForestRegressor(random_state=100, n_jobs=1, verbose=1, oob_score=True)
    classifier.fit(X_train, y_train)
    indices = np.argsort(classifier.feature_importances_)[::-1][:40]
    g = sns.barplot(y=X_train.columns[indices][:40],x = classifier.feature_importances_[indices][:40] , orient='h')
    g.set_xlabel("Relative importance",fontsize=12)
    g.set_ylabel("Features",fontsize=12)
    g.tick_params(labelsize=9)
    g.set_title("RF feature importances")
    return classifier
      
def map_cat_to_numerics(df, features, mappings):
    for feature in features:
        null_idx = df[feature].isnull()
        df.loc[null_idx, feature] = None 
        df[feature] = df[feature].map(mappings)
        
def get_continuous_columns(df):
    return df.select_dtypes(include=['number']).columns

def get_categorical_columns(df):
    return df.select_dtypes(exclude=['number']).columns

def transform_cat_to_cont(df, cat_features, cont_features):
    feature_defs = []
    for col_name in cat_features:
        feature_defs.append((col_name, MyLabelBinarizer()))
    
    for col_name in cont_features:
        feature_defs.append((col_name, None))

    mapper = DataFrameMapper(feature_defs, input_df=True, df_out=True)
    mapper.fit(df)
    return mapper.transform(df)

def cast_cont_to_cat(df, features):
    for feature in features:
        df[feature] = df[feature].astype('category')

def impute_categorical_features(df, features):
    #impute missing values for categorical features
    cat_imputer = CategoricalImputer()
    cat_imputer.fit(df[features])
    print(cat_imputer.fill_)
    df[features] = cat_imputer.transform(df[features])

def impute_continuous_features(df, features):
    cont_imputer = preprocessing.Imputer()
    cont_imputer.fit(df[features])
    print(cont_imputer.statistics_)
    df[features] = cont_imputer.transform(df[features])

def get_features_to_drop(df, threshold) :
    tmp = df.isnull().sum()
    return list(tmp[tmp/float(df.shape[0]) > threshold].index)

def drop_features(df, features):
    return df.drop(features, axis=1, inplace=False)
    
def select_features(X_train, y_train):
    return


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

#preprocess features
features_to_cast = ['MSSubClass']
cast_cont_to_cat(house_all, features_to_cast)

house_train = house_all[0:house_train.shape[0]].copy()
house_train.info()
features_to_drop = get_features_to_drop(house_train, 0.0)
features_to_drop.append('SalePrice')
features_to_drop.append('Id')

house_train1 = drop_features(house_train, features_to_drop)
house_train1.info()

cat_features = get_categorical_columns(house_train1)
cont_features = get_continuous_columns(house_train1)
print(cat_features)
print(cont_features)

house_train1 = transform_cat_to_cont(house_train1, cat_features, cont_features)
rf = plot_feature_importances(house_train1, house_train['SalePrice'])
fs_model = feature_selection.SelectFromModel(rf, prefit=True)
house_train2 = fs_model.transform(house_train1)

#explore sale price
sns.kdeplot(house_train['SalePrice'])
house_train['log_sale_price'] = np.log(house_train['SalePrice'])
sns.kdeplot(house_train['log_sale_price'])

#explore mszoing
sns.countplot('MSZoning', data = house_train)
sns.FacetGrid(house_train, hue="MSZoning",size=8).map(sns.kdeplot, "SalePrice").add_legend()
sns.FacetGrid(house_train, row="MSZoning",size=8).map(sns.kdeplot, "SalePrice").add_legend()

#explore neighborhood
sns.countplot('Neighborhood', data = house_train).set_xticklabels(rotation=90)
sns.FacetGrid(house_train, hue="Neighborhood",size=8).map(sns.kdeplot, "SalePrice").add_legend()

#explore OverallQual column
sns.countplot('OverallQual', data = house_train)
sns.FacetGrid(house_train, hue="OverallQual",size=8).map(sns.kdeplot, "SalePrice").add_legend()
sns.jointplot(x="OverallQual", y="SalePrice", data=house_train)


#explore ordered categorical features
sns.countplot('ExterQual', data = house_train)
sns.FacetGrid(house_train, hue="ExterQual",size=8).map(sns.kdeplot, "SalePrice").add_legend()
sns.countplot('GarageQual', data = house_train)
sns.FacetGrid(house_train, hue="GarageQual",size=8).map(sns.kdeplot, "SalePrice").add_legend()
ordinal_features1 = ["ExterQual", "ExterCond", "BsmtQual", "BsmtCond", "GarageQual", "GarageCond", "PoolQC", "FireplaceQu", "KitchenQual", "HeatingQC"]
quality_dict1 = {None: 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5 }
transform_cat_to_cont(house_train, ordinal_features1, quality_dict1)
sns.jointplot(x="ExterQual", y="SalePrice", data=house_train)


sns.countplot('BsmtExposure', data = house_train)
sns.FacetGrid(house_train, hue="BsmtExposure",size=8).map(sns.kdeplot, "SalePrice").add_legend()
ordinal_features2 = ["BsmtExposure"]
quality_dict2 = {None: 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4 }
transform_cat_to_cont(house_train, ordinal_features2, quality_dict2)

#explore central air conditioning
sns.countplot('CentralAir', data = house_train)
sns.FacetGrid(house_train, hue="CentralAir",size=8).map(sns.kdeplot, "SalePrice").add_legend()

#explore 1stfloor sqft
sns.kdeplot(house_train['1stFlrSF'])
sns.jointplot(x="1stFlrSF", y="SalePrice", data=house_train)

#explore bedroom feature
sns.countplot('BedroomAbvGr', data = house_train)
sns.FacetGrid(house_train, hue="BedroomAbvGr",size=8).map(sns.kdeplot, "SalePrice").add_legend()
sns.jointplot(x="BedroomAbvGr", y="SalePrice", data=house_train)

#explore garage year built
sns.countplot('YearBuilt', data = house_train)
sns.FacetGrid(house_train, hue="YearBuilt",size=8).map(sns.kdeplot, "SalePrice").add_legend()
sns.jointplot(x="YearBuilt", y="SalePrice", data=house_train)


house_test.to_csv(os.path.join(path,'submission.csv'), columns=['PassengerId','Survived'], index=False)
