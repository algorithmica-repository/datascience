import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import seaborn as sns

def viz(df, scaled_df1, scaled_df2, scaled_df3):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(12, 5))

    ax1.set_title('Before Scaling')
    sns.kdeplot(df['x1'], ax=ax1)
    sns.kdeplot(df['x2'], ax=ax1)
    sns.kdeplot(df['x3'], ax=ax1)

    ax2.set_title('After Standard Scaler')
    sns.kdeplot(scaled_df1['x1'], ax=ax2)
    sns.kdeplot(scaled_df1['x2'], ax=ax2)
    sns.kdeplot(scaled_df1['x3'], ax=ax2)
    
    ax3.set_title('After Min-Max Scaler')
    sns.kdeplot(scaled_df2['x1'], ax=ax3)
    sns.kdeplot(scaled_df2['x2'], ax=ax3)
    sns.kdeplot(scaled_df2['x3'], ax=ax3)
    
    ax4.set_title('After Robust Scaler')
    sns.kdeplot(scaled_df3['x1'], ax=ax4)
    sns.kdeplot(scaled_df3['x2'], ax=ax4)
    sns.kdeplot(scaled_df3['x3'], ax=ax4)

    plt.show()

def get_guassian_data():
    return pd.DataFrame({
            'x1': np.random.normal(0, 2, 10000),
            'x2': np.random.normal(5, 3, 10000),
            'x3': np.random.normal(-5, 5, 10000)
            })

def get_mixed_data1():
    return pd.DataFrame({
            # positive skew
            'x1': np.random.chisquare(8, 1000),
            # negative skew 
            'x2': np.random.beta(8, 2, 1000) * 40,
            # no skew
            'x3': np.random.normal(50, 3, 1000)
            })

def get_mixed_data2():
    return pd.DataFrame({
            # Distribution with lower outliers
            'x1': np.concatenate([np.random.normal(20, 1, 1000), np.random.normal(1, 1, 25)]),
            # Distribution with higher outliers
            'x2': np.concatenate([np.random.normal(30, 1, 1000), np.random.normal(50, 1, 25)]),
             # Distribution with no outliers
            'x3': np.concatenate([np.random.normal(20, 1, 1000), np.random.normal(20, 1, 25)]),
            })
    

def get_transformed_data(df):
    scaler1 = preprocessing.StandardScaler()
    scaled_df1 = scaler1.fit_transform(df)
    scaled_df1 = pd.DataFrame(scaled_df1, columns=['x1', 'x2', 'x3'])
    
    scaler2 = preprocessing.MinMaxScaler()
    scaled_df2 = scaler2.fit_transform(df)
    scaled_df2 = pd.DataFrame(scaled_df2, columns=['x1', 'x2', 'x3'])
    
    scaler3 = preprocessing.RobustScaler()
    scaled_df3 = scaler3.fit_transform(df)
    scaled_df3 = pd.DataFrame(scaled_df3, columns=['x1', 'x2', 'x3'])
    
    return scaled_df1, scaled_df2, scaled_df3

np.random.seed(1)

df = get_guassian_data()
scaled_df1, scaled_df2, scaled_df3 = get_transformed_data(df)
viz(df, scaled_df1, scaled_df2, scaled_df3)

df = get_mixed_data1()
scaled_df1, scaled_df2, scaled_df3 = get_transformed_data(df)
viz(df, scaled_df1, scaled_df2, scaled_df3)

df = get_mixed_data2()
scaled_df1, scaled_df2, scaled_df3 = get_transformed_data(df)
viz(df, scaled_df1, scaled_df2, scaled_df3)