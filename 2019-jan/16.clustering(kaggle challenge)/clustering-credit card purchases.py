import sys
import os
path = os.path.abspath(os.path.join('.'))
sys.path.append(path)
path = 'E://'
sys.path.append(path)

from common_utils  import *
from clustering_utils import *
from classification_utils import *
from feature_reduction_utils import *
import pandas as pd
import numpy as np
from sklearn import preprocessing
 
purchase_data = pd.read_csv(os.path.join(path,'creditcard_purchases.csv'))
purchase_data.info()
purchase_data = drop_features(purchase_data, ['CUST_ID'])

g = sns.heatmap(purchase_data.isnull(), yticklabels=False, cbar=False, cmap='viridis');
g.set_xticklabels(g.get_xticklabels(), rotation=90)

imputable_cont_features = get_continuous_features(purchase_data)
cont_imputer = get_continuous_imputers(purchase_data, imputable_cont_features)
purchase_data[imputable_cont_features] = cont_imputer.transform(purchase_data[imputable_cont_features])
purchase_data.info()

fig = plt.figure()
for i in range(purchase_data1.shape[1]):
    sns.kdeplot(purchase_data1.iloc[:, i])

m_scaler = preprocessing.StandardScaler()
purchase_data1 = m_scaler.fit_transform(purchase_data)
purchase_data1 = pd.DataFrame(purchase_data1, columns = purchase_data.columns)

lpca_data = feature_reduction_linear_pca(purchase_data1, 8)
tsne_data = feature_reduction_tsne(lpca_data)
plot_data_2d(tsne_data)
labels = ['PC'+str(i)  for i in range(1,9)]
lpca_data = pd.DataFrame(lpca_data, columns = labels)

scoring = 's_score'
kmeans_estimator = cluster.KMeans()
kmeans_grid = {'n_clusters':list(range(2,7))}
#grid_search_plot_models_clustering(kmeans_estimator, kmeans_grid, X)
kmeans_final_model = grid_search_best_model_clustering(kmeans_estimator, kmeans_grid, lpca_data, scoring=scoring)
#plot_model_2d_kmeans(kmeans_final_model, X)

lpca_data['cluster'] = kmeans_final_model.predict(lpca_data)

sns.countplot(x='cluster', data=lpca_data)
sns.pairplot(data=lpca_data.iloc[:,5:], hue='cluster')
