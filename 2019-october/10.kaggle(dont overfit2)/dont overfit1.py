import pandas as pd
import os
from sklearn import tree, ensemble, model_selection, preprocessing, decomposition, manifold, feature_selection, svm
import seaborn as sns
import numpy as np

import sys
sys.path.append("E:/New Folder/utils")

import classification_utils as cutils

dir = 'C:/Users/Algorithmica/Downloads/dont-overfit-ii'
train = pd.read_csv(os.path.join(dir, 'train.csv'))
print(train.info())
print(train.columns)

#filter unique value features
train1 = train.iloc[:,2:] 
y = train['target'].astype(int)

#filter zero-variance features
variance = feature_selection.VarianceThreshold()
train2 = variance.fit_transform(train1)

lpca = decomposition.PCA(n_components=0.95)
lpca.fit(train2)
np.cumsum(lpca.explained_variance_ratio_)
train_pca = lpca.transform(train2)

tsne = manifold.TSNE(n_components=3)
train_tsne = tsne.fit_transform(train_pca)
cutils.plot_data_3d_classification(train_tsne, y)

X_train, X_eval, y_train, y_eval = model_selection.train_test_split(train_pca, y, test_size=0.1, random_state=1)

sns.countplot(x='target',data=train)

kernel_svm_estimator = svm.SVC(kernel='rbf')
kernel_svm_grid = {'gamma':[0.01, 0.1, 1, 2, 5, 10], 'C':[0.001, 0.01, 0.1, 0.5] }
final_estimator = cutils.grid_search_best_model(kernel_svm_estimator, kernel_svm_grid, X_train, y_train)

print(final_estimator.score(X_eval, y_eval))

test = pd.read_csv(os.path.join(dir, 'test.csv'))
print(test.info())
print(test.columns)

test1 = test.iloc[:,1:] 
test2 = variance.transform(test1)
test_pca = lpca.transform(test2)
test['target'] = final_estimator.predict(test_pca)
test.to_csv(os.path.join(dir, 'submission.csv'), columns=['id', 'target'], index=False)
