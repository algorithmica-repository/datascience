import sys
path = 'F://New Folder/utils'
sys.path.append(path)

import outlier_utils as outils
from sklearn import covariance, ensemble, svm, neighbors

n_samples = 400
outliers_fraction = 0.10
cluster_separation = 1
X, y = outils.generate_synthetic_data_outliers(n_samples, outliers_fraction, cluster_separation)
outils.plot_data_2d_outliers(X, xlim=[-7,7], ylim=[-7,7])

iso_forest_estimator = ensemble.IsolationForest(contamination=0.05)
iso_forest_estimator.fit(X)
print(iso_forest_estimator.decision_function(X))
print(iso_forest_estimator.predict(X))
outils.plot_model_2d_outliers(iso_forest_estimator, X)

lof_estimator = neighbors.LocalOutlierFactor(contamination=0.05)
lof_estimator.fit(X)
print(lof_estimator.negative_outlier_factor_)
outils.plot_model_2d_outliers(lof_estimator, X)

osvm_estimator = svm.OneClassSVM(nu= 0.85 * 0.05)
osvm_estimator.fit(X)
print(osvm_estimator.decision_function(X))
print(osvm_estimator.predict(X))
outils.plot_model_2d_outliers(osvm_estimator, X)

iso_forest_estimator = ensemble.IsolationForest()
iso_forest_grid = {'contamination':[0.01, 0.02, 0.05, 0.1]}
outils.grid_search_plot_models_outliers(iso_forest_estimator, iso_forest_grid, X, xlim=[-7,7], ylim=[-7,7])

cov_estimator = covariance.EllipticEnvelope()
cov_grid = {'contamination':[0.1, 0.2, 0.25, 0.3]}
outils.grid_search_plot_models_outliers(cov_estimator, cov_grid, X, xlim=[-7,7], ylim=[-7,7])

svm_estimator = svm.OneClassSVM(kernel="rbf", gamma=0.1)
tmp = 0.95 * outliers_fraction
svm_grid = {'nu':[tmp+0.03, tmp+0.05, tmp+0.06, tmp+0.07]}
outils.grid_search_plot_models_outliers(svm_estimator, svm_grid, X, xlim=[-7,7], ylim=[-7,7])
