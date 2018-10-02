import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import numpy as np
from sklearn import neighbors
from sklearn import svm


def displayResults(inliers, outliers, classifier, outputTitle, outputName):
    plt.figure()        
    
    # Defining grid
    gridX, gridY = np.meshgrid(np.linspace(-0.5, 1.5, 1000), np.linspace(-0.5, 1.5, 1000))
    
    # Computing decision for each point of the grid
    gridDecisions = classifier.decision_function(np.c_[gridX.ravel(), gridY.ravel()])
    
    # Plotting decision boundary (each point of the grid whose decision value is 0)
    gridDecisions = gridDecisions.reshape(gridX.shape)
    plotBoundary = plt.contour(gridX, gridY, gridDecisions, levels=[0], linewidths=2, colors='blue')
    plt.clabel(plotBoundary, inline=1, fontsize=12)

    # Plotting inliers and outliers
    plt.scatter(inliers.loc[:, 'accommodates'], inliers.loc[:, 'price'], label="Inliers", color='green', alpha=0.2)
    plt.scatter(outliers.loc[:, 'accommodates'], outliers.loc[:, 'price'], label="Outliers", color='red', alpha=1.0)
    
    plt.xlabel("Accomodates")
    plt.ylabel("Price")
    plt.title(outputTitle)
    plt.legend()    
    plt.show()


X = pd.read_csv("E:/outlier.csv")
X = X.iloc[:,1:3]
print(X.info())
sns.jointplot(x="accommodates", y="price", data=X)

iso_forest_model = IsolationForest(n_estimators=100, contamination=0.01)
iso_forest_model.fit(X)
iso_forest_model.estimators_
outlier_scores = iso_forest_model.decision_function(X)
decisions_iso = iso_forest_model.predict(X)

displayResults(inliers=X[decisions_iso == 1],
               outliers=X[decisions_iso == -1],
               classifier=iso_forest_model,
               outputTitle = "detecting potential outliers using isolation forest",
               outputName="outliers_isolation_forest")

lof_model = neighbors.LocalOutlierFactor(n_neighbors=20, contamination=0.01)
decisions_lof = lof_model.fit_predict(X)
print(lof_model.negative_outlier_factor_)

displayResults(inliers=X[decisions_lof == 1],
               outliers=X[decisions_lof == -1],
               classifier=lof_model,
               outputTitle = "detecting potential outliers using LOF",
               outputName="outliers_LOF")


svm_model = svm.OneClassSVM(nu=0.1, gamma=4.0)
svm_model.fit(X)
decisions_svm = svm_model.decision_function(X)

displayResults(inliers=X[decisions_svm >= 0],
               outliers=X[decisions_svm < 0],
               classifier=svm_model ,
               outputTitle = "detecting potential outliers using one class svm",
               outputName="outliers_oneclass_svm")

