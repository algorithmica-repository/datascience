import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

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
    
    #plt.savefig(outputName + ".png")
    #plt.clf()
    
listings = pd.read_csv("E:/outlier.csv")
listings.shape
listings = listings[['accommodates','price']]
listings.info()

svm_model = svm.OneClassSVM(nu=0.1, gamma=4.0)
svm_model .fit(listings)
decisions_svm = svm_model.decision_function(listings)

displayResults(inliers=listings[decisions_svm >= 0],
               outliers=listings[decisions_svm < 0],
               classifier=svm_model ,
               outputTitle = "detecting potential outliers using one class svm",
               outputName="outliers_oneclass_svm")

iso_forest_model = IsolationForest(n_estimators=10)
param_grid = {'n_estimators': [10,20]}
grid = GridSearchCV(iso_forest_model, param_grid, cv=10, scoring = "roc_auc")
iso_forest_model.fit(listings)

decisions_if = iso_forest_model.predict(listings)

displayResults(inliers=listings[decisions_if == 1],
               outliers=listings[decisions_if == -1],
               classifier=iso_forest_model,
               outputTitle = "detecting potential outliers using isolation forest",
               outputName="outliers_isolation_forest")
               