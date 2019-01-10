from sklearn.datasets import make_classification
from sklearn import model_selection , metrics
import matplotlib.pyplot as plt
from sklearn import neighbors 
from sklearn.metrics import confusion_matrix, roc_curve, auc

def roc_binary(y_truth, y_pred):    
    fpr, tpr, _ = roc_curve(y_truth, y_pred)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange',
             lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()


X, y = make_classification(n_samples=1000, n_features=4, n_classes=2)

X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.25, random_state=1)

#using roc_auc for validation data
knn_classifier = neighbors.KNeighborsClassifier()
knn_grid = {'n_neighbors':[3,5,7,9,11], 'weights':['uniform','distance']}
grid_classifier = model_selection.GridSearchCV(knn_classifier, knn_grid, cv=10, scoring="roc_auc", return_train_score=True)
grid_classifier.fit(X_train, y_train)
print(grid_classifier.best_params_)
print(grid_classifier.best_score_)
print(grid_classifier.score(X_train, y_train))
final_estimator = grid_classifier.best_estimator_

#using roc_auc for test data
y_pred_probs = final_estimator.predict_proba(X_test)
metrics.roc_auc_score(y_test, y_pred_probs[:,1])
roc_binary(y_test, y_pred_probs[:,1])

#using accuracy for test data
y_pred_classes = final_estimator.predict(X_test)
metrics.confusion_matrix(y_test, y_pred_classes)
metrics.accuracy_score(y_test, y_pred_classes)
