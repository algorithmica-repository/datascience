from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

def roc_binary_for_models(y_truth, knn_pred, lr_pred):    
    knn_fpr, knn_tpr, _ = roc_curve(y_truth, knn_pred)
    knn_roc_auc = auc(knn_fpr, knn_tpr)
    
    lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_pred)
    lr_roc_auc = auc(lr_fpr, lr_tpr)

    plt.figure()
    plt.plot(knn_fpr, knn_tpr, color='darkorange',
         lw=2, label='K-nearest Neighbours (area = %0.2f)' % knn_roc_auc)
    plt.plot(lr_fpr, lr_tpr, color='darkgreen',
         lw=2, label='Logistic Regression (area = %0.2f)' % lr_roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()
    
X, y = make_classification(n_samples=1000, n_features=4, n_classes=2)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

model = KNeighborsClassifier()
model.fit(X_train, y_train)
knn_predict_probabilities = model.predict_proba(X_test)[:,1]

lr_model = LogisticRegression()
lr_model.fit(X_train, y_train)
lr_predict_probabilities = lr_model.predict_proba(X_test)[:,1]

roc_binary_for_models(y_test, knn_predict_probabilities, lr_predict_probabilities)