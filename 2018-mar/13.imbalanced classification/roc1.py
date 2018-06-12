#http://www.navan.name/roc/
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

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

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=1)

model = KNeighborsClassifier()
model.fit(X_train, y_train)

#traditional approach: cutoff = 0.5
y_predict = model.predict(X_test)
confusion_matrix(y_test, y_predict)

#better approach: ROC curver
y_predict_probabilities = model.predict_proba(X_test)[:,1]
roc_binary(y_test, y_predict_probabilities)
