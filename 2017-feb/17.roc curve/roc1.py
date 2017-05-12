import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
y_pred_prob = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, y_pred_prob, pos_label=2)
fpr
tpr
thresholds