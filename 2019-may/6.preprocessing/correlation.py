import sys
sys.path.append("E:/")

import classification_utils as cutils
import numpy as np

#2-d classification pattern
X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.5, 0.5])

np.cov(X, rowvar=False)
np.corrcoef(X, rowvar=False)

X, y = cutils.generate_linear_synthetic_data_classification(n_samples=1000, n_features=10, n_classes=2, weights=[0.5, 0.5], n_redundant=4)

np.cov(X, rowvar=False)
np.corrcoef(X, rowvar=False)
