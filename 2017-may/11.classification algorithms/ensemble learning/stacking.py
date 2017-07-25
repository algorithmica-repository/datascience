import time
import numpy as np
from sklearn.cross_validation import KFold
from sklearn import model_selection

class StackEnsemble(object):
    def __init__(self, n_folds, base_models, stacker, stacker_grid):
        self.n_folds = n_folds
        self.stacker = stacker
        self.base_models = base_models
        self.stacker_grid = stacker_grid

    def fit(self, X, y):
        start_time = time.time()
        X = np.array(X)
        y = np.array(y)
        folds = list(KFold(len(y), n_folds=self.n_folds, shuffle=True, random_state=2017))
        S_train = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):

            print('Fitting For Base Model #%d / %d ---', i+1, len(self.base_models))
            for j, (train_idx, test_idx) in enumerate(folds):

                print('--- Fitting For Fold %d / %d ---', j+1, self.n_folds)

                X_train = X[train_idx]
                y_train = y[train_idx]
                X_holdout = X[test_idx]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_holdout)[:]
                S_train[test_idx, i] = y_pred

                print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

            print('Elapsed: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        print('--- Base Models Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

        grid = model_selection.GridSearchCV(estimator=self.stacker, param_grid= self.stacker_grid, n_jobs=1, cv=5)
        grid.fit(S_train, y)
        print(grid.grid_scores_)
        print('Best CV Score:')
        print(grid.best_score_)

        print('--- Stacker Trained: %s minutes ---' % round(((time.time() - start_time) / 60), 2))

    def predict(self, X):
        X = np.array(X)
        folds = list(KFold(len(X), n_folds=self.n_folds, shuffle=True, random_state=2017))
        S_test = np.zeros((X.shape[0], len(self.base_models)))

        for i, clf in enumerate(self.base_models):
            S_test_i = np.zeros((X.shape[0], len(folds)))
            for j, (train_idx, test_idx) in enumerate(folds):
                S_test_i[:, j] = clf.predict(X)[:]
            S_test[:, i] = S_test_i.mean(1)

        clf = self.stacker
        y_pred = clf.predict(S_test)[:]
        return y_pred
