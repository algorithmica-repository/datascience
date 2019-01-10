from surprise import Reader, Dataset, SVD
import os
from surprise.model_selection import GridSearchCV
import pandas as pd
import csv
   
def read_train_data(path):
    file_path = os.path.normpath(path)
    reader = Reader(line_format='timestamp user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    return data

movie_train = read_train_data('F:/train_v2.csv')
print(movie_train.raw_ratings)

param_grid = {'n_epochs': [50, 100,150], 'lr_all': [0.005],
              'reg_all': [0.4, 0.6]
             }
gs = GridSearchCV(SVD, param_grid, measures=['rmse'], cv=3)
gs.fit(movie_train)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
results_df = pd.DataFrame.from_dict(gs.cv_results)
algo = gs.best_estimator['rmse']
trainSet = movie_train.build_full_trainset()
algo.fit(trainSet)

