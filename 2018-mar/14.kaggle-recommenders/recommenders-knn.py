#https://www.visualstudio.com/downloads/#build-tools-for-visual-studio-2017

from surprise import KNNBasic
from surprise import Reader, Dataset
import os
from surprise.model_selection import GridSearchCV
import pandas as pd
import csv
   
def read_train_data(path):
    file_path = os.path.normpath(path)
    reader = Reader(line_format='timestamp user item rating', sep=',')
    data = Dataset.load_from_file(file_path, reader=reader)
    return data

movie_train = read_train_data('E:/train_v2.csv')
print(movie_train.raw_ratings)

param_grid = {'k': [10, 20],
              'sim_options': {'name': ['cosine'],
                              'min_support': [1, 5],
                              'user_based': [False]}
              }
gs = GridSearchCV(KNNBasic, param_grid, measures=['rmse'], cv=3)
gs.fit(movie_train)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
results_df = pd.DataFrame.from_dict(gs.cv_results)
algo = gs.best_estimator['rmse']
trainSet = movie_train.build_full_trainset()
algo.fit(trainSet)

rows = csv.reader(open('F:/test_v2.csv'))
rows = list(rows)
rows.pop(0)
f = open('F:/submission.csv', 'w',newline='')
writer = csv.writer(f)
writer.writerow(['ID', 'rating'])
count = 0
for row in rows:
  count = count + 1
  pred = algo.predict(row[1], row[2])
  print(pred)
  writer.writerow([row[0], pred[3]])
f.close()
print(count)