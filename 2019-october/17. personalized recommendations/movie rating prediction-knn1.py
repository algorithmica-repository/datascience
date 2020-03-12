from surprise import KNNBasic, Reader, Dataset, model_selection
from collections import defaultdict


def get_top_n(predictions, n=10):
    # First map the predictions to each user.
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_n[uid].append((iid, est))

    # Then sort the predictions for each user and retrieve the k highest ones.
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n

movie_train = Dataset.load_builtin('ml-100k')
print(movie_train.raw_ratings)

knn_estimator = KNNBasic
knn_grid = {'k': [10, 20],
              'sim_options': {'name': ['cosine', 'msd'],
                              'min_support': [1, 5],
                              'user_based': [True, False]
                              }
              }
gs = model_selection.GridSearchCV(knn_estimator, knn_grid, measures=['rmse'], cv=3)
gs.fit(movie_train)

print(gs.best_score['rmse'])
print(gs.best_params['rmse'])
final_estimator = gs.best_estimator['rmse']
#build final model on entire train data
movie_train = movie_train.build_full_trainset()
final_estimator.fit(movie_train)

movie_test= movie_train.build_anti_testset()
predictions = final_estimator.test(movie_test)

top_n = get_top_n(predictions, n=3)

# Print the recommended items for each user
for uid, user_ratings in top_n.items():
    print(uid, [iid for (iid, _) in user_ratings])