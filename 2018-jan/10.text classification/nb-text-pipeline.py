import os
from sklearn.feature_extraction import text
from sklearn import naive_bayes, model_selection
from sklearn import pipeline
import pandas as pd
import re
from bs4 import BeautifulSoup

def preprocess_review(review):        #
        review_text = BeautifulSoup(review).get_text()
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        return review_text.lower()

os.chdir("E:/")
   
movie_train = pd.read_csv("TrainData.tsv", header=0, 
                    delimiter="\t", quoting=3)
movie_train.shape
movie_train.info()

movie_train1 = movie_train[0:1000]
X_train = movie_train1['review']
print(X_train.shape)
y_train = movie_train1['sentiment']
print(y_train.shape)

vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  stop_words = 'english')
tfidf_tr = text.TfidfTransformer()

nb_estimator = naive_bayes.MultinomialNB()

steps = [('cv', vectorizer), ('tfidf', tfidf_tr), ('nb', nb_estimator)]
nb_pipeline = pipeline.Pipeline(steps)

nb_pipeline_grid = {'cv__ngram_range':[(1,1),(2,2)], 'cv__max_features':[10,20],
            'tfidf__use_idf':[False, True],
            'nb__alpha':[0.01,0.5,1]}

grid_nb_pipeline_estimator = model_selection.GridSearchCV(nb_pipeline, nb_pipeline_grid, cv = 10, n_jobs=-1)
grid_nb_pipeline_estimator.fit(X_train, y_train)
print(grid_nb_pipeline_estimator.best_params_)
print(grid_nb_pipeline_estimator.best_score_)
