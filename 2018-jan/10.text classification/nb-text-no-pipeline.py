import os
from sklearn.feature_extraction import text
from sklearn import naive_bayes, model_selection
import pandas as pd
import re
from bs4 import BeautifulSoup

def preprocess_review(review):        #
        review_text = BeautifulSoup(review).get_text()
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        return review_text.lower()

os.chdir("E:/")
#os.chdir('/home/algo/Downloads')
    
movie_train = pd.read_csv("TrainData.tsv", header=0, 
                    delimiter="\t", quoting=3)
movie_train.shape
movie_train.info()

movie_train1 = movie_train[0:1000]
#build dtm with unigram features and tf
vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  stop_words = 'english', \
                                  max_features = 10)
review_vectors = vectorizer.fit_transform(movie_train1['review']).toarray()
print(review_vectors.shape)

tfidf_transformer = text.TfidfTransformer()
X_train = tfidf_transformer.fit_transform(review_vectors).toarray()
y_train = movie_train1['sentiment']

nb_estimator = naive_bayes.MultinomialNB()
nb_grid = {'alpha':[0.01,0.5,1]}
grid_nb_estimator = model_selection.GridSearchCV(nb_estimator, nb_grid, cv = 4)
grid_nb_estimator.fit(X_train, y_train)

print(grid_nb_estimator.best_score_)
