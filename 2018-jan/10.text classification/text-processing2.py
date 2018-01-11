import os
from sklearn.feature_extraction import text
import pandas as pd
import re
from bs4 import BeautifulSoup

os.chdir("E:/")
#os.chdir('/home/algo/Downloads')
    
movie_train = pd.read_csv("TrainData.tsv", header=0, 
                    delimiter="\t", quoting=3)
movie_train.shape
movie_train.info()

def preprocess_review(review):        #
        review_text = BeautifulSoup(review).get_text()
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        return review_text.lower()

#build dtm with unigram features and tf
vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  stop_words = 'english', \
                                  max_features = 10)
review_vectors = vectorizer.fit_transform(movie_train.loc[0:3,'review']).toarray()

tf_transformer = text.TfidfTransformer(use_idf=False)
review_vectors1 = tf_transformer.fit_transform(review_vectors).toarray()

#build dtm with unigram features and tf*idf
tfidf_transformer = text.TfidfTransformer()
review_vectors1 = tfidf_transformer.fit_transform(review_vectors).toarray()
