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
print(movie_train.loc[0:1,'review'])

#text cleaning example for one sample review
review_tmp = movie_train['review'][0]
review_tmp = BeautifulSoup(review_tmp).get_text()
review_tmp = re.sub("[^a-zA-Z]"," ", review_tmp)
review_tmp = review_tmp.lower()

def preprocess_review(review):        #
        # 1. Remove HTML
        review_text = BeautifulSoup(review).get_text()
        #
        # 2. Remove non-letters
        review_text = re.sub("[^a-zA-Z]"," ", review_text)
        #
        # 3. Convert words to lower case
        return review_text.lower()


#extract unigram  features  
vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  stop_words = 'english', \
                                  max_features = 10)
#fit method builds mappings between terms and indexes
#the terms are ordered based on frequency 
#transform the reviews to count vectors(dtm)
vectorizer.fit(movie_train.loc[0:3,'review'])
#get the mapping between the term features and dtm column index
print(vectorizer.vocabulary_)
#get the feature names
print(vectorizer.get_feature_names())
review_vectors = vectorizer.transform(movie_train.loc[0:3,'review']).toarray()

#extract bigram  features  
vectorizer = text.CountVectorizer(preprocessor = preprocess_review, \
                                  stop_words = 'english', \
                                  ngram_range = (1,3), \
                                  max_features = 100)
#fit method builds mappings between terms and indexes
#the terms are ordered based on frequency 
#transform the reviews to count vectors(dtm)
vectorizer.fit(movie_train.loc[0:3,'review'])
#get the mapping between the term features and dtm column index
print(vectorizer.vocabulary_)
#get the feature names
print(vectorizer.get_feature_names())
review_vectors = vectorizer.transform(movie_train.loc[0:3,'review']).toarray()
