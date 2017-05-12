library(recommenderlab)
library(ggplot2)
library(reshape2)

ls(pos="package:recommenderlab")

setwd("C:\\Users\\Algorithmica\\Downloads")
ratings_train = read.csv("train_v2.csv")
dim(ratings_train)
str(ratings_train)
ratings_train = ratings_train[,-1]
dim(ratings_train)
str(ratings_train)

ratings_train1 = acast(ratings_train, user ~ movie)
dim(ratings_train1)
class(ratings_train1)
ratings_train1[2,1:100]

ratings_train2 = as(ratings_train1, "realRatingMatrix")
dim(ratings_train2)
class(ratings_train2)
image(ratings_train2, main = "Raw Ratings") 

#evaluate the recommender algorithm using cross validation scheme. Final model is not built by evaluate
topn_scheme = evaluationScheme(ratings_train2, method="cross-validation", k=10, given=2, goodRating=2.5)
topn_random_results = evaluate(topn_scheme, method="RANDOM",type="topNList")
getConfusionMatrix(topn_random_results)
avg(topn_random_results)

#evaluate the recommender algorithm using cross validation scheme. Final model is not built by evaluate
rating_scheme = evaluationScheme(ratings_train2, method="cross-validation", k=10, given=2)
rating_random_results = evaluate(rating_scheme, method="RANDOM",type="ratings")
getConfusionMatrix(rating_random_results)
avg(rating_random_results)

??UBCF
#building model on entire train data
ubcf_model=Recommender(ratings_train2,method="UBCF", param=list(method="Cosine",nn=4, minRating=1))
getModel(ubcf.model)

recom_topn = predict(random_model, ratings_train2, type="topNList", n=3)
recom_topn_list = as(recom_topn, "list")
recom_topn_list[[3]]

recom_ratings = predict(ubcf_model, ratings_train2, type="ratings")
class(recom_ratings)
recom_ratings_matrix = as(recom_ratings,"matrix")
recom_ratings_matrix[1:3,1:10]
