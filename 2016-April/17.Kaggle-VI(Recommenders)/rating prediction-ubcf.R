library(recommenderlab)
library(ggplot2)
library(reshape2)

ls(pos="package:recommenderlab")

setwd("C:\\Users\\Thimma Reddy\\Documents\\data")
ratings_train = read.csv("train.csv")
dim(ratings_train)
str(ratings_train)
ratings_train = ratings_train[,-1]
dim(ratings_train)
str(ratings_train)

ratings_train1 = acast(ratings_train, user ~ movie)
dim(ratings_train1)
class(ratings_train1)
ratings_train1[1:5,1:5]

ratings_train2 = as(ratings_train1, "realRatingMatrix")
image(ratings_train2, main = "Raw Ratings") 

scheme = evaluationScheme(ratings_train2[1:100,], method="cross-validation", k=2, given=2)
results = evaluate(scheme, method="UBCF",type="ratings")
results
getConfusionMatrix(results)
avg(results)

recommender1 = Recommender(ratings_train2, method="UBCF", param=list(method="Cosine",nn=5))
recommender1

rownames(ratings_train2)
recom = predict(recommender1, ratings_train2[1:5,], type="ratings")
recom
class(recom)
recom = as(recom,"matrix")

ratings_test = read.csv("test.csv")
dim(ratings_test)
str(ratings_test)

ratings_test1 = ratings_test[ratings_test$user==1,]
recom

ratings = numeric()
for ( u in 1:nrow(ratings_test1)) {
  userid = ratings_test1[u,2]
  movieid = ratings_test1[u,3]
  ratings = c(ratings,recom[userid,movieid])
}
ratings = ifelse(is.na(ratings), 0, ratings)
result = cbind(ID=ratings_test1[,1],rating=round(ratings))
write.csv(result,file="submission.csv",row.names=FALSE)
