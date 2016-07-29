library(recommenderlab)
library(ggplot2)
library(reshape2)

ls(pos="package:recommenderlab")

setwd("")
ratings_train = read.csv("train.csv")
ratings_train = ratings_train[,-c(1)]
ratings_train1 = acast(ratings_train, user ~ movie)
class(g)
ratings_train1 = as.matrix(ratings_train1)
#realRatingMatrix is a recommenderlab sparse-matrix like data-structure
ratings_train2 = as(ratings_train1, "realRatingMatrix")
image(ratings_train2, main = "Raw Ratings") 

recommender1 = Recommender(ratings_train2[1:nrow(ratings_train2)], method="UBCF", param=list(method="Cosine",nn=5))
recommender2 = Recommender(ratings_train2[1:nrow(ratings_train2)], method="IBCF", param=list(method="Jaccard"))
recommender3 = Recommender(ratings_train2[1:nrow(ratings_train2)], method="POPULAR")

print(recommender2)
names(getModel(recommender2))
getModel(recommender2)$nn

recom = predict(recommender2, ratings2[1:nrow(ratings2)], type="ratings")
recom

ratings_test = read.csv("test.csv")
rec_list = as(recom,"list")
summary(rec_list)

ratings = NULL
# For all lines in test file, one by one
for ( u in 1:length(ratings_test[,2]))
{
  # Read userid and movieid from columns 2 and 3 of test data
  userid = ratings_test[u,2]
  movieid = ratings_test[u,3]
  
  # Get as list & then convert to data frame all recommendations for user: userid
  u1 = as.data.frame(rec_list[[userid]])
  
  # Create a (second column) column-id in the data-frame u1 and populate it with row-names
  # Remember (or check) that rownames of u1 contain are by movie-ids
  # We use row.names() function
  u1$id = row.names(u1)
  
  # Now access movie ratings in column 1 of u1
  x= u1[u1$id==movieid,1]
  # print(u)
  # print(length(x))
  # If no ratings were found, assign 0. You could also
  #   assign user-average
  if (length(x)==0) {
    ratings[u] <- 0
  }
  else {
    ratings[u] <-x
  }
  
}
length(ratings)
tx = cbind(ratings_test[,1],round(ratings))

# Write to a csv file: submitfile.csv in your folder
write.table(tx,file="submitfile.csv",row.names=FALSE,col.names=FALSE,sep=',')
