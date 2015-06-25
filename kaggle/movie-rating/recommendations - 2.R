library(recommenderlab)
library(ggplot2)
library(reshape2)

ls(pos="package:recommenderlab")

setwd("E:/data analytics/kaggle/movie-rating")
data = read.csv("train_v2.csv", header = TRUE)
str(data)
head(data)
data = data[,-c(1)]
user_movie=acast(data, user~movie,value.var='rating')
class(user_movie)
dim(user_movie)
str(user_movie)
user_movie[1:2,1:2]

user_movie_rating = as(user_movie,"realRatingMatrix")
class(user_movie_rating)
str(user_movie_rating)
rowMeans(user_movie_rating[1,])
user_movie_rating_norm1 = normalize(user_movie_rating)
as(user_movie_rating_norm1, "matrix")[1,1:10]
user_movie_rating_norm2 = normalize(user_movie_rating, method="Z-score")
as(user_movie_rating_norm2, "matrix")[1,1:10]

image(user_movie_rating,main="Raw Ratings")
image(user_movie_rating_norm1,main="Centered Ratings")
image(user_movie_rating_norm2,main="Normalized Ratings")

summary(getRatings(user_movie_rating))
qplot(getRatings(user_movie_rating), binwidth = 1, 
      main = "Histogram of ratings", xlab = "Rating")
qplot(rowCounts(user_movie_rating), binwidth = 10, 
      main = "Movies Rated on average", 
      xlab = "# of users", 
      ylab = "# of movies rated")

recommenderRegistry$get_entries(dataType = "realRatingMatrix")

rec.model=Recommender(user_movie_rating,method="UBCF", param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))
str(rec.model)
getModel(rec.model)

# recommended top 5 items for first user
recommended.items = predict(rec.model, user_movie_rating[1:2,], n=5, type="topNList")
class(recommended.items)
str(recommended.items)
as(recommended.items, "list")

# to predict affinity to all non-rated items 
recommended.ratings = predict(rec.model, user_movie_rating[1:2,], type="ratings")
as(recommended.ratings, "matrix")[2][1:20]