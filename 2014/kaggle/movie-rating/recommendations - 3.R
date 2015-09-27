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
user_movie[1:2,1:2]

user_movie_rating = as(user_movie,"realRatingMatrix")
class(user_movie_rating)
image(user_movie_rating,main="Raw Ratings")

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
recommended.items = predict(rec.model, user_movie_rating[1:2,], n=5)
class(recommended.items)
str(recommended.items)
as(recommended.items, "list")

# to predict affinity to all non-rated items 
recommended.ratings = predict(rec.model, user_movie_rating[1:2,], type="ratings")
as(recommended.ratings, "matrix")[2][1:20]


#rec=Recommender(r[1:400],method="UBCF", param=list(normalize = "Z-score",method="Jaccard",nn=5, minRating=1))
#rec=Recommender(r[1:100],method="IBCF", param=list(normalize = "Z-score",method="Jaccard",minRating=1))


scheme = evaluationScheme(user_movie_rating, method="cross-validation", k=4, given=5, goodRating=5)

scheme

algorithms = list(
  "random items" = list(name="RANDOM", param=NULL),
  "popular items" = list(name="POPULAR", param=NULL),
  "user-based CF" = list(name="UBCF", param=NULL),
  "item-based CF" = list(name="IBCF", param=NULL),
  "svd-based CF" = list(name="SVD", param=NULL)
)
# run algorithms, predict next n movies
results = evaluate(scheme, algorithms)

# Draw ROC curve
plot(results, annotate = 1:5, legend="topleft")

# See precision / recall
plot(results, "prec/rec", annotate=3)