library(recommenderlab)
library(ggplot2)

#loading the data
data(MovieLense)

r = MovieLense
str(r)
class(r)
dim(r)

as(r,"matrix")[1:2,1:10]

summary(getRatings(r))
qplot(getRatings(r), binwidth = 1, 
      main = "Histogram of ratings", xlab = "Rating")

rowMeans(r[1,])

r_norm = normalize(r)
as(r_norm, "matrix")[1,1:10]

r_norm = normalize(r, method="Z-score")
as(r_norm, "matrix")[1,1:10]

image(MovieLense,main="Raw Ratings")
image(r_norm,main="Normalized Ratings")

qplot(rowCounts(MovieLense), binwidth = 10, 
      main = "Movies Rated on average", 
      xlab = "# of users", 
      ylab = "# of movies rated")

recommenderRegistry$get_entries(dataType = "realRatingMatrix")

rec=Recommender(r,method="UBCF", param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))
str(rec)
getModel(rec)
recom = predict(rec, r[1:50], type="ratings")

as(recom, "matrix")[1:2,1:10]

rec=Recommender(r[1:400],method="UBCF", param=list(normalize = "Z-score",method="Jaccard",nn=5, minRating=1))
rec=Recommender(r[1:100],method="IBCF", param=list(normalize = "Z-score",method="Jaccard",minRating=1))


scheme <- evaluationScheme(MovieLense, method="split", train=0.9, k=1, given=10, goodRating=4)

scheme

algorithms <- list(
  "random items" = list(name="RANDOM", param=list(normalize = "Z-score")),
  "popular items" = list(name="POPULAR", param=list(normalize = "Z-score")),
  "user-based CF" = list(name="UBCF", param=list(normalize = "Z-score", method="Cosine", nn=50, minRating=3)),
  "item-based CF" = list(name="IBCF2", param=list(normalize = "Z-score", method="Cosine"))
)
# run algorithms, predict next n movies
results <- evaluate(scheme, algorithms, n=c(1, 3, 5, 10, 15, 20))

# Draw ROC curve
plot(results, annotate = 1:4, legend="topleft")

# See precision / recall
plot(results, "prec/rec", annotate=3)

#ggplot(r_df, aes(x = ratings)) + geom_histogram(aes(y=..density..),binwidth=0.5,colour="black", fill="white") + geom_density() + xlab("Rating") + labs(title="Histogram of ratings")
