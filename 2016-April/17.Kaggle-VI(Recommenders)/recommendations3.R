library(recommenderlab)
library(ggplot2)
library(reshape2)

#         Carrots  Grass  Pork  Beef  Corn  Fish
Rabbit   = c(10,  7, 1,  2, NA,  1)
Cow      = c( 7, 10, NA, NA, NA, NA)
Dog      = c(NA,  1, 10, 10, NA, NA)
Pig      = c( 5,  6,  4, NA,  7,  3)
Chicken  = c( 7,  6,  2, NA, 10, NA)
Pinguin  = c( 2,  2, NA,  2,  2, 10)
Bear     = c( 2, NA,  8,  8,  2,  7)
Lion     = c(NA, NA,  9, 10,  2, NA)
Tiger    = c(NA, NA,  8, NA, NA,  5)
Antilope = c( 6, 10,  1,  1, NA, NA)
Wolf     = c( 1, NA, NA,  8, NA,  3)
Sheep    = c(NA,  8, NA, NA, NA,  2)

# all the animals
animals = c("Rabbit","Cow","Dog","Pig","Chicken","Pinguin","Bear","Lion","Tiger","Antilope","Wolf","Sheep")
# all the foods
foods = c("Carrots","Grass","Pork", "Beef", "Corn", "Fish")
matrixRowAndColNames = list(animals, foods)
# create a matrix from the ratings
animal_food_ratings = matrix(
  data=c(Rabbit,Cow,Dog,Pig,Chicken,Pinguin,Bear,Lion,Tiger,Antilope,Wolf,Sheep),
  nrow=12,ncol=6,byrow=TRUE, matrixRowAndColNames)

dim(animal_food_ratings)
animal_food_ratings

animal_food_ratings_real = as(animal_food_ratings,"realRatingMatrix")
class(animal_food_ratings_real)
as(animal_food_ratings_real,"matrix")
image(animal_food_ratings_real,main="Raw Ratings",xlab="FoodItems",ylab="Animals")

summary(getRatings(animal_food_ratings_real))
qplot(getRatings(animal_food_ratings_real), binwidth = 1, 
      main = "Histogram of ratings", xlab = "Rating")

rec.model=Recommender(animal_food_ratings_real,method="UBCF", param=list(normalize = "Z-score",method="Cosine",nn=5, minRating=1))
getModel(rec.model)

# recommended top 3 items for first user
recommended.items = predict(rec.model, animal_food_ratings_real, n=3, type="topNList")
class(recommended.items)
as(recommended.items, "list")

# to predict affinity to all non-rated items 
recommended.ratings = predict(rec.model, animal_food_ratings_real, type="ratings")
as(recommended.ratings, "list")

scheme = evaluationScheme(animal_food_ratings_real, method="bootstrap", given=2, goodRating=5)
scheme

results = evaluate(scheme, method="UBCF",type="ratings")
results
getConfusionMatrix(results)
avg(results)

results = evaluate(scheme, method="UBCF", n=c(1,3,5,10,15,20),type="topNList")
getConfusionMatrix(results)
avg(results)

# plot true positive rate vs false positive rate
plot(results, annotate=TRUE)

#create precision recall plot
plot(results, "prec/rec", annotate=TRUE)