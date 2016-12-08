library(caret)

setwd("D:\\revenue-prediction")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""), stringsAsFactors = T)
dim(restaurant_train)
str(restaurant_train)

#filter unwanted features
restaurant_train1 = restaurant_train[,-c(1,2,3)]
dim(restaurant_train1)

set.seed(100)
resampling_strategy = trainControl(method="cv", number=50)
cart_grid = expand.grid(.cp=0)
cart_model = train(revenue ~ ., restaurant_train1, method="rpart", trControl = resampling_strategy, tuneGrid = cart_grid)
cart_model$finalModel

