library(caret)

setwd("D:\\revenue-prediction")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_train)
str(restaurant_train)
restaurant_train1 = restaurant_train[,-c(1,2,3,5)]

set.seed(100)
knn_strategy = trainControl(method="cv", number = 10)
knn_model = train(revenue ~ ., restaurant_train1, method="knn", trControl = knn_strategy)
plot(knn_model)