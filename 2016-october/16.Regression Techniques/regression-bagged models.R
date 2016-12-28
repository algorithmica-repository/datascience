library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\datasets\\restaurant-revenue")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_train)
str(restaurant_train)
restaurant_train1 = restaurant_train[,-c(1,2,3,5)]

set.seed(100)

bagging_strategy = trainControl(method="oob")
bagged_tree_model = train(revenue ~ ., restaurant_train1, method="treebag", trControl = bagging_strategy, ntree=1000,keepX=T, importance=T)
bagged_tree_model$finalModel
plot(varImp(bagged_tree_model))

rf_grid = expand.grid(.mtry=3:20)
rf_model = train(revenue ~ ., restaurant_train1, method="rf", trControl = bagging_strategy, tuneGrid = rf_grid,  ntree=1000, importance=T)
rf_model$finalModel
plot(rf_model$finalModel)
plot(varImp(rf_model))