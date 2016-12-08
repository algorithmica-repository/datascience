library(caret)

setwd("D:\\revenue-prediction")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""), stringsAsFactors = T)
dim(restaurant_train)
str(restaurant_train)

restaurant_train$Open.Date = as.character(restaurant_train$Open.Date)
#add the missing level to existing levels
levels(restaurant_train$Type) = c(levels(restaurant_train$Type), "MB")

#creating a new feature
restaurant_train$num_days = as.numeric(as.Date("31-12-2014", format="%d-%m-%Y") - as.Date(restaurant_train$Open.Date, format= "%m/%d/%Y"))

#filter unwanted features
restaurant_train1 = restaurant_train[,-c(1,2,3)]
dim(restaurant_train1)

set.seed(100)
resampling_strategy = trainControl(method="boot")
treebag_model = train(revenue ~ ., restaurant_train1, method="treebag", trControl = resampling_strategy)
treebag_model$finalModel$mtrees[[15]]

rf_model = train(revenue ~ ., restaurant_train1, method="rf", trControl = resampling_strategy)

