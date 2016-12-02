library(caret)

setwd("D:\\revenue-prediction")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_train)
str(restaurant_train)
restaurant_train1 = restaurant_train[,-c(1,2,3,5)]

set.seed(100)
resampling_strategy = trainControl(method="cv", number = 10)
knn_model = train(revenue ~ ., restaurant_train1, method="knn", trControl = resampling_strategy)

restaurant_test = read.csv("test.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_test)
restaurant_test1 = restaurant_test[,-c(1,2,3,5)] 

restaurant_test1$Id = restaurant_test$Id
restaurant_test1$Prediction = predict(knn_model, restaurant_test1)
result = restaurant_test1[,c("Id","Prediction")]
write.csv(result,"submission.csv",row.names = F)
