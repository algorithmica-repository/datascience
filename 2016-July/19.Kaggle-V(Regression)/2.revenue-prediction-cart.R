library(caret)

setwd("D:\\revenue-prediction")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_train)
str(restaurant_train)
summary(restaurant_train)
restaurant_train$Type = factor(restaurant_train$Type,levels=c('FC','DT','IL','MB'))

set.seed(100)
tr_ctrl = trainControl(method="cv")
cart_grid = expand.grid(.cp=c(0,0.005,0.006,0.009,0.05))
cart_model = train(x = restaurant_train[,-c(1,2,3,ncol(restaurant_train))], y = restaurant_train[,ncol(restaurant_train)], method='rpart', trControl = tr_ctrl, tuneGrid = cart_grid)
cart_model$finalModel

restaurant_test = read.csv("test.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_test)
summary(restaurant_test)
restaurant_test$Prediction = predict(cart_model, restaurant_test[,-c(1,2,3)])
result = restaurant_test[,c("Id","Prediction")]
write.csv(result,"submission.csv",row.names = F)
