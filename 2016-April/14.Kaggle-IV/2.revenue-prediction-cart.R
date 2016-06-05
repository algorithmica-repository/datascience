library(caret)

setwd("D:\\kaggle\\restaurant-revenue")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
restaurant_test = read.csv("test.csv", header = TRUE, na.strings=c("NA",""))
restaurant_test$revenue = NA
restaurant = rbind(restaurant_train, restaurant_test)
dim(restaurant)
str(restaurant)
summary(restaurant)

restaurant_train = restaurant[1:137,]
set.seed(100)
tr_ctrl = trainControl(method="cv", number = 20)
model1 = train(x = restaurant_train[,-c(1,2,ncol(restaurant_train))], y = restaurant_train[,ncol(restaurant_train)], method='rpart', trControl = tr_ctrl)
model$finalModel

restaurant_test = restaurant[138:nrow(restaurant),]
dim(restaurant_test)
str(restaurant_test)
restaurant_test$revenue = predict(model, restaurant_test[,-c(1,2,ncol(restaurant_test))])
result = restaurant_test[,c("Id","revenue")]
names(result) = c("Id","Prediction")
write.csv(result,"submission.csv",row.names = F)
