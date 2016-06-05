library(caret)

setwd("D:\\kaggle\\restaurant-revenue")

restaurant_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_train)
str(restaurant_train)
rev_range = range(restaurant_train$revenue)
summary(restaurant_train)

restaurant_test = read.csv("test.csv", header = TRUE, na.strings=c("NA",""))
dim(restaurant_test)
set.seed(100)
restaurant_test$Prediction = sample(rev_range[1]:rev_range[2], nrow(restaurant_test))
result = restaurant_test[,c("Id","Prediction")]
write.csv(result,"submission.csv",row.names = F)
