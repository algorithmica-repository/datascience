library(caret)

setwd("D:\\kaggle\\restaurant-revenue")

load("prediction-models.RData")
print(ls())

restaurant_test = read.csv("test.csv", na.strings=c("","NA"))
dim(restaurant_test)
str(restaurant_test)



restaurant_test1 = predict(imputeObj,restaurant_test[,-1])
dim(restaurant_test1)
str(restaurant_test1)

restaurant_test2 = restaurant_test1[, var_obj$zeroVar==FALSE]
dim(restaurant_test2)
str(restaurant_test2)

features_new_df = add.features(restaurant_test2$Open.Date)
restaurant_test3 = cbind(restaurant_test2, features_new_df)
dim(restaurant_test3)
str(restaurant_test3)

features.exclude = c(1,2)
revenue = predict(model_rf,restaurant_test3[,-features.exclude])
result = data.frame(Id = restaurant_test[,"Id"], Prediction = revenue)

str(result)


