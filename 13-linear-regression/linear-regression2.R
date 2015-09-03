library(ggplot2)
library(dplyr)
library(klaR)
library(e1071)
library(caret)
library(Lock5Data)
data(RestaurantTips)

dim(RestaurantTips)
str(RestaurantTips)
head(RestaurantTips)
  
RestaurantTips$attr1 = RestaurantTips$Bill + RestaurantTips$Guests

RestaurantTips$attr2 = RestaurantTips$Bill + RestaurantTips$Guests + rnorm(157,0,1) *0.05

ctrl = trainControl(method="cv", 10)

reg_model = train(Tip ~ Bill +  Guests  + attr2, data=RestaurantTips, method="lm", trControl=ctrl)
reg_model
reg_model$finalModel

predicted = predict(reg_model, RestaurantTips)
str(predicted)
