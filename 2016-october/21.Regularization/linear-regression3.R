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

reg_model1 = train(Tip ~ Bill + Guests + attr1, data=RestaurantTips, method="lm", trControl=ctrl)

reg_model2 = train(Tip ~ Bill + Guests + attr1, data=RestaurantTips, method="ridge", trControl=ctrl)

reg_model3 = train(Tip ~ Bill + Guests + attr1, data=RestaurantTips, method="lasso", trControl=ctrl)

reg_model1
reg_model1$finalModel

reg_model2
reg_model2$finalModel

reg_model3
reg_model3$finalModel
