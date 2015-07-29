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

ctrl = trainControl(method="cv", 10)

reg_model = train(Tip ~ ., data=RestaurantTips, method="lm", trControl=ctrl)
reg_model
reg_model$finalModel
reg_model$finalModel$residuals

predicted = predict(reg_model, RestaurantTips)
str(predicted)
