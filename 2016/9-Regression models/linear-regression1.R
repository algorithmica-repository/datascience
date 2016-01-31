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

preObj = preProcess(RestaurantTips[,-c(2)], method=c("center","scale"))
RestaurantTips = predict(preObj, RestaurantTips)

ctrl = trainControl(method="cv", 10)
plot(RestaurantTips$Tip,RestaurantTips$Bill)

reg_model = train(Tip ~ Bill, data=RestaurantTips[1:150,], method="lm", trControl=ctrl)
reg_model
reg_model$finalModel$coefficients
reg_model$finalModel$residuals
reg_model$finalModel$fitted.values

predicted = predict(reg_model, RestaurantTips[151:157,])
str(predicted)
