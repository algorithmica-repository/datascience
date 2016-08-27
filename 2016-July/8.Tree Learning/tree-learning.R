library(rpart)
library(ggplot2)
library(caret)
setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

tree = rpart(Survived~Sex + Pclass + Embarked + Fare + Parch,data=titanic_train,control = rpart.control(cp=0))
set.seed(50)
tr_ctrl = trainControl(method="cv", number = 10)

#automatic parameter tuning
cart_model1 = train(titanic_train[,c("Sex", "Pclass", "Embarked")], titanic_train[,"Survived"], method="rpart", trControl = tr_ctrl, tuneLength = 15)

cart_model1$finalModel
cart_model1$results
cart_model1$resample
cart_model1$resampledCM
cart_model1$control$index

#explicit parameter tuning
cart_grid = expand.grid(.cp=seq(0,1,0.01))
cart_model2 = train(titanic_train[,c("Sex", "Pclass", "Embarked","Fare", "Parch")], titanic_train[,"Survived"], method="rpart", trControl = tr_ctrl,  tuneGrid = cart_grid)