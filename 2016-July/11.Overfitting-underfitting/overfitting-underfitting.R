library(rpart)
library(ggplot2)
library(caret)
setwd("D:/kaggle/titanic/data/")
titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

set.seed(50)
tr_ctrl = trainControl(method="cv")

#CART tree
cart_grid1 = expand.grid(.cp=0)
cart_model1 = train(titanic_train[,c("Sex", "Pclass", "Embarked","Fare","Parch","SibSp")], titanic_train[,"Survived"], method="rpart", trControl = tr_ctrl, tuneGrid = cart_grid1)
cart_model1$finalModel

survived = predict(cart_model1, titanic_train)
table(survived == titanic_train$Survived)

#Pruning CART tree with cp parameter: to reduce overfitting
cart_grid2 = expand.grid(.cp=seq(0,1,0.01))
cart_model2 = train(titanic_train[,c("Sex", "Pclass", "Embarked","Fare","Parch","SibSp")], titanic_train[,"Survived"], method="rpart", trControl = tr_ctrl, tuneGrid = cart_grid2)
cart_model2$finalModel
survived = predict(cart_model2, titanic_train)
table(survived == titanic_train$Survived)

#Under-fitted tree: CART tree with few features
cart_grid3 = expand.grid(.cp=0)
cart_model3 = train(titanic_train[,c("Sex", "Embarked")], titanic_train[,"Survived"], method="rpart", trControl = tr_ctrl, tuneGrid = cart_grid3)
cart_model3$finalModel
survived = predict(cart_model3, titanic_train)
table(survived == titanic_train$Survived)
