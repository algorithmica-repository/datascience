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
tr_ctrl = trainControl(method="boot")
baggedtree_model = train(titanic_train[,c("Sex", "Pclass", "Embarked","Fare","Parch","SibSp")], titanic_train[,"Survived"], method="treebag", trControl = tr_ctrl)
baggedtree_model$finalModel
baggedtree_model$finalModel$mtrees

rf_model = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp, titanic_train, method="rf", trControl = tr_ctrl)
rf_model$finalModel$forest


