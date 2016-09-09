library(ggplot2)
library(caret)
library(doParallel)

cluster = makeCluster(detectCores())
registerDoParallel(cluster)

setwd("D:/kaggle/titanic/data/")
titanic_train = read.csv("train.csv", na.strings=c("NA",""," "))
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

set.seed(100)
tr_ctrl = trainControl(method="boot")
model = train(Survived ~ Sex + Pclass + Embarked + Fare + Parch + SibSp, data = titanic_train, method='rf', trControl = tr_ctrl, ntree = 500)

stopCluster(cl)


