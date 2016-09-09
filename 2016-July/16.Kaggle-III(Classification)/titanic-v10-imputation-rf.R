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

summary(titanic_train)
titanic_train$Embarked[is.na(titanic_train$Embarked)] = 'S'

preObj = preProcess(titanic_train[,c("Age", "Sex", "Parch","SibSp","Fare","Pclass","Embarked")], method=c("bagImpute"))
preObj$bagImp$Age$model$mtrees
titanic_train1=predict(preObj, titanic_train[,c("Age","Sex", "Parch","SibSp","Fare","Pclass","Embarked")] )
dim(titanic_train1)

set.seed(50)
tr_ctrl = trainControl(method="boot")
rf_grid = expand.grid(.mtry=2:6)
rf_model1 = train(titanic_train1[,c("Age","Sex", "Parch","SibSp","Fare","Pclass","Embarked")], titanic_train$Survived, method="rf", trControl = tr_ctrl)

stopCluster(cluster)

titanic_test = read.csv("test.csv")
dim(titanic_test)

titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test1=predict(preObj, titanic_test[,c("Age","Sex", "Parch","SibSp","Fare","Pclass","Embarked")] )
dim(titanic_test1)

titanic_test1$Survived = predict(rf_model1,titanic_test1)
titanic_test1$PassengerId = titanic_test$PassengerId
write.csv(titanic_test1[,c("PassengerId", "Survived")],"submission.csv", row.names=F)
