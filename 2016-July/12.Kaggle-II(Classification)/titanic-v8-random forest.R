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
rf_model = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp, titanic_train, method="rf", trControl = tr_ctrl)
rf_model$finalModel$forest

titanic_test = read.csv("test.csv")
dim(titanic_test)

titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Survived = predict(rf_model,titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names=F)


