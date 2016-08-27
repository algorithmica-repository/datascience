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
nb_model = train(titanic_train[,c("Sex", "Pclass", "Embarked","Fare","Parch","SibSp")], titanic_train[,"Survived"], method="nb", trControl = tr_ctrl)
nb_model$finalModel


titanic_test = read.csv("test.csv")
dim(titanic_test)

titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Survived = predict(nb_model,titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names=F)


