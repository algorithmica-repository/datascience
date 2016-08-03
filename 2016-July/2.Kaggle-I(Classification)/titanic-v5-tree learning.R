library(rpart)
library(ggplot2)
setwd("D:/kaggle/titanic/data/")
titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

tree_model1 = rpart(Survived~Sex + Pclass + Embarked, titanic_train)

tree_model2 = rpart(Survived~Sex + Pclass + Embarked + Age + Parch + SibSp + Fare, titanic_train)

tree_model3 = rpart(Survived~Sex + Pclass + Embarked + Age + Fare, titanic_train)

titanic_test = read.csv("test.csv")
dim(titanic_test)
nrow(titanic_test)
str(titanic_test)

titanic_test$Pclass = as.factor(titanic_test$Pclass)

titanic_test$Survived = predict(tree_model2,titanic_test, type="class")
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names=F)

 