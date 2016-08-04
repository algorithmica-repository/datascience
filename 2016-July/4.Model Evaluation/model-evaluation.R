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

tree_model1 = rpart(Survived~Sex + Pclass + Embarked, titanic_train)

tree_model2 = rpart(Survived~Sex + Pclass + Embarked + Age + Parch + SibSp + Fare, titanic_train)

tree_model3 = rpart(Survived~Sex + Pclass + Embarked + Age + Fare, titanic_train)

set.seed(50)
tr_ctrl = trainControl(method="LGOCV", number = 10, p = 0.8)
#It perform resampling
#It builds model and validates it based on given resampling strategy
#It also allows us to tune model parameters and picks the model with best paramters
#It also provides final model with entire training data
tree_model1 = train(Survived~Sex + Pclass + Embarked, titanic_train, method="rpart", trControl = tr_ctrl)
tree_model1
tree_model1$finalModel
tree_model1$resample
tree_model1$control$index
tree_model1$control$indexOut

tree_model2 = train(Survived~Sex + Pclass + Age + Embarked + Parch + SibSp + Fare, titanic_train, method="rpart", trControl = tr_ctrl)
tree_model2
tree_model2$finalModel

tree_model3 = train(Survived~Sex + Pclass + Embarked + Age + Fare, titanic_train, method="rpart", trControl = tr_ctrl)
tree_model3
tree_model3$finalModel
tree_model3$resample
