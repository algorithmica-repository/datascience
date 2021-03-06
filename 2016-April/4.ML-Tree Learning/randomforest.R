library(rpart)
library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Survived = as.factor(titanic_train$Survived)

set.seed(100)

tr_ctrl = trainControl(method="boot", number = 100)
model = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rf', trControl = tr_ctrl)
model$finalModel$mtrees
