library(rpart)
library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Survived = as.factor(titanic_train$Survived)

set.seed(100)

tr_ctrl = trainControl(method="boot")
model = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rf', trControl = tr_ctrl, importance = TRUE, ntree=10)
model
model$finalModel
getTree(model$finalModel,1,T)

varImp(model$finalModel)
varImpPlot(model$finalModel, type = 1)
varImpPlot(model$finalModel, type = 2)
