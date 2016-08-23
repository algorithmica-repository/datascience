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
tr_ctrl = trainControl(method="cv", number = 10)

#automatic parameter tuning
tree_model1 = train(titanic_train[,c("Sex", "Pclass", "Embarked")], titanic_train[,"Survived"], method="rpart", trControl = tr_ctrl, tuneLength = 15)

#explicit parameter tuning
cart_grid = expand.grid(.cp=seq(0,1,0.01))
tree_model2 = train(titanic_train[,c("Sex", "Pclass", "Embarked","Fare","Age")], titanic_train[,"Survived"], method="rpart", trControl = tr_ctrl, tuneGrid = cart_grid)

tree_model2$finalModel
tree_model2$results
var(tree_model2$resample$Accuracy)
tree_model2$resampledCM
tree_model2$control$index


