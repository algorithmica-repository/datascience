library(rpart)
library(ggplot2)
library(caret)
setwd("E:/data")
titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

set.seed(50)
tr_ctrl = trainControl(method="cv", number = 10)

#explicit parameter tuning
svm_grid = expand.grid(.degree=1:4, .scale=c(0.01,0.1), .C=seq(0,1,0.3))
svm_model = train(Survived ~ Sex + Pclass + Embarked + Fare + Parch  + SibSp, titanic_train, method="svmPoly", trControl = tr_ctrl, tuneGrid =  svm_grid)
svm_model$finalModel

svm_grid = expand.grid(.degree=1:4, .scale=c(0.01,0.1), .C=seq(0,1,0.3))
svm_model = train(Survived ~ Sex + Pclass + Embarked + Fare + Parch  + SibSp, titanic_train, method="svmRadial", trControl = tr_ctrl)
svm_model$finalModel

svm_model = train(Survived ~ Sex + Pclass + Embarked + Fare + Parch  + SibSp, titanic_train, method="svmPoly", trControl = tr_ctrl)
svm_model$finalModel

#explicit parameter tuning
cart_grid = expand.grid(.cp=seq(0,1,0.01))
cart_model = train(titanic_train[,c("Sex", "Pclass", "Embarked","Fare","Parch","SibSp")], titanic_train[,"Survived"], method="rpart", trControl = tr_ctrl, tuneGrid = cart_grid)

titanic_test = read.csv("test.csv")
dim(titanic_test)

titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Survived = predict(cart_model,titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names=F)


