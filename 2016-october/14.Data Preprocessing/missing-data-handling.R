library(rpart)
library(caret)

setwd("C:/Users/Algorithmica/Downloads")
titanic_train = read.csv("train.csv", na.strings = c("NA",""))
class(titanic_train)
dim(titanic_train)
str(titanic_train)
summary(titanic_train)

#typecast the required columns
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)

#handle missing data
titanic_train$Embarked[is.na(titanic_train$Embarked)] = 'S'
summary(titanic_train$Embarked)

titanic_train$Age[is.na(titanic_train$Age)] = mean(titanic_train$Age, na.rm = T)
summary(titanic_train$Age)

#model building using formula interface
#In formula interface, caret train function automatically transforms all factor types to continuous types
set.seed(100)
resampling_strategy = trainControl(method="boot", number = 10)
rf_grid = expand.grid(.mtry=c(1,2,3,4))

rf_model = train(titanic_train[,c("Sex", "Age", "Embarked", "Pclass", "Parch", "SibSp")], titanic_train$Survived, method="rf", trControl = resampling_strategy, ntree=200, tuneGrid = rf_grid)
getTree(rf_model$finalModel, 1, labelVar = T)

titanic_test = read.csv("test.csv", na.strings = c("NA",""))
dim(titanic_test)
str(titanic_test)
summary(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

titanic_test$Survived = predict(rf_model, titanic_test[,c("Sex", "Pclass", "Parch", "SibSp")])
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)
