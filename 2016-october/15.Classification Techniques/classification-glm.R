library(rpart)
library(caret)

setwd("C:/Users/Algorithmica/Downloads")
titanic_train = read.csv("train.csv", na.strings = c("NA",""))
class(titanic_train)
dim(titanic_train)
str(titanic_train)
#preparation of data
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)

#model building using formula interface
#In formula interface, caret train function automatically transforms all factor types to continuous types
set.seed(100)
#bag_tree_grid = expand.grid(.cp=seq(0,0.2,0.001))
resampling_strategy = trainControl(method="boot", number = 10)

bag_tree_model = train(titanic_train[,c("Sex", "Pclass", "Fare")], titanic_train$Survived, method="treebag", trControl = resampling_strategy)
bag_tree_model$finalModel$mtrees[[2]]

rf_grid = expand.grid(.mtry=c(2,3))
rf_model = train(titanic_train[,c("Sex", "Pclass", "Parch", "SibSp")], titanic_train$Survived, method="rf", trControl = resampling_strategy, ntree=3000, tuneGrid = rf_grid)
getTree(rf_model$finalModel, 1, labelVar = T)

titanic_test = read.csv("test.csv", na.strings = c("NA",""))
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

titanic_test$Survived = predict(bag_tree_model1, titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)


