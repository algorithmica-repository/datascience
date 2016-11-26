library(rpart)
library(caret)
library(RWeka)

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
cart_grid = expand.grid(.cp=seq(0,0.2,0.001))
resampling_strategy = trainControl(method="cv", number = 10)

cart_model1 = train(titanic_train[,c("Sex", "Pclass", "Fare")], titanic_train$Survived, method="rpart", trControl = resampling_strategy,  tuneGrid = cart_grid)
cart_model1$finalModel

cart_model2 = train(titanic_train[,c("Sex", "Pclass", "Embarked", "Fare")], titanic_train$Survived, method="rpart", trControl = resampling_strategy,  tuneGrid = cart_grid)
cart_model2$finalModel

titanic_test = read.csv("test.csv", na.strings = c("NA",""))
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

titanic_test$Survived = predict(cart_model1, titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)


