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

#explore individual variables
summary(titanic_train)

#explore multi-variate relationships numerically & graphically
xtabs(~Sex + Survived, titanic_train)
xtabs(~Pclass + Survived, titanic_train)
xtabs(~ Pclass + Survived + Sex, titanic_train)
xtabs(~ Embarked + Survived, titanic_train)
xtabs(~ Embarked + Survived + Sex, titanic_train)
xtabs(~ Embarked + Survived +  Pclass + Sex, titanic_train)
summary(titanic_train$Fare)
xtabs(~ Fare + Survived + Sex, titanic_train)

#model building using data frame interface
resample_strategy = trainControl(method="cv", number = 10) 
tree_grid = expand.grid(.cp=seq(0,1,0.1))
tree_model = train(titanic_train[,c("Sex", "Pclass", "Fare")], titanic_train$Survived, method="rpart", tuneGrid = tree_grid, trControl = resample_strategy)

#logic/pattern extracted on entire data
tree_model$finalModel

#the performance of each iteration of resampling technique (for optimal parameter value)
tree_model$resample

#train data used for each iteration of resampling technique
tree_model$control$index

#validation data used for each iteration of resampling technique
tree_model$control$indexOut


#apply model to test/unseen data
titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

titanic_test$Survived = predict(tree_model, titanic_test, type="class")
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)
