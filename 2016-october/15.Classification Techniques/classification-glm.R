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
resampling_strategy = trainControl(method="cv", number = 10)

lr_model1 = train(Survived ~ Sex + Pclass, titanic_train, method="glm", trControl = resampling_strategy)
lr_model1$finalModel

titanic_test = read.csv("test.csv", na.strings = c("NA",""))
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

titanic_test$Survived = predict(lr_model1, titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)


