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
resampling_strategy = trainControl(method="cv", number = 10)
#knn_grid = expand.grid(.k=c(1,2,3,4))

dummy_obj = dummyVars(~Sex + Age + Embarked + Pclass + Parch + SibSp, titanic_train)
titanic_train1 = as.data.frame(predict(dummy_obj, titanic_train))
class(titanic_train1)
titanic_train1$Survived = titanic_train$Survived
head(titanic_train1)

knn_model = train(Survived ~ ., titanic_train1, method="knn", trControl = resampling_strategy)


titanic_test = read.csv("test.csv", na.strings = c("NA",""))
dim(titanic_test)
str(titanic_test)
summary(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

titanic_test1 = predict(dummy_obj, titanic_test)
head(titanic_test1)
dim(titanic_test1)

titanic_test$Survived = predict(knn_model, titanic_test1)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)
