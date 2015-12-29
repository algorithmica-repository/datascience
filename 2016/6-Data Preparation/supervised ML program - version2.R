library(caret)
library(e1071)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic_train = read.table("train.csv", header = TRUE, sep= ",",na.strings=c("NA",""))
dim(titanic_train)
str(titanic_train)
head(titanic_train)

titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)
titanic_train_numeric = titanic_train[sapply(titanic_train, is.numeric)]

preObj = preProcess(titanic_train, method=c("BoxCox","center","scale","medianImpute"))

titanic_train_std = predict(preObj,titanic_train)

features = with(titanic_train, Survived ~ Pclass + Age + Sex + Fare)

tune_grid = data.frame(.k=c(3,5,7,9,11))

train_control = trainControl(method = "repeatedcv", number = 10, repeats = 5)
  
knn_model = train(features, data=titanic_train_std, method="knn", tuneGrid=tune_grid, trControl=train_control)
knn_model

titanic_test = read.table("test.csv", TRUE, ",")
dim(titanic_test)
str(titanic_test)
summary(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test_std = predict(preObj, titanic_test)
titanic_test$Survived=predict(knn_model,titanic_test_std)
titanic_predict=titanic_test[,c("PassengerId","Survived")]
write.table(titanic_predict,"predictions.csv",col.names = TRUE, row.names = FALSE, sep = ",")
