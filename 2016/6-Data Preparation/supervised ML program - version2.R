library(caret)
library(e1071)

#read and explore data
setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic_train = read.table("train.csv", header = TRUE, sep= ",",na.strings=c("NA",""))
dim(titanic_train)
str(titanic_train)
head(titanic_train)

#cast the features to desired type
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

#standardize the numeric features
titanic_train_numeric = titanic_train[sapply(titanic_train, is.numeric)]
preObj = preProcess(titanic_train_numeric, method=c("BoxCox","center","scale","medianImpute"))
titanic_train_std = predict(preObj,titanic_train_numeric)
titanic_train_all = data.frame(titanic_train[c("Pclass","Sex","Embarked","Survived")], titanic_train_std)

# build the model with parameter grid
tune_grid = data.frame(.k=c(3,5,7,9,11))
train_control = trainControl(method = "repeatedcv", number = 10, repeats = 5)
knn_model = train(Survived ~ Pclass + Age + Sex + Embarked, data=titanic_train_all, method="knn", tuneGrid=tune_grid, trControl=train_control)
knn_model

#predict the test data based on model
titanic_test = read.table("test.csv", TRUE, ",")
dim(titanic_test)
str(titanic_test)
summary(titanic_test)

#apply same preprocessing logic as that of train data
titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Name = as.character(titanic_test$Name)
titanic_test_numeric = titanic_test[sapply(titanic_test, is.numeric)]
titanic_test_std = predict(preObj, titanic_test_numeric)
titanic_test_all = data.frame(titanic_test[c("Pclass","Sex","Embarked")],titanic_test_std)

#predict the class using the model built with train data
titanic_test$Survived=predict(knn_model,titanic_test_all)
titanic_predict=titanic_test[,c("PassengerId","Survived")]
write.table(titanic_predict,"predictions.csv",col.names = TRUE, row.names = FALSE, sep = ",")
