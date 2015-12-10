library(caret)

setwd("E:/data analytics/kaggle/titanic/data")
titanic_train = read.table("train.csv", TRUE, ",")
dim(titanic_train)
str(titanic_train)
head(titanic_train)

titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

features = with(titanic_train, Survived ~ Pclass + Sex + Embarked)
  
control1 = trainControl(method = "LGOCV", p=0.6, number = 10)

control2 = trainControl(method = "cv", number = 10)

control3 = trainControl(method = "repeatedcv", number = 10, repeats = 5)

control4 = trainControl(method = "LOOCV", number = 10)

control5 = trainControl(method = "boot", number = 10)

knn_model = train(features,data=titanic_train,method="knn",trControl = control5)

titanic_test = read.table("test.csv", TRUE, ",")
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Survived=predict(knn_model2,titanic_test)
titanic_predict=titanic_test[,c("PassengerId","Survived")]
write.table(titanic_predict,"predictions.csv",col.names = TRUE, row.names = FALSE, sep = ",")
