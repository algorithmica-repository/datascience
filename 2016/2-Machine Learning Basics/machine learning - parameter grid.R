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
  
#By default, the train method tries 2 values for parameter k
knn_model1 = train(features,data=titanic_train,method="knn")

#You can force the train method to try more values for k using tuneLength
knn_model2 = train(features,data=titanic_train,method="knn",tuneLength = 5)

#You can force the train method to try custom values for k using tuneGrid
tune_grid = data.frame(.k=c(3,4,5,6,7))
knn_model3 = train(features,data=titanic_train,method="knn",tuneGrid = tune_grid)

titanic_test = read.table("test.csv", TRUE, ",")
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Survived=predict(knn_model2,titanic_test)
titanic_predict=titanic_test[,c("PassengerId","Survived")]
write.table(titanic_predict,"predictions.csv",col.names = TRUE, row.names = FALSE, sep = ",")
