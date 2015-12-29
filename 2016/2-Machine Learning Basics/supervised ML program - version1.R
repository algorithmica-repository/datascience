library(caret)

# 1.read and explore data
setwd("E:/data analytics/kaggle/titanic/data")
titanic_train = read.table("train.csv", TRUE, ",")
dim(titanic_train)
str(titanic_train)
head(titanic_train)

# 2.prepare data
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

# 3.Feature Engineering
# Here, features are selected manually
features = with(titanic_train, Survived ~ Pclass + Sex + Embarked)
  
# 4.Teach the machine the approach to learn patterns from data
# We are using knn approach here
knn_model = train(features,data=titanic_train,method="knn")
# Summary of model
knn_model
# Model details
knn_model$finalModel


# 5.Predict the classes of future data
titanic_test = read.table("test.csv", TRUE, ",")
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Survived=predict(knn_model,titanic_test)
# Select the required columns for submission
titanic_predict=titanic_test[,c("PassengerId","Survived")]
# Writing the predicted outcomes of passengers given in testdata
write.table(titanic_predict,"predictions.csv",col.names = TRUE, row.names = FALSE, sep = ",")
