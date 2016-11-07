library(caret)
setwd("C:/Users/Algorithmica/Downloads")

#read test data
titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

#import RData file to current session
load("titanic_model.RData")
ls()
tree_model1$finalModel

# use deploymed model for prediction on live data
titanic_test$Survived = predict(tree_model1, titanic_test, type="prob")
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)
