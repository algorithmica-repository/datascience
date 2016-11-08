library(caret)
library(pmml)
setwd("C:/Users/Algorithmica/Downloads")

ls("package:pmml")
#read test data
titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

#import titanic pmml model
#no package available

# use deploymed model for prediction on live data
titanic_test$Survived = predict(, titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)
