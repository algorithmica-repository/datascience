library(caret)
library(filehash)
setwd("C:/Users/Algorithmica/Downloads")

#read test data
titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)
titanic_test$Pclass = as.factor(titanic_test$Pclass)

#import filehash db to current session
db = dbInit("titanic_models.db")
env = db2env(db)
print(ls(env))
env$tree_model1$finalModel

# use deploymed model for prediction on live data
titanic_test$Survived = predict(env$tree_model1, titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)
