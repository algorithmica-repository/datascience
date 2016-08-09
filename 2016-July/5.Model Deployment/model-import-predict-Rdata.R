setwd("D:/kaggle/titanic/data/")
load("tree-model.RData")
print(ls())

titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)

titanic_test$Survived = predict(tree-model1,titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names=F)