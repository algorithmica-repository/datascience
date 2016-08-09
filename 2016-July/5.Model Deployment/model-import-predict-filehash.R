library(filehash)
setwd("D:/kaggle/titanic/data/")

db = dbInit("tree-models.db")
env = db2env(db)
print(ls(env))

titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)

titanic_test$Survived = predict(env$tree-model1,titanic_test)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names=F)