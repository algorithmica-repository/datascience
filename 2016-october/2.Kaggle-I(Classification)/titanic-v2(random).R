titanic_test = read.csv("C:/Users/Algorithmica/Downloads/test.csv")
dim(titanic_test)
titanic_test$Survived=sample(c(0,1),418,replace=T)
write.csv(titanic_test[,c("PassengerId","Survived")],"C:/Users/Algorithmica/Downloads/submission.csv", row.names = F)
