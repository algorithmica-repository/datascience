titanic_test = read.csv("C:/Users/Algorithmica/Downloads/test.csv")
class(titanic_test)
dim(titanic_test)
titanic_test$Survived=0
write.csv(titanic_test[,c("PassengerId","Survived")],"C:/Users/Algorithmica/Downloads/submission.csv", row.names = F)
