setwd("D:\\kaggle\\titanic\\data")
titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)
titanic_test
head(titanic_test,10)
tail(titanic_test)

titanic_test$Survived = rep(0,nrow(titanic_test))
result = titanic_test[,c("PassengerId","Survived")]
write.csv(result,"submission.csv",row.names = F, col.names = T )
