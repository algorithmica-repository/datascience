setwd("D:\\kaggle\\titanic\\data")
titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)
titanic_test
head(titanic_test,10)
tail(titanic_test)

set.seed(100)
tmp = runif(nrow(titanic_test))

titanic_test$Survived =ifelse(tmp <= 0.5,0,1) 
result = titanic_test[,c("PassengerId","Survived")]
write.csv(result,"submission.csv",row.names = F, col.names = T )
