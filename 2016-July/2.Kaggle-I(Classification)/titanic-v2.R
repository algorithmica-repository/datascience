setwd("D:\\kaggle\\titanic\\data")
titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)

set.seed(100)
titanic_test$Survived = sample(c(0,1),nrow(titanic_test),replace=T) 
result = titanic_test[,c("PassengerId","Survived")]
write.csv(result,"submission.csv",row.names = F)
