setwd("D:\\kaggle\\titanic\\data")
titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)

titanic_test$Survived = ifelse(titanic_test$Sex=="female",0,1) 
result = titanic_test[,c("PassengerId","Survived")]
write.csv(result,"submission.csv",row.names = F)
