getwd()
setwd("C:/Users/Algorithmica/Downloads")
titanic_train = read.csv("train.csv")
class(titanic_train)
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)

#explore individual variables
summary(titanic_train)

#explore multi-variate relationships
xtabs(~Sex + Survived, titanic_train)
xtabs(~Pclass + Survived, titanic_train)
xtabs(~ Pclass + Survived + Sex, titanic_train)
xtabs(~ Embarked + Survived, titanic_train)
xtabs(~ Embarked + Survived + Sex, titanic_train)
xtabs(~ Embarked + Pclass + Survived + Sex, titanic_train)

titanic_test = read.csv("test.csv")
dim(titanic_test)
titanic_test$Survived=ifelse(titanic_test$Sex == "female",1,0)
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)
