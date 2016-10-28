setwd("C:/Users/Algorithmica/Downloads")
titanic_train = read.csv("train.csv")
class(titanic_train)
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)

#explore individual variables
summary(titanic_train)

#explore multi-variate relationships numerically & graphically
xtabs(~Sex + Survived, titanic_train)
xtabs(~Pclass + Survived, titanic_train)
xtabs(~ Pclass + Survived + Sex, titanic_train)
xtabs(~ Embarked + Survived, titanic_train)
xtabs(~ Embarked + Survived + Sex, titanic_train)
xtabs(~ Embarked + Survived +  Pclass + Sex, titanic_train)
summary(titanic_train$Fare)
xtabs(~ Fare + Survived + Sex, titanic_train)

titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)
titanic_test$Survived = NA
for(i in 1:nrow(titanic_test)) {
  if(titanic_test[i,"Sex"] == "female") {
    if(titanic_test[i,"Pclass"]=='3' & titanic_test[i,"Embarked"]=='S')
      titanic_test$Survived[i] = 0
    else
      titanic_test$Survived[i] = 1
  } else {
    titanic_test$Survived[i] = 0
  }
  titanic_test$Survived[i] = ifelse(titanic_test[i,"Fare"] > 75.0, 1, titanic_test$Survived[i]) 
}

titanic_test$Survived[153] = 0
write.csv(titanic_test[,c("PassengerId","Survived")],"submission.csv", row.names = F)
