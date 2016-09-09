library(ggplot2)
library(caret)
library(doParallel)
library(stringr)

cluster = makeCluster(detectCores())
registerDoParallel(cluster)

setwd("D:/kaggle/titanic/data/")
titanic_train = read.csv("train.csv", na.strings=c("NA",""," "))
titanic_test = read.csv("test.csv", na.strings=c("NA",""," "))
titanic_test$Survived = NA
titanic = rbind(titanic_train, titanic_test)
dim(titanic)
str(titanic)

titanic$Survived = as.factor(titanic$Survived)
titanic$Pclass = as.factor(titanic$Pclass)
titanic$Name = as.character(titanic$Name)

summary(titanic)

titanic$Embarked[is.na(titanic$Embarked)] = 'S'
titanic$Fare[is.na(titanic$Fare)] = median(titanic$Fare, na.rm=T)

preObj = preProcess(titanic[,c("Age", "Sex", "Parch","SibSp","Fare","Pclass","Embarked")], method=c("bagImpute"))
preObj$bagImp$Age$model$mtrees
titanic1=predict(preObj, titanic[,c("Age","Sex", "Parch","SibSp","Fare","Pclass","Embarked")] )
summary(titanic1)

titanic1$Survived = titanic$Survived

# function to extract title from names of passengers
extract_title = function(x) {
  title = str_trim(strsplit(x, split='[,.]')[[1]][2])
  if(title %in% c('Mme', 'Mlle') ) 
    return('Mlle')
  else if(title %in% c('Dona', 'Lady', 'the Countess'))
    return('Lady')
  else if(title %in% c('Capt', 'Don', 'Major', 'Sir', 'Jonkheer', 'Dr') )
    return('Sir')
  else
    return(title)
}

# Feature engineering of name column
titanic1$Title = sapply(titanic$Name, FUN=extract_title)
titanic1$Title = factor(titanic1$Title)
str(titanic1)

titanic1$FamilySize = titanic1$SibSp + titanic1$Parch + 1

extract_id = function(x) {
  lname = str_trim(strsplit(x, split='[,.]')[[1]][1])
  return(lname)
}
titanic1$Surname = sapply(titanic$Name, FUN=extract_id)
titanic1$FamilyId = paste(titanic1$Surname, titanic1$FamilySize,sep="")
titanic1$FamilyId[titanic1$FamilySize <= 3] = "Small"
titanic1$FamilyId = factor(titanic1$FamilyId)
str(titanic1$FamilyId)

titanic1$PassengerId = titanic$PassengerId

titanic_train = titanic1[1:891,]
str(titanic_train)
titanic_test = titanic1[892:nrow(titanic1),]
str(titanic_test)

set.seed(50)
tr_ctrl = trainControl(method="boot")
rf_grid = expand.grid(.mtry=2:10)
rf_model = train(titanic_train[,c("Title", "FamilyId" , "FamilySize", "Age","Sex", "Parch","SibSp","Fare","Pclass","Embarked")], titanic_train$Survived, method="rf", trControl = tr_ctrl, tuneGrid = rf_grid, ntree = 1000, importance=T)
varImp(rf_model)
X11()
varImpPlot(rf_model$finalModel)

stopCluster(cluster)

titanic_test$Survived = predict(rf_model,titanic_test[,-c(8)])
write.csv(titanic_test[,c("PassengerId", "Survived")],"submission.csv", row.names=F)
