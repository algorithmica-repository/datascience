library(ggplot2)
library(caret)
library(doParallel)
library(stringr)

cluster = makeCluster(detectCores())
registerDoParallel(cluster)

setwd("D:/kaggle/titanic/data/")
titanic_train = read.csv("train.csv", na.strings=c("NA",""," "))
dim(titanic_train)
str(titanic_train)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)

summary(titanic_train)
titanic_train$Embarked[is.na(titanic_train$Embarked)] = 'S'

preObj = preProcess(titanic_train[,c("Age", "Sex", "Parch","SibSp","Fare","Pclass","Embarked")], method=c("bagImpute"))
preObj$bagImp$Age$model$mtrees
titanic_train1=predict(preObj, titanic_train[,c("Age","Sex", "Parch","SibSp","Fare","Pclass","Embarked")] )
dim(titanic_train1)

titanic_train1$Survived = titanic_train$Survived

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
titanic_train1$Title = sapply(titanic_train$Name, FUN=extract_title)
titanic_train1$Title = factor(titanic_train1$Title)
str(titanic_train1)
X11()
ggplot(titanic_train1) + geom_bar(aes(x=Title, fill="Survived"), position = "fill")
xtabs(~ Title + Survived, titanic_train1)

extract_id = function(x) {
  lname = str_trim(strsplit(x, split='[,.]')[[1]][1])
  return(lname)
}
titanic_train1$Surname = sapply(titanic_train$Name, FUN=extract_id)
titanic_train1$FamilySize = titanic_train1$SibSp + titanic_train1$Parch + 1
titanic_train1$FamilyId = paste(titanic_train1$Surname, titanic_train1$FamilySize,sep="")
titanic_train1$FamilyId[titanic_train1$FamilySize <= 3] = "Small"
titanic_train1$FamilyId = factor(titanic_train1$FamilyId)
summary(titanic_train1$FamilyId)
write.csv(titanic_train1,"titanic_new.csv", row.names=F)
