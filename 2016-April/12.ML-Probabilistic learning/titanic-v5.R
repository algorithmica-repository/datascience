library(rpart)
library(caret)
library(stringr)
library(magrittr)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
titanic_test = read.csv("test.csv", header = TRUE, na.strings=c("NA",""))
titanic_test$Survived = NA

# Combine train and test data for common preprocessing. It avoids issues like factor column mismatch,.,
titanic = rbind(titanic_train, titanic_test)
dim(titanic)
summary(titanic)

# Type Cast the features to required target taype 
titanic$Pclass = as.factor(titanic$Pclass)
titanic$Name = as.character(titanic$Name)
titanic$Survived = as.factor(titanic$Survived)

# Handle missing data with mode and mean
titanic$Embarked[is.na(titanic$Embarked)] = 'S'
titanic$Fare[is.na(titanic$Fare)] = mean(titanic_train$Fare, na.rm = TRUE)

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
titanic$Title = sapply(titanic$Name, FUN=extract_title)
titanic$Title = factor(titanic$Title)

# Feature engineering of family size column
titanic$FamilySize = titanic$Parch + titanic$SibSp + 1

titanic_train = titanic[1:891,]
titanic_test = titanic[892:1309,]
dim(titanic_train)
dim(titanic_test)

xtabs(~Survived + Title, data=titanic_train)
ggplot(titanic_train, aes(x = Title, fill = Survived)) + geom_bar(position = "fill")

set.seed(100)
tr_ctrl = trainControl(method="cv")

model1 = train(Survived ~ FamilySize + Title + Sex + Pclass + Embarked + Fare, data = titanic_train, method='rf', trControl = tr_ctrl, ntree = 500)

model2 = train(x = titanic_train[,c("FamilySize", "Title", "Sex", "Pclass", "Embarked", "Fare")], y = titanic_train[,"Survived"], data = titanic_train, method='nb', trControl = tr_ctrl)
model2$finalModel
model2$finalModel$apriori
model2$finalModel$tables

tn_grid = data.frame(.fL=1, .usekernel=F)
model3 = train(x = titanic_train[,c("FamilySize", "Title", "Sex", "Pclass", "Embarked", "Fare")], y = titanic_train[,"Survived"], data = titanic_train, method='nb', trControl = tr_ctrl, tuneGrid = tn_grid)


titanic_test$Survived = predict(model2, titanic_test[,c("FamilySize", "Title", "Sex", "Pclass", "Embarked", "Fare")])
result = titanic_test[,c("PassengerId","Survived")]
write.csv(result,"submission.csv",row.names = F)
