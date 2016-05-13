library(rpart)
library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")
titanic_train = read.csv("train.csv",header = TRUE, na.strings=c("NA",""))
dim(titanic_train)
str(titanic_train)

titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Embarked[is.na(titanic_train$Embarked)] = 'S'

set.seed(100)
tr_ctrl = trainControl(method="boot")
#model1 = train(Survived ~ FamilySize + Title + Sex + Pclass + Embarked + Fare, data = titanic_train, method='rf', trControl = tr_ctrl, ntree = 250)

model2 = train(x = titanic_train[,c("Sex", "Pclass", "Embarked", "Fare", "Parch", "SibSp")], y = titanic_train[,"Survived"], data = titanic_train, method='rf', trControl = tr_ctrl, ntree = 500)

titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_train)

titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Fare = ifelse(is.na(titanic_test$Fare), mean(titanic_test$Fare, na.rm = TRUE), titanic_test$Fare)

titanic_test$Survived = predict(model2, titanic_test[,c("Sex", "Pclass", "Embarked", "Fare", "Parch", "SibSp")])
result = titanic_test[,c("PassengerId","Survived")]
write.csv(result,"submission.csv",row.names = F)
