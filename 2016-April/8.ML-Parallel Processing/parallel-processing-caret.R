library(caret)
library(doParallel)

#register cluster for parallel processing
cl = makeCluster(detectCores())
registerDoParallel(cl)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Survived = as.factor(titanic_train$Survived)

set.seed(100)
tr_ctrl = trainControl(method="boot")

model = train(Survived ~ Sex + Pclass + Embarked + Fare + Parch + SibSp, data = titanic_train, method='rf', trControl = tr_ctrl, ntree = 500)

stopCluster(cl)


