library(rpart)
library(caret)
library(randomForest)
library(foreach) 
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

#Build random forest model in parallel
model_rf = foreach(ntree=rep(100,4), .combine=combine, .multicombine=TRUE, .packages="randomForest") %dopar% 
            {
              randomForest(titanic_train[,c("Sex","Pclass","Embarked","Parch","SibSp","Fare")], titanic_train[,"Survived"], ntree=ntree)
            }
stopCluster(cl)

