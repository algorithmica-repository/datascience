library(rpart)
library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Survived = as.factor(titanic_train$Survived)

set.seed(100)

model0 = rpart(Survived ~ Pclass + Sex + Age + Fare, data = titanic_train, method="class")
model0

tr_ctrl = trainControl(method="none")
tr_grid = data.frame(.cp=0.02)

model1 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl, tuneGrid = tr_grid)
model11$finalModel

model2 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='J48', trControl = tr_ctrl, tuneGrid = tr_grid)
model2$finalModel