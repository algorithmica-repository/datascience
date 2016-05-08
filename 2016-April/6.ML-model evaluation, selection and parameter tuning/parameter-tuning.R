library(rpart)
library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Survived = as.factor(titanic_train$Survived)

set.seed(100)

#using tune grid of values
tr_grid = data.frame(.cp=seq(0,1,0.01))
tr_ctrl = trainControl(method="cv", number = 10)
model1 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl, tuneGrid = tr_grid)
model1$finalModel

#using tunelength
model2 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl, tuneLength = 5 )
model2$finalModel


