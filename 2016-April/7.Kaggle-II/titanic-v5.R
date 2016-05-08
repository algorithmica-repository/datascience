library(rpart)
library(caret)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)
titanic_train$Survived = as.factor(titanic_train$Survived)
summary(titanic_train)

set.seed(100)
tr_ctrl = trainControl(method="cv")
tr_grid = data.frame(.cp=0.02)
#rpart_ctrl = rpart.control(minsplit=2)
model = bagging(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train)
model$mtrees

model1 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='rpart', trControl = tr_ctrl, tuneGrid = tr_grid)
model11$finalModel

model1 = train(x = titanic_train[,c("Sex", "Pclass", "Embarked", "Parch", "SibSp", "Fare")] , y = titanic_train$Survived,  method='rpart', trControl = tr_ctrl, tuneGrid = tr_grid)
model1$finalModel

model11 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='J48', trControl = tr_ctrl)
model11$finalModel

varImp(model3)
plot(varImp(model1))

tr_ctrl3 = trainControl(method="boot", number=10)
model3 = train(Survived ~ Sex + Pclass + Embarked + Parch + SibSp + Fare, data = titanic_train, method='treebag', trControl = tr_ctrl3)
model3$finalModel$mtrees
show(model3)

tr_ctrl4 = trainControl(method="boot")
model4 = train(x = titanic_train[,c("Sex", "Pclass", "Embarked", "Parch", "SibSp", "Fare")] , y = titanic_train$Survived, method='rf', trControl = tr_ctrl4, ntree = 100, importance = TRUE)
varImp(model4$finalModel)
getTree(model4$finalModel)
show(model3)

model5 = train(Survived ~ ., data = titanic_train, method='rf', trControl = tr_ctrl4, ntree = 100)

titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)

titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Name = as.character(titanic_test$Name)

titanic_test$Survived = predict(model, titanic_test , type="class")

result = titanic_test[,c("PassengerId","Survived")]
write.csv(result,"submission.csv",row.names = F, col.names = T )
