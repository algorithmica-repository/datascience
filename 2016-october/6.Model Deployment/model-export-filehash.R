library(rpart)
library(caret)
library(filehash)

ls("package:filehash")
?dumpObjects
?show

setwd("C:/Users/Algorithmica/Downloads")
titanic_train = read.csv("train.csv", na.strings = c("NA",""))
class(titanic_train)
dim(titanic_train)
str(titanic_train)
#preparation of data
titanic_train$Survived = as.factor(titanic_train$Survived)
titanic_train$Pclass = as.factor(titanic_train$Pclass)

#model building using data frame interface
resample_strategy = trainControl(method="cv", number = 10) 
tree_grid = expand.grid(.cp=seq(0,1,0.1))
tree_model1 = train(titanic_train[,c("Sex", "Pclass", "Fare")], titanic_train$Survived, method="rpart", tuneGrid = tree_grid, trControl = resample_strategy)
var(tree_model1$resample$Accuracy)
#acc: 81.2  var:0.001540438

tree_model2 = train(titanic_train[,c("Sex", "Pclass", "Fare","Embarked")], titanic_train$Survived, method="rpart", tuneGrid = tree_grid, trControl = resample_strategy)
var(tree_model2$resample$Accuracy)
#acc:80.4  var:0.00178

#tree model1 has low bias than tree model2. we pick model1 for deployment
#save model object to filehash db
dumpObjects(tree_model1, dbName="titanic_models.db")
