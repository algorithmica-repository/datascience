library(rpart)
library(caret)

resubstitution_accuracy = function(train_data) {
  model = rpart(Survived ~ Pclass + Sex + Age + Fare, data = train_data, method="class")
  pred = predict(model, train_data , type="class")
  cm = confusionMatrix(train_data$Survived, pred)
  return(cm$overall["Accuracy"])
}

holdout_accuracy = function(train_data, holdout_percent, rep) {
  holdout_quantity = holdout_percent * nrow(train_data)

  all.acc = numeric(0)
  for(k in 1:rep) {
    train = sample(1:nrow(train_data), holdout_quantity)
    model = rpart(Survived ~ Pclass + Sex + Age + Fare, data = train_data[train,], method="class")
    pred = predict(model, train_data[-train,] , type="class")
    cm = confusionMatrix(train_data$Survived[-train], pred)
    all.acc = rbind(all.acc, cm$overall["Accuracy"])
  }
  return(mean(all.acc))
}

crossvalidate_accuracy = function(train_data, K) {
  set.seed(5)
  n = nrow(train_data)
  part_size = as.integer(n/K)
  range = rank(runif(n))
  block = as.integer((range-1)/part_size) + 1
  block = as.factor(block)
  
  all.acc = numeric(0)
  for(k in 1:K) {
    model = rpart(Survived ~ Pclass + Sex + Age + Fare, data = train_data[block!=k,], method="class")
    pred = predict(model, train_data[block==k,] , type="class")
    cm = confusionMatrix(train_data$Survived[block==k], pred)
    all.acc = rbind(all.acc, cm$overall["Accuracy"])
  }
  return(mean(all.acc))
}

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv")
dim(titanic_train)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Survived = as.factor(titanic_train$Survived)

print(resubstitution_accuracy(titanic_train,model3))
print(holdout_accuracy(titanic_train,10))
print(crossvalidate_accuracy(titanic_train,10))