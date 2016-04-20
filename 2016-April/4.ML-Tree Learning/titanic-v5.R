library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

setwd("C:\\Users\\Thimma Reddy\\Documents\\GitHub\\datascience\\2014\\kaggle\\titanic\\data")

titanic_train = read.csv("train.csv")
dim(titanic_train)
str(titanic_train)
titanic_train$Pclass = as.factor(titanic_train$Pclass)
titanic_train$Name = as.character(titanic_train$Name)
titanic_train$Survived = as.factor(titanic_train$Survived)
summary(titanic_train)

model = rpart(Survived ~ Pclass + Sex + Age + Fare, data = titanic_train, method="class")
model
plot(model)
text(model)

titanic_test = read.csv("test.csv")
dim(titanic_test)
str(titanic_test)

titanic_test$Pclass = as.factor(titanic_test$Pclass)
titanic_test$Name = as.character(titanic_test$Name)

titanic_test$Survived = predict(model, titanic_test , type="class")

result = titanic_test[,c("PassengerId","Survived")]
write.csv(result,"submission.csv",row.names = F, col.names = T )
