library(caret)
setwd("E:/data analytics/kaggle/titanic/data")

titanic = read.csv("train.csv", header = TRUE, na.strings=c("NA",""))
dim(titanic)
str(titanic)

plot(titanic$SibSp, titanic$Parch)
cov(titanic$SibSp, titanic$Parch)
cor(titanic$SibSp, titanic$Parch)

plot(titanic$Fare, titanic$Parch)
cov(titanic$Fare, titanic$Parch)
cor(titanic$Fare, titanic$Parch)